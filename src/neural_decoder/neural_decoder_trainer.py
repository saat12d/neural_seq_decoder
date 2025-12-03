import os
import pickle
import time
import json
import logging

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset

# Setup logging
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def ctc_greedy_decode(logits, T_eff):
    """
    Greedy CTC decode with blank=0, collapse repeats, drop blanks.
    logits: (B,T,C) or (T,C); T_eff: effective time length (after stride/kernel).
    """
    if logits.ndim == 3:
        logits = logits[0]
    path = logits[:T_eff].argmax(dim=-1).cpu().numpy().tolist()
    decoded, prev = [], None
    for p in path:
        if p != 0 and p != prev:
            decoded.append(p)
        prev = p
    return np.array(decoded, dtype=np.int32)


def getDatasetLoaders(
    datasetName,
    batchSize,
    args=None,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    # Get time masking parameters from args if available
    if args is None:
        args = {}
    time_mask_prob = args.get("time_mask_prob", 0.10)
    time_mask_width = args.get("time_mask_width", 20)
    time_mask_max_masks = args.get("time_mask_max_masks", 1)
    
    train_ds = SpeechDataset(
        loadedData["train"], 
        transform=None, 
        split="train",
        time_mask_prob=time_mask_prob,
        time_mask_width=time_mask_width,
        time_mask_max_masks=time_mask_max_masks
    )
    test_ds = SpeechDataset(loadedData["test"], split="test")

    # Speed optimization: Use multiple workers for data loading
    num_workers = args.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args["outputDir"])
    logger.info("=" * 80)
    logger.info("Starting training run")
    logger.info("=" * 80)
    
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"
    
    # Log training configuration
    logger.info(f"Output directory: {args['outputDir']}")
    logger.info(f"Dataset path: {args['datasetPath']}")
    logger.info(f"Batch size: {args['batchSize']}")
    logger.info(f"Total batches: {args['nBatch']}")
    logger.info(f"Seed: {args['seed']}")
    
    # Note: args will be saved again after nDays is determined

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        args,
    )

    # Save nDays to args for proper model loading later
    n_days = len(loadedData["train"])
    args["nDays"] = n_days
    
    # Log dataset statistics
    logger.info(f"Dataset loaded: {n_days} training days")
    logger.info(f"Training samples: {len(trainLoader.dataset)}")
    logger.info(f"Test samples: {len(testLoader.dataset)}")
    
    # Save args again now that nDays is included
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # Log model architecture
    logger.info("=" * 80)
    logger.info("Model Architecture")
    logger.info("=" * 80)
    logger.info(f"Input features: {args['nInputFeatures']}")
    logger.info(f"Hidden units: {args['nUnits']}")
    logger.info(f"GRU layers: {args['nLayers']}")
    logger.info(f"Output classes: {args['nClasses']} (+ 1 blank = {args['nClasses'] + 1})")
    logger.info(f"Days (per-day embeddings): {n_days}")
    logger.info(f"Dropout: {args['dropout']}")
    logger.info(f"Input dropout: {args.get('input_dropout', 0.0)}")
    logger.info(f"Layer norm: {args.get('use_layer_norm', False)}")
    logger.info(f"Bidirectional: {args['bidirectional']}")
    logger.info(f"Stride length: {args['strideLen']}, Kernel length: {args['kernelLen']}")
    
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=n_days,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        input_dropout=args.get("input_dropout", 0.0),
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # Speed optimization: Mixed precision training (FP16/BF16)
    # Prefer BF16 for better stability (works on Ampere+ GPUs)
    use_amp = args.get("use_amp", True)  # Enable by default
    if use_amp:
        # Check if BF16 is supported - test autocast directly
        dtype = torch.float16  # Default to FP16
        if torch.cuda.is_available():
            # Try to detect BF16 support by testing autocast
            try:
                # Check if torch has is_bf16_supported method (newer PyTorch)
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    # Test autocast with BF16 to ensure it works
                    try:
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            test_tensor = torch.randn(1, device=device)
                        dtype = torch.bfloat16
                        logger.info("Using mixed precision training with BF16 (more stable)")
                    except RuntimeError:
                        dtype = torch.float16
                        logger.info("BF16 autocast not supported, using FP16")
                else:
                    # Fallback: check device capability (Ampere 8.0+)
                    device_cap = torch.cuda.get_device_capability()
                    if device_cap[0] >= 8:
                        # Try autocast with BF16 to verify it works
                        try:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                test_tensor = torch.randn(1, device=device)
                            dtype = torch.bfloat16
                            logger.info("Using mixed precision training with BF16 (more stable)")
                        except RuntimeError:
                            dtype = torch.float16
                            logger.info("BF16 autocast not supported, using FP16")
                    else:
                        dtype = torch.float16
                        logger.info("Using mixed precision training with FP16 (device doesn't support BF16)")
            except Exception as e:
                dtype = torch.float16
                logger.info(f"BF16 check failed ({e}), using FP16")
        else:
            dtype = torch.float16
            logger.info("Using mixed precision training with FP16")
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        dtype = torch.float32
        scaler = None
        logger.info("Using full precision (FP32) training")
    
    # Use AdamW optimizer with weight_decay parameter
    optimizer_name = args.get("optimizer", "adamw").lower()
    weight_decay = args.get("weight_decay", args.get("l2_decay", 1e-4))
    
    # Check if resuming from checkpoint
    start_batch = args.get("start_batch", 0)  # Allow manual override
    resume_from = args.get("resume_from_checkpoint", None)
    if resume_from:
        checkpoint_path = os.path.join(args["outputDir"], resume_from)
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    start_batch = checkpoint.get("batch", start_batch)
                    logger.info(f"Loaded model and optimizer, resuming from batch {start_batch}")
                else:
                    # Assume it's just a state dict (most common case)
                    model.load_state_dict(checkpoint)
                    # Try to infer batch number from metrics.jsonl
                    metrics_file = os.path.join(args["outputDir"], "metrics.jsonl")
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, "r") as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = json.loads(lines[-1])
                                    start_batch = last_line.get("step", 0) + 1
                                    logger.info(f"Loaded model weights, resuming from batch {start_batch} (inferred from metrics)")
                        except Exception:
                            logger.info("Loaded model weights (could not infer batch number, starting from 0)")
                    else:
                        logger.info("Loaded model weights (starting from batch 0)")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
                logger.info("Starting training from scratch")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from scratch")
    
    # Initialize optimizer with peak_lr (will be scaled during warmup)
    peak_lr = args.get("peak_lr", args.get("lrStart", 0.002))
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            weight_decay=weight_decay,
        )
    else:
        # Fallback to Adam for backward compatibility
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=peak_lr,
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=weight_decay,
        )
    
    # If resuming, try to load optimizer state
    if resume_from and start_batch > 0:
        checkpoint_path = os.path.join(args["outputDir"], resume_from)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state")
        except Exception:
            logger.info("Could not load optimizer state, using fresh optimizer")
    
    # Run #5: Support constant LR or warmup → cosine decay
    warmup_steps = args.get("warmup_steps", 1000)
    cosine_steps = args.get("cosine_T_max", args["nBatch"] - warmup_steps)
    
    # Log optimizer and scheduler settings
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Optimizer: {optimizer_name.upper()}")
    logger.info(f"Peak LR: {peak_lr} (capped), End LR: {args['lrEnd']}")
    logger.info(f"Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Gradient clipping: max_norm=1.0")
    adaptive_lr_enabled = args.get("adaptive_lr", True)
    if adaptive_lr_enabled:
        logger.info(f"Adaptive LR: ENABLED (reduce by {args.get('lr_reduction_factor', 0.8):.1%} if grad_norm > {args.get('lr_reduction_threshold', 10.0)} or NaN detected)")
        logger.info(f"  Min LR: {args.get('min_lr', 0.0005)}, Max reductions: {args.get('max_lr_reductions', 5)}")
    else:
        logger.info(f"Adaptive LR: DISABLED")
    logger.info(f"Augmentation - White noise SD: {args.get('whiteNoiseSD', 0.0)}")
    logger.info(f"Augmentation - Constant offset SD: {args.get('constantOffsetSD', 0.0)}")
    logger.info(f"Time masking - Prob: {args.get('time_mask_prob', 0.0)}, Width: {args.get('time_mask_width', 0)}, Max masks: {args.get('time_mask_max_masks', 1)}")
    
    # Check if using constant LR (no warmup, no decay)
    if warmup_steps == 0 and abs(peak_lr - args["lrEnd"]) < 1e-6:
        # Constant LR - use a simple scheduler that does nothing
        logger.info("Using constant LR (no warmup, no decay)")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,  # Always return 1.0 (no change)
        )
    else:
        # Use LambdaLR for warmup (more reliable than LinearLR with SequentialLR)
        # Warmup: linearly increase from 0 to peak_lr over warmup_steps
        def warmup_lambda(step):
            if step >= warmup_steps:
                return 1.0
            return float(step) / float(max(1, warmup_steps))
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda,
        )
        
        # Cosine decay: decay from peak_lr to lrEnd over remaining steps
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=args["lrEnd"],
        )
        
        # Chain schedulers: warmup first, then cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)

    # --train--
    testLoss = []
    testPER = []
    startTime = time.time()
    
    # Run #5: Track moving average PER (over last 200 steps) and best actual PER
    moving_avg_window = 200
    per_history = []
    best_per_ma = float('inf')
    best_per = float('inf')  # Track best actual PER for checkpoint saving
    
    # Track gradient norms for debugging
    grad_norms = []
    loss_history = []
    
    # Track skipped updates for debugging
    skipped_updates = 0
    skipped_reasons = {"loss_nan": 0, "grad_nan": 0, "grad_norm_nan": 0}
    
    # Run #6: Adaptive LR reduction
    adaptive_lr_enabled = args.get("adaptive_lr", True)
    lr_reduction_factor = args.get("lr_reduction_factor", 0.8)  # Reduce by 20%
    lr_reduction_threshold = args.get("lr_reduction_threshold", 10.0)  # Reduce if grad_norm > this
    min_lr = args.get("min_lr", 0.0005)  # Don't go below this
    lr_reductions = 0
    max_lr_reductions = args.get("max_lr_reductions", 5)  # Max number of reductions
    recent_skipped_window = 50  # Check skipped updates over last N steps
    recent_skipped_history = []  # Track skipped updates in recent window
    
    # Metrics logging file
    metrics_file = os.path.join(args["outputDir"], "metrics.jsonl")
    
    # Speed optimization: Use proper iterator instead of next(iter()) each time
    train_iter = iter(trainLoader)
    
    for batch in range(start_batch, args["nBatch"]):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            # Recreate iterator when exhausted
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)
        
        # Speed optimization: Use non_blocking transfers
        X, y, X_len, y_len, dayIdx = (
            X.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            X_len.to(device, non_blocking=True),
            y_len.to(device, non_blocking=True),
            dayIdx.to(device, non_blocking=True),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device, dtype=X.dtype) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device, dtype=X.dtype)
                * args["constantOffsetSD"]
            )

        # Speed optimization: Use mixed precision for forward pass
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            # Compute prediction error
            pred = model.forward(X, dayIdx)

            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                y_len,
            )
            # CTCLoss with reduction="mean" already returns a scalar, no need for torch.sum()

        # Backpropagation with mixed precision
        # Run #4: Hard guard - check for NaN/Inf in loss before proceeding
        if not torch.isfinite(loss):
            skipped_updates += 1
            skipped_reasons["loss_nan"] += 1
            logger.warning(f"Step {batch}: Loss is {loss.item()}, skipping update (total skipped: {skipped_updates})")
            # Run #6: Check adaptive LR reduction even on skip
            if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr > min_lr:
                    new_lr = max(current_lr * lr_reduction_factor, min_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    lr_reductions += 1
                    logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf loss detected")
                    logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
            scheduler.step()  # Still step scheduler to maintain schedule
            # Track metrics even on skip
            loss_history.append(loss.item())
            grad_norms.append(float('inf'))  # Sentinel for skipped update
            continue
        
        optimizer.zero_grad()
        has_nan_grad = False  # Track NaN gradients for adaptive LR
        grad_norm = None  # Initialize grad_norm (will be set after clipping)
        if scaler is not None:
            scaler.scale(loss).backward()
            # Run #3: Strong gradient clipping to prevent loss spikes
            scaler.unscale_(optimizer)
            
            # Check for NaN/Inf gradients before clipping
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                skipped_updates += 1
                skipped_reasons["grad_nan"] += 1
                logger.warning(f"Step {batch}: NaN/Inf gradients detected, skipping update (total skipped: {skipped_updates})")
                # Run #6: Check adaptive LR reduction
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradients detected")
                        logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                # Zero grads to prevent accumulation
                optimizer.zero_grad()
                scaler.update()  # Update scaler state even on skip
                scheduler.step()
                # Track metrics even on skip
                loss_history.append(loss.item())
                grad_norms.append(float('inf'))  # Sentinel for skipped update
                continue
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check if grad_norm is NaN/Inf after clipping
            if not torch.isfinite(grad_norm):
                skipped_updates += 1
                skipped_reasons["grad_norm_nan"] += 1
                logger.warning(f"Step {batch}: Gradient norm is {grad_norm.item()}, skipping update (total skipped: {skipped_updates})")
                # Run #6: Check adaptive LR reduction
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradient norm detected")
                        logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                # Zero grads to prevent accumulation
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
                # Track metrics even on skip
                loss_history.append(loss.item())
                grad_norms.append(float('inf'))  # Sentinel for skipped update
                continue
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Check for NaN/Inf gradients before clipping
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                skipped_updates += 1
                skipped_reasons["grad_nan"] += 1
                logger.warning(f"Step {batch}: NaN/Inf gradients detected, skipping update (total skipped: {skipped_updates})")
                # Run #6: Check adaptive LR reduction
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradients detected")
                        logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                optimizer.zero_grad()  # Clear gradients
                scheduler.step()
                # Track metrics even on skip
                loss_history.append(loss.item())
                grad_norms.append(float('inf'))  # Sentinel for skipped update
                continue
            
            # Run #3: Strong gradient clipping to prevent loss spikes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check if grad_norm is NaN/Inf after clipping
            if not torch.isfinite(grad_norm):
                skipped_updates += 1
                skipped_reasons["grad_norm_nan"] += 1
                logger.warning(f"Step {batch}: Gradient norm is {grad_norm.item()}, skipping update (total skipped: {skipped_updates})")
                # Run #6: Check adaptive LR reduction
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradient norm detected")
                        logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                optimizer.zero_grad()  # Clear gradients
                scheduler.step()
                # Track metrics even on skip
                loss_history.append(loss.item())
                grad_norms.append(float('inf'))  # Sentinel for skipped update
                continue
            
            optimizer.step()
        scheduler.step()  # Step scheduler after optimizer
        
        # Track metrics for debugging
        loss_history.append(loss.item())
        # Only track grad_norm if it's finite (skip if update was skipped)
        if torch.isfinite(grad_norm):
            grad_norms.append(grad_norm.item())
        else:
            # Use a sentinel value to indicate skipped update
            grad_norms.append(float('inf'))
        
        # Run #6: Adaptive LR reduction - check if we need to reduce LR
        if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
            current_lr = optimizer.param_groups[0]['lr']
            should_reduce = False
            reduction_reason = ""
            
            # Track recent skipped updates
            if skipped_updates > 0:
                recent_skipped_history.append(1)
            else:
                recent_skipped_history.append(0)
            if len(recent_skipped_history) > recent_skipped_window:
                recent_skipped_history.pop(0)
            
            # Condition 1: NaN/Inf gradients detected in this step
            if has_nan_grad or (grad_norm is not None and not torch.isfinite(grad_norm)):
                should_reduce = True
                reduction_reason = "NaN/Inf gradients detected"
            
            # Condition 2: High gradient norm (even after clipping)
            elif grad_norm is not None and torch.isfinite(grad_norm) and grad_norm.item() > lr_reduction_threshold:
                should_reduce = True
                reduction_reason = f"High gradient norm ({grad_norm.item():.2f} > {lr_reduction_threshold})"
            
            # Condition 3: Too many skipped updates in recent window
            elif len(recent_skipped_history) >= recent_skipped_window:
                recent_skipped_count = sum(recent_skipped_history)
                if recent_skipped_count >= 3:  # 3+ skipped in last 50 steps
                    should_reduce = True
                    reduction_reason = f"Too many skipped updates ({recent_skipped_count} in last {recent_skipped_window} steps)"
            
            # Reduce LR if conditions met
            if should_reduce and current_lr > min_lr:
                new_lr = max(current_lr * lr_reduction_factor, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                lr_reductions += 1
                logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: {reduction_reason}")
                logger.warning(f"   LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                # Reset recent skipped history after reduction
                recent_skipped_history = []

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                        X_len.to(device, non_blocking=True),
                        y_len.to(device, non_blocking=True),
                        testDayIdx.to(device, non_blocking=True),
                    )

                    # Use mixed precision for eval too (faster, minimal accuracy impact)
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                        pred = model.forward(X, testDayIdx)
                        loss = loss_ctc(
                            torch.permute(pred.log_softmax(2), [1, 0, 2]),
                            y,
                            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                            y_len,
                        )
                        # CTCLoss with reduction="mean" already returns a scalar, no need for torch.sum()
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        # Use proper CTC greedy decode (same as eval.py)
                        T_eff = int(adjustedLens[iterIdx].item())
                        decodedSeq = ctc_greedy_decode(pred[iterIdx], T_eff)

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                per = total_edit_distance / total_seq_length

                # Run #3: Update moving average PER
                per_history.append(per)
                if len(per_history) > moving_avg_window:
                    per_history.pop(0)
                per_ma = np.mean(per_history) if per_history else per

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100
                current_lr = scheduler.get_last_lr()[0]
                
                # Compute gradient and loss statistics
                recent_grad_norms = grad_norms[-100:] if len(grad_norms) >= 100 else grad_norms
                recent_losses = loss_history[-100:] if len(loss_history) >= 100 else loss_history
                # Filter out Infinity values for statistics
                finite_grad_norms = [g for g in recent_grad_norms if np.isfinite(g)]
                avg_grad_norm = np.mean(finite_grad_norms) if finite_grad_norms else 0.0
                avg_train_loss = np.mean(recent_losses) if recent_losses else 0.0
                max_grad_norm = np.max(finite_grad_norms) if finite_grad_norms else 0.0
                if not finite_grad_norms and recent_grad_norms:
                    max_grad_norm = float('inf')  # All were Infinity
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
                else:
                    memory_allocated = memory_reserved = 0.0
                
                # Log detailed metrics
                log_msg = (
                    f"batch {batch:5d} | "
                    f"loss: {avgDayLoss:>7.4f} (train: {avg_train_loss:>7.4f}) | "
                    f"per: {per:>7.4f} (ma: {per_ma:>7.4f}) | "
                    f"grad_norm: {avg_grad_norm:>6.4f} (max: {max_grad_norm:>6.4f}) | "
                    f"lr: {current_lr:.6f} | "
                    f"skipped: {skipped_updates} | "
                    f"time: {time_per_batch:>5.2f}s | "
                    f"mem: {memory_allocated:>4.2f}GB/{memory_reserved:>4.2f}GB"
                )
                logger.info(log_msg)
                print(log_msg)  # Also print to console
                
                # Log skipped update breakdown if any
                if skipped_updates > 0:
                    logger.info(f"  Skipped updates breakdown: {skipped_reasons}")
                
                # Log metrics to JSONL file
                metrics_entry = {
                    "step": batch,
                    "ctc_loss": float(avgDayLoss),
                    "train_loss_avg": float(avg_train_loss),
                    "per": float(per),
                    "per_ma": float(per_ma),
                    "grad_norm_avg": float(avg_grad_norm),
                    "grad_norm_max": float(max_grad_norm),
                    "lr": float(current_lr),
                    "lr_reductions": int(lr_reductions),
                    "skipped_updates": int(skipped_updates),
                    "skipped_reasons": skipped_reasons.copy(),
                    "time_per_batch": float(time_per_batch),
                    "memory_allocated_gb": float(memory_allocated),
                    "memory_reserved_gb": float(memory_reserved),
                }
                
                # Log sample predictions occasionally for debugging (first eval and every 1000 steps)
                if batch == 0 or batch % 1000 == 0:
                    # Get a sample from the test set
                    sample_X, sample_y, sample_X_len, sample_y_len, sample_dayIdx = next(iter(testLoader))
                    sample_X = sample_X[:1].to(device, non_blocking=True)
                    sample_y = sample_y[:1].to(device, non_blocking=True)
                    sample_X_len = sample_X_len[:1].to(device, non_blocking=True)
                    sample_y_len = sample_y_len[:1].to(device, non_blocking=True)
                    sample_dayIdx = sample_dayIdx[:1].to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                        sample_pred = model.forward(sample_X, sample_dayIdx)
                    
                    sample_T_eff = int(((sample_X_len[0] - model.kernelLen) / model.strideLen).item())
                    sample_decoded = ctc_greedy_decode(sample_pred[0], sample_T_eff)
                    sample_target = sample_y[0, :sample_y_len[0]].cpu().numpy()
                    
                    logger.info(f"Sample prediction (step {batch}):")
                    logger.info(f"  Target length: {len(sample_target)}, Pred length: {len(sample_decoded)}")
                    logger.info(f"  Target IDs (first 20): {sample_target[:20].tolist()}")
                    logger.info(f"  Pred IDs (first 20): {sample_decoded[:20].tolist()}")
                    matcher = SequenceMatcher(a=sample_target.tolist(), b=sample_decoded.tolist())
                    sample_per = matcher.distance() / max(1, len(sample_target))
                    logger.info(f"  Sample PER: {sample_per:.4f}")
                    
                    metrics_entry["sample_per"] = float(sample_per)
                    metrics_entry["sample_target_len"] = int(len(sample_target))
                    metrics_entry["sample_pred_len"] = int(len(sample_decoded))
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics_entry) + "\n")
                
                startTime = time.time()

            # Run #5: Save best checkpoint based on actual PER (not moving average)
            # Moving average can improve even as actual PER degrades, so use actual PER
            if per < best_per:
                best_per = per
                best_per_ma = per_ma  # Also track best MA for logging
                # Save best model checkpoint
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "best_model.pt"))
                logger.info(f"✓ New best checkpoint saved (PER: {per:.4f}, PER_MA: {per_ma:.4f})")
            elif per_ma < best_per_ma:
                # Still track best moving average for logging, but don't save checkpoint
                best_per_ma = per_ma
            testLoss.append(avgDayLoss)
            testPER.append(per)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testPER"] = np.array(testPER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
    
    # Save final model checkpoint
    torch.save(model.state_dict(), os.path.join(args["outputDir"], "final_model.pt"))
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best PER (actual): {best_per:.4f}")
    logger.info(f"Best PER (moving avg): {best_per_ma:.4f}")
    logger.info(f"Final PER: {testPER[-1]:.4f}" if testPER else "N/A")
    logger.info(f"Total skipped updates: {skipped_updates}")
    logger.info(f"Skipped reasons: {skipped_reasons}")
    if adaptive_lr_enabled:
        logger.info(f"Adaptive LR reductions: {lr_reductions}")
        final_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Final LR: {final_lr:.6f} (started at {peak_lr:.6f})")
    logger.info("=" * 80)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    # Get nDays from saved args, or load from dataset if not present
    n_days = args.get("nDays", None)
    if n_days is None:
        # Fallback: load from dataset
        with open(args["datasetPath"], "rb") as f:
            loaded_data = pickle.load(f)
            n_days = len(loaded_data["train"])
    # Use nInputLayers parameter if provided (for backward compatibility)
    if nInputLayers != 24:
        n_days = nInputLayers

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=n_days,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        input_dropout=args.get("input_dropout", 0.0),
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()