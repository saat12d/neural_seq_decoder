# neural_decoder_trainer.py
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


# ---------------------------
# Logging
# ---------------------------
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------------------------
# Decoders
# ---------------------------
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


def ctc_prefix_beam_search(log_probs, beam_size=20, blank=0):
    """
    Prefix beam search for CTC decoding with proper p_b / p_nb bookkeeping.
    log_probs: (T, C) torch.Tensor or np.ndarray of log-probs.
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()

    T, C = log_probs.shape
    p_b = {(): 0.0}
    p_nb = {(): -np.inf}

    for t in range(T):
        p_b_new, p_nb_new = {}, {}
        prefixes = set(p_b.keys()) | set(p_nb.keys())

        for prefix in prefixes:
            pb = p_b.get(prefix, -np.inf)
            pnb = p_nb.get(prefix, -np.inf)

            # blank transition
            p_blank = log_probs[t, blank]
            prev = p_b_new.get(prefix, -np.inf)
            p_b_new[prefix] = np.logaddexp(prev, np.logaddexp(pb + p_blank, pnb + p_blank))

            # non-blank transitions
            for c in range(C):
                if c == blank:
                    continue
                p_c = log_probs[t, c]
                if len(prefix) > 0 and prefix[-1] == c:
                    prev_pnb_new = p_nb_new.get(prefix, -np.inf)
                    p_nb_new[prefix] = np.logaddexp(prev_pnb_new, pb + p_c)
                else:
                    new_prefix = prefix + (c,)
                    prev_pnb_new = p_nb_new.get(new_prefix, -np.inf)
                    p_nb_new[new_prefix] = np.logaddexp(prev_pnb_new, np.logaddexp(pb + p_c, pnb + p_c))

        # prune to beam
        all_prefixes = set(p_b_new.keys()) | set(p_nb_new.keys())
        scores = []
        for prefix in all_prefixes:
            pb = p_b_new.get(prefix, -np.inf)
            pnb = p_nb_new.get(prefix, -np.inf)
            scores.append((np.logaddexp(pb, pnb), prefix))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:beam_size]

        p_b = {prefix: p_b_new.get(prefix, -np.inf) for _, prefix in top}
        p_nb = {prefix: p_nb_new.get(prefix, -np.inf) for _, prefix in top}

    # best
    best_prefix, best_score = (), -np.inf
    for prefix in set(p_b.keys()) | set(p_nb.keys()):
        pb = p_b.get(prefix, -np.inf)
        pnb = p_nb.get(prefix, -np.inf)
        total = np.logaddexp(pb, pnb)
        if total > best_score:
            best_score, best_prefix = total, prefix

    return np.array(best_prefix, dtype=np.int32)


# ---------------------------
# Data loading
# ---------------------------
def getDatasetLoaders(datasetName, batchSize, args=None):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (X_padded, y_padded, torch.stack(X_lens), torch.stack(y_lens), torch.stack(days))

    if args is None:
        args = {}
    time_mask_prob = args.get("time_mask_prob", 0.10)
    time_mask_width = args.get("time_mask_width", 20)
    time_mask_max_masks = args.get("time_mask_max_masks", 1)
    freq_mask_prob = args.get("freq_mask_prob", 0.10)
    freq_mask_width = args.get("freq_mask_width", 12)
    freq_mask_max_masks = args.get("freq_mask_max_masks", 2)

    train_ds = SpeechDataset(
        loadedData["train"],
        transform=None,
        split="train",
        time_mask_prob=time_mask_prob,
        time_mask_width=time_mask_width,
        time_mask_max_masks=time_mask_max_masks,
        freq_mask_prob=freq_mask_prob,
        freq_mask_width=freq_mask_width,
        freq_mask_max_masks=freq_mask_max_masks,
    )
    test_ds = SpeechDataset(loadedData["test"], split="test")

    # DataLoader micro-tuning
    num_workers = int(args.get("num_workers", 8))
    prefetch_factor = int(args.get("prefetch_factor", 2))
    pin_dev_kw = {}
    try:
        # PyTorch 2.0+: keep it best-effort
        pin_dev_kw["pin_memory_device"] = "cuda"
    except TypeError:
        pin_dev_kw = {}

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        collate_fn=_padding,
        **pin_dev_kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        collate_fn=_padding,
        **pin_dev_kw,
    )

    return train_loader, test_loader, loadedData


# ---------------------------
# Trainer
# ---------------------------
def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)

    # Less fragmentation; fewer OOMs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")

    logger = setup_logging(args["outputDir"])

    # TF32 where available (Ampere+). T4 will just ignore.
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("Enabled TF32 for faster FP32 matmuls (Ampere+ GPUs)")
        except AttributeError:
            logger.info("TF32 not available")

    logger.info("=" * 80)
    logger.info("Starting training run")
    logger.info("=" * 80)

    if "run_number" in args:
        logger.info(f"Run Number: {args['run_number']}")
    if "run_name" in args:
        logger.info(f"Run Name: {args['run_name']}")
    if "run_purpose" in args:
        logger.info(f"Run Purpose: {args['run_purpose']}")

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Output directory: {args['outputDir']}")
    logger.info(f"Dataset path: {args['datasetPath']}")
    logger.info(f"Batch size: {args['batchSize']}")
    logger.info(f"Total batches: {args['nBatch']}")
    logger.info(f"Seed: {args['seed']}")

    trainLoader, testLoader, loadedData = getDatasetLoaders(args["datasetPath"], args["batchSize"], args)

    # nDays in args (for load)
    n_days = len(loadedData["train"])
    args["nDays"] = n_days

    logger.info(f"Dataset loaded: {n_days} training days")
    logger.info(f"Training samples: {len(trainLoader.dataset)}")
    logger.info(f"Test samples: {len(testLoader.dataset)}")

    with open(os.path.join(args["outputDir"], "args"), "wb") as f:
        pickle.dump(args, f)

    # Model
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

    torch.backends.cudnn.benchmark = True
    logger.info("Enabled cuDNN benchmark for faster training")

    # Optional compile (off by default for T4 + GRU)
    if args.get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            import logging as pylog

            pylog.getLogger("torch._dynamo").setLevel(pylog.WARNING)
            pylog.getLogger("torch._inductor").setLevel(pylog.WARNING)

            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
            logger.info(f"Compiling model with torch.compile for {dev_name}...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("✓ Model compilation complete")
        except Exception as e:
            logger.warning(f"torch.compile failed ({e}), continuing without compilation")
    else:
        logger.info("torch.compile disabled (recommended off on T4 + GRU)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    use_beam_search = args.get("use_beam_search", False)
    beam_size = args.get("beam_size", 20)

    # CTC sanity checks
    logger.info("=" * 80)
    logger.info("CTC Sanity Checks")
    logger.info("=" * 80)
    assert loss_ctc.blank == 0, f"CTCLoss blank must be 0, got {loss_ctc.blank}"
    logger.info(f"✓ CTCLoss blank index: {loss_ctc.blank}")

    with torch.no_grad():
        Xs, ys, Xls, yls, ds = next(iter(trainLoader))
        Xs, ys, Xls, yls, ds = (Xs.to(device), ys.to(device), Xls.to(device), yls.to(device), ds.to(device))
        T_eff_s = ((Xls - args["kernelLen"]) // args["strideLen"]).clamp(min=1)
        logger.info(f"✓ T_eff calculation verified (min={T_eff_s.min().item()}, max={T_eff_s.max().item()})")

        has_blank = False
        for b in range(ys.size(0)):
            Lb = int(yls[b].item())
            if Lb <= 0:
                continue
            valid = ys[b, :Lb]
            if (valid == 0).any().item():
                has_blank = True
                break
        if has_blank:
            logger.warning("⚠️  Found blank (0) in valid labels; CTC expects labels >= 1 when blank=0.")
        else:
            logger.info("✓ Labels verified: no blanks found in valid label spans (all labels >= 1)")

        logger.info(f"✓ Input lengths: min={Xs.size(1)}, max={Xs.size(1)}")
        logger.info(f"✓ Target lengths: min={yls.min().item()}, max={yls.max().item()}")
        logger.info(f"✓ T_eff lengths: min={T_eff_s.min().item()}, max={T_eff_s.max().item()}")

    logger.info("=" * 80)

    # AMP setup
    use_amp = args.get("use_amp", True)
    auto_enable_amp = args.get("auto_enable_amp", False)
    auto_amp_step = args.get("auto_amp_step", 1500)
    auto_amp_per_threshold = args.get("auto_amp_per_threshold", 0.45)
    amp_auto_enabled = False

    if use_amp:
        dtype = torch.float16
        if torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    try:
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            _ = torch.randn(1, device=device)
                        dtype = torch.bfloat16
                        logger.info("Using mixed precision BF16 (Ampere+)")
                    except RuntimeError:
                        dtype = torch.float16
                        logger.info("BF16 autocast not supported; using FP16")
                else:
                    dtype = torch.float16
                    logger.info("Using mixed precision FP16 (T4 etc.)")
            except Exception:
                dtype = torch.float16
                logger.info("BF16 check failed; using FP16")
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
        logger.info("  Safety: log_softmax + CTCLoss computed in FP32")
    else:
        dtype = torch.float32
        scaler = None
        logger.info("Using full precision (FP32) training")

    # Optimizer
    optimizer_name = args.get("optimizer", "adamw").lower()
    weight_decay = args.get("weight_decay", args.get("l2_decay", 1e-4))
    peak_lr = args.get("peak_lr", args.get("lrStart", 0.002))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=peak_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
        )

    # Resume (optional)
    start_batch = args.get("start_batch", 0)
    resume_from = args.get("resume_from_checkpoint", None)
    if resume_from:
        ckpt_path = os.path.join(args["outputDir"], resume_from)
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    start_batch = checkpoint.get("batch", start_batch)
                    logger.info(f"Loaded model and optimizer, resuming from batch {start_batch}")
                else:
                    model.load_state_dict(checkpoint)
                    mfile = os.path.join(args["outputDir"], "metrics.jsonl")
                    if os.path.exists(mfile):
                        try:
                            with open(mfile, "r") as f:
                                lines = f.readlines()
                                if lines:
                                    last = json.loads(lines[-1])
                                    start_batch = last.get("step", 0) + 1
                                    logger.info(f"Loaded weights; inferred start batch {start_batch}")
                        except Exception:
                            logger.info("Loaded weights; starting from 0 (inference failed)")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
                logger.info("Starting training from scratch")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}; starting from scratch")

    # Scheduler config logs
    warmup_steps = args.get("warmup_steps", 1000)
    cosine_steps = args.get("cosine_T_max", args["nBatch"] - warmup_steps)
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Optimizer: {optimizer_name.upper()}")
    logger.info(f"Peak LR: {peak_lr} (capped), End LR: {args['lrEnd']}")
    logger.info(f"Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}")
    logger.info(f"Weight decay: {weight_decay}")
    grad_clip_norm = args.get("grad_clip_norm", 1.0)
    logger.info(f"Gradient clipping: max_norm={grad_clip_norm}")

    adaptive_lr_enabled = args.get("adaptive_lr", True)
    if adaptive_lr_enabled:
        logger.info(
            f"Adaptive LR: ENABLED (reduce by {args.get('lr_reduction_factor', 0.8):.1%} "
            f"if grad issues; min LR {args.get('min_lr', 0.0005)})"
        )
    else:
        logger.info("Adaptive LR: DISABLED")

    logger.info(f"Augmentation - White noise SD: {args.get('whiteNoiseSD', 0.0)}")
    logger.info(f"Augmentation - Constant offset SD: {args.get('constantOffsetSD', 0.0)}")
    logger.info(
        f"Time masking - Prob: {args.get('time_mask_prob', 0.0)}, Width: {args.get('time_mask_width', 0)}, "
        f"Max masks: {args.get('time_mask_max_masks', 1)}"
    )
    logger.info(
        f"Frequency masking - Prob: {args.get('freq_mask_prob', 0.0)}, Width: {args.get('freq_mask_width', 0)}, "
        f"Max masks: {args.get('freq_mask_max_masks', 0)}"
    )
    logger.info("CTC Decoding: " + ("Prefix beam search" if use_beam_search else "Greedy"))

    # Early stop guard vars
    early_stop_guard = args.get("early_stop_guard", False)
    early_stop_step = args.get("early_stop_step", 1200)
    early_stop_per_threshold = args.get("early_stop_per_threshold", 0.38)
    early_stop_reduced_max_lr = args.get("early_stop_reduced_max_lr", 0.003)
    early_stop_triggered = False
    original_max_lr = None

    # Scheduler selection
    use_plateau_scheduler = args.get("use_plateau_scheduler", False)
    if use_plateau_scheduler:
        logger.info("Using ReduceLROnPlateau on validation PER")
        plateau_patience = args.get("plateau_patience", 8)
        plateau_factor = args.get("plateau_factor", 0.5)
        plateau_min_lr = args.get("plateau_min_lr", 1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr, verbose=True
        )
        logger.info(f"  Patience: {plateau_patience}, Factor: {plateau_factor}, Min LR: {plateau_min_lr}")
    elif args.get("use_onecycle", False):
        import math

        max_lr = args.get("onecycle_max_lr", peak_lr)
        pct_start = args.get("onecycle_pct_start", 0.1)
        div_factor = args.get("onecycle_div_factor", 25.0)
        final_div_factor = args.get("onecycle_final_div_factor", 1e3)

        original_max_lr = max_lr

        # CRITICAL: match scheduler steps to optimizer steps with grad accumulation
        accumulation_steps = int(args.get("gradient_accumulation_steps", 1))
        effective_steps = max(1, math.ceil(args["nBatch"] / accumulation_steps))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=effective_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        logger.info("Using OneCycleLR scheduler:")
        logger.info(f"  Max LR: {max_lr}, Start LR: {max_lr/div_factor:.6f}, End LR: {max_lr/final_div_factor:.6f}")
        logger.info(f"  Warmup: {int(effective_steps * pct_start)} steps ({pct_start*100:.1f}%)")
        if early_stop_guard:
            logger.info(
                f"  Early stop guard: reduce max_lr to {early_stop_reduced_max_lr} if PER > {early_stop_per_threshold} "
                f"at step {early_stop_step}"
            )
    elif warmup_steps == 0 and abs(peak_lr - args["lrEnd"]) < 1e-6:
        logger.info("Using constant LR (no warmup, no decay)")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        # warmup + cosine
        def warmup_lambda(step):
            if step >= warmup_steps:
                return 1.0
            return float(step) / float(max(1, warmup_steps))

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=args["lrEnd"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)

    # Tracking
    testLoss, testPER = [], []
    startTime = time.time()
    moving_avg_window = 200
    per_history = []
    best_per_ma = float("inf")
    best_per = float("inf")
    grad_norms, loss_history = [], []
    skipped_updates = 0
    skipped_reasons = {"loss_nan": 0, "grad_nan": 0, "grad_norm_nan": 0}

    adaptive_lr_enabled = args.get("adaptive_lr", True)
    lr_reduction_factor = args.get("lr_reduction_factor", 0.8)
    lr_reduction_threshold = args.get("lr_reduction_threshold", 10.0)
    min_lr = args.get("min_lr", 0.0005)
    lr_reductions = 0
    max_lr_reductions = args.get("max_lr_reductions", 5)
    adaptive_lr_warmup = args.get("adaptive_lr_warmup", 500)
    recent_skipped_window = 50
    recent_skipped_history = []

    metrics_file = os.path.join(args["outputDir"], "metrics.jsonl")

    beam_search_eval = args.get("beam_search_eval", False)
    beam_search_interval = args.get("beam_search_interval", 500)
    beam_search_subset_size = args.get("beam_search_subset_size", 100)
    beam_search_beam_size = args.get("beam_size", 10)

    accumulation_steps = args.get("gradient_accumulation_steps", 1)
    effective_batch_size = args["batchSize"] * accumulation_steps
    if accumulation_steps > 1:
        logger.info(f"Gradient accumulation: {accumulation_steps} steps (effective batch size: {effective_batch_size})")

    train_iter = iter(trainLoader)

    for batch in range(start_batch, args["nBatch"]):
        model.train()
        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            X_len.to(device, non_blocking=True),
            y_len.to(device, non_blocking=True),
            dayIdx.to(device, non_blocking=True),
        )

        # GPU noise
        if args.get("whiteNoiseSD", 0) > 0:
            X += torch.randn(X.shape, device=device, dtype=X.dtype) * args["whiteNoiseSD"]
        if args.get("constantOffsetSD", 0) > 0:
            X += torch.randn([X.shape[0], 1, X.shape[2]], device=device, dtype=X.dtype) * args["constantOffsetSD"]

        T_eff = ((X_len - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)

        # Forward
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            pred = model.forward(X, dayIdx, lengths=T_eff)

        log_probs = pred.float().log_softmax(2)  # FP32 safety
        loss = loss_ctc(torch.permute(log_probs, [1, 0, 2]), y, T_eff, y_len)

        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        # Guard NaN
        if not torch.isfinite(loss):
            skipped_updates += 1
            skipped_reasons["loss_nan"] += 1
            if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr > min_lr:
                    new_lr = max(current_lr * lr_reduction_factor, min_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    lr_reductions += 1
                    logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf loss")
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            loss_history.append(float(loss.item()))
            grad_norms.append(float("inf"))
            continue

        if batch % accumulation_steps == 0:
            optimizer.zero_grad()

        has_nan_grad = False
        grad_norm = None

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                skipped_updates += 1
                skipped_reasons["grad_nan"] += 1
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradients (step {batch})")
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
                loss_history.append(float(loss.item()))
                grad_norms.append(float("inf"))
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            if not torch.isfinite(grad_norm):
                skipped_updates += 1
                skipped_reasons["grad_norm_nan"] += 1
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf grad norm")
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
                loss_history.append(float(loss.item()))
                grad_norms.append(float("inf"))
                continue

            if (batch + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                skipped_updates += 1
                skipped_reasons["grad_nan"] += 1
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf gradients (step {batch})")
                optimizer.zero_grad()
                scheduler.step()
                loss_history.append(float(loss.item()))
                grad_norms.append(float("inf"))
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            if not torch.isfinite(grad_norm):
                skipped_updates += 1
                skipped_reasons["grad_norm_nan"] += 1
                if adaptive_lr_enabled and lr_reductions < max_lr_reductions:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_reduction_factor, min_lr)
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr
                        lr_reductions += 1
                        logger.warning(f"⚠️  ADAPTIVE LR REDUCTION #{lr_reductions}: NaN/Inf grad norm")
                optimizer.zero_grad()
                scheduler.step()
                loss_history.append(float(loss.item()))
                grad_norms.append(float("inf"))
                continue

            if (batch + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Step step-based schedulers only when optimizer stepped
        if (batch + 1) % accumulation_steps == 0:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        # Auto-enable AMP (if configured off initially)
        if auto_enable_amp and not use_amp and not amp_auto_enabled and batch == auto_amp_step:
            if per_history:
                current_per = per_history[-1]
                if current_per < auto_amp_per_threshold:
                    logger.info(f"🚀 Auto-enabling AMP at step {batch} (PER: {current_per:.4f} < {auto_amp_per_threshold})")
                    use_amp = True
                    amp_auto_enabled = True
                    if torch.cuda.is_available():
                        try:
                            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                    _ = torch.randn(1, device=device)
                                dtype = torch.bfloat16
                                logger.info("  Using BF16 (more stable)")
                            else:
                                dtype = torch.float16
                                logger.info("  Using FP16")
                        except Exception:
                            dtype = torch.float16
                            logger.info("  Using FP16 (BF16 not available)")
                    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
                else:
                    logger.info(f"  Skipping AMP auto-enable (PER: {current_per:.4f} >= {auto_amp_per_threshold})")

        loss_history.append(float(loss.item()))
        if torch.isfinite(grad_norm):
            grad_norms.append(float(grad_norm.item()))
        else:
            grad_norms.append(float("inf"))

        # --- EVAL ------------------------------------------------------------
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0

                for Xb, yb, Xlb, ylb, ddb in testLoader:
                    Xb, yb, Xlb, ylb, ddb = (
                        Xb.to(device, non_blocking=True),
                        yb.to(device, non_blocking=True),
                        Xlb.to(device, non_blocking=True),
                        ylb.to(device, non_blocking=True),
                        ddb.to(device, non_blocking=True),
                    )
                    T_eff_b = ((Xlb - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)

                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                        pred_b = model.forward(Xb, ddb, lengths=T_eff_b)

                    lp_b = pred_b.float().log_softmax(2)
                    loss_b = loss_ctc(torch.permute(lp_b, [1, 0, 2]), yb, T_eff_b, ylb)
                    allLoss.append(loss_b.cpu().detach().numpy())

                    adjusted = ((Xlb - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)
                    for i in range(pred_b.shape[0]):
                        Te = int(adjusted[i].item())
                        if use_beam_search:
                            lp = pred_b[i, :Te].log_softmax(dim=-1)
                            decoded = ctc_prefix_beam_search(lp, beam_size=beam_size, blank=0)
                        else:
                            decoded = ctc_greedy_decode(pred_b[i], Te)

                        trueSeq = np.array(yb[i][0: ylb[i]].cpu().detach())
                        m = SequenceMatcher(a=trueSeq.tolist(), b=decoded.tolist())
                        total_edit_distance += m.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = float(np.sum(allLoss) / max(1, len(testLoader)))
                per = (total_edit_distance / max(1, total_seq_length)) if total_seq_length > 0 else 1.0

                # moving avg PER
                per_history.append(per)
                if len(per_history) > moving_avg_window:
                    per_history.pop(0)
                per_ma = float(np.mean(per_history)) if per_history else per

                # Early stop guard (OneCycle)
                if (
                    args.get("use_onecycle", False)
                    and early_stop_guard
                    and original_max_lr is not None
                    and (not early_stop_triggered)
                    and batch == early_stop_step
                    and per > early_stop_per_threshold
                ):
                    logger.warning(
                        f"⚠️  EARLY STOP GUARD TRIGGERED: PER {per:.4f} > {early_stop_per_threshold} at step {batch}"
                    )
                    logger.warning(
                        f"   Reducing OneCycleLR max_lr from {original_max_lr} to {early_stop_reduced_max_lr}"
                    )
                    current_lr = optimizer.param_groups[0]["lr"]
                    scale_factor = early_stop_reduced_max_lr / original_max_lr
                    new_lr = current_lr * scale_factor
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    logger.warning(f"   Scaled current LR: {current_lr:.6f} → {new_lr:.6f}")
                    early_stop_triggered = True

                # Optional beam search subset
                beam_per_subset = None
                if beam_search_eval and batch % beam_search_interval == 0 and batch > 0:
                    logger.info(f"🔍 Running beam search eval on {beam_search_subset_size} samples (step {batch})")
                    with torch.no_grad():
                        model.eval()
                        beam_total_edit, beam_total_len, beam_count = 0, 0, 0
                        for Xb, yb, Xlb, ylb, ddb in testLoader:
                            if beam_count >= beam_search_subset_size:
                                break
                            Xb, yb, Xlb, ylb, ddb = (
                                Xb.to(device, non_blocking=True),
                                yb.to(device, non_blocking=True),
                                Xlb.to(device, non_blocking=True),
                                ylb.to(device, non_blocking=True),
                                ddb.to(device, non_blocking=True),
                            )
                            adjusted = ((Xlb - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)
                            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                                pred_b = model.forward(Xb, ddb, lengths=adjusted)
                            for i in range(min(pred_b.shape[0], beam_search_subset_size - beam_count)):
                                Te = int(adjusted[i].item())
                                lp = pred_b[i, :Te].log_softmax(dim=-1)
                                decoded = ctc_prefix_beam_search(lp, beam_size=beam_search_beam_size, blank=0)
                                trueSeq = np.array(yb[i][0: ylb[i]].cpu().detach())
                                m = SequenceMatcher(a=trueSeq.tolist(), b=decoded.tolist())
                                beam_total_edit += m.distance()
                                beam_total_len += len(trueSeq)
                                beam_count += 1
                                if beam_count >= beam_search_subset_size:
                                    break
                        if beam_total_len > 0:
                            beam_per_subset = beam_total_edit / beam_total_len
                            logger.info(f"  Beam search PER (subset): {beam_per_subset:.4f} (vs greedy PER: {per:.4f})")

                # plateau scheduler steps on metric
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(per)

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    else scheduler.get_last_lr()[0]
                )

                recent_grad_norms = grad_norms[-100:] if len(grad_norms) >= 100 else grad_norms
                recent_losses = loss_history[-100:] if len(loss_history) >= 100 else loss_history
                finite_grad_norms = [g for g in recent_grad_norms if np.isfinite(g)]
                avg_grad_norm = float(np.mean(finite_grad_norms)) if finite_grad_norms else 0.0
                avg_train_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                max_grad_norm = float(np.max(finite_grad_norms)) if finite_grad_norms else 0.0
                if not finite_grad_norms and recent_grad_norms:
                    max_grad_norm = float("inf")

                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                else:
                    memory_allocated = memory_reserved = 0.0

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
                print(log_msg)

                if skipped_updates > 0:
                    logger.info(f"  Skipped updates breakdown: {skipped_reasons}")

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
                if beam_per_subset is not None:
                    metrics_entry["beam_per_subset"] = float(beam_per_subset)

                # Occasional single-sample decode snapshot
                if batch == 0 or batch % 1000 == 0:
                    sample_X, sample_y, sample_X_len, sample_y_len, sample_dayIdx = next(iter(testLoader))
                    sample_X = sample_X[:1].to(device, non_blocking=True)
                    sample_y = sample_y[:1].to(device, non_blocking=True)
                    sample_X_len = sample_X_len[:1].to(device, non_blocking=True)
                    sample_y_len = sample_y_len[:1].to(device, non_blocking=True)
                    sample_dayIdx = sample_dayIdx[:1].to(device, non_blocking=True)

                    sample_T_eff = ((sample_X_len - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                        sample_pred = model.forward(sample_X, sample_dayIdx, lengths=sample_T_eff)

                    sample_T_eff_int = int(sample_T_eff[0].item())
                    sample_decoded = ctc_greedy_decode(sample_pred[0], sample_T_eff_int)
                    sample_target = sample_y[0, :sample_y_len[0]].cpu().numpy()

                    logger.info(f"Sample prediction (step {batch}):")
                    logger.info(f"  Target length: {len(sample_target)}, Pred length: {len(sample_decoded)}")
                    logger.info(f"  Target IDs (first 20): {sample_target[:20].tolist()}")
                    logger.info(f"  Pred IDs (first 20): {sample_decoded[:20].tolist()}")
                    m2 = SequenceMatcher(a=sample_target.tolist(), b=sample_decoded.tolist())
                    sample_per = m2.distance() / max(1, len(sample_target))
                    logger.info(f"  Sample PER: {sample_per:.4f}")

                    metrics_entry["sample_per"] = float(sample_per)
                    metrics_entry["sample_target_len"] = int(len(sample_target))
                    metrics_entry["sample_pred_len"] = int(len(sample_decoded))

                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics_entry) + "\n")

                startTime = time.time()

            # Save best by *actual* PER
            if per < best_per:
                best_per = per
                best_per_ma = per_ma
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "modelWeights"))
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "best_model.pt"))
                logger.info(f"✓ New best checkpoint saved (PER: {per:.4f}, PER_MA: {per_ma:.4f})")
            elif per_ma < best_per_ma:
                best_per_ma = per_ma

            testLoss.append(avgDayLoss)
            testPER.append(per)

            tStats = {"testLoss": np.array(testLoss), "testPER": np.array(testPER)}
            with open(os.path.join(args["outputDir"], "trainingStats"), "wb") as file:
                pickle.dump(tStats, file)

    # Final save
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
        final_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Final LR: {final_lr:.6f} (started at {peak_lr:.6f})")
    logger.info("=" * 80)


# Backward-compatible loader
def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = os.path.join(modelDir, "modelWeights")
    with open(os.path.join(modelDir, "args"), "rb") as handle:
        args = pickle.load(handle)

    n_days = args.get("nDays", None)
    if n_days is None:
        with open(args["datasetPath"], "rb") as f:
            loaded_data = pickle.load(f)
            n_days = len(loaded_data["train"])
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
