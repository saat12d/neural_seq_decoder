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
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------------------------
# CTC decode helpers
# ---------------------------
def ctc_greedy_decode(logits, T_eff):
    """Greedy CTC decode with blank=0, collapse repeats, drop blanks."""
    if logits.ndim == 3:
        logits = logits[0]
    path = logits[:T_eff].argmax(dim=-1).detach().cpu().tolist()
    out, prev = [], None
    for p in path:
        if p != 0 and p != prev:
            out.append(p)
        prev = p
    return np.asarray(out, dtype=np.int32)


# ---------------------------
# Data
# ---------------------------
def _collate(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    Xp = pad_sequence(X, batch_first=True, padding_value=0)
    yp = pad_sequence(y, batch_first=True, padding_value=0)
    return Xp, yp, torch.stack(X_lens), torch.stack(y_lens), torch.stack(days)


def getDatasetLoaders(datasetPath, batchSize, args):
    with open(datasetPath, "rb") as f:
        data = pickle.load(f)

    train_ds = SpeechDataset(
        data["train"],
        transform=None,
        split="train",
        time_mask_prob=args.get("time_mask_prob", 0.0),
        time_mask_width=args.get("time_mask_width", 0),
        time_mask_max_masks=args.get("time_mask_max_masks", 1),
        freq_mask_prob=args.get("freq_mask_prob", 0.0),
        freq_mask_width=args.get("freq_mask_width", 0),
        freq_mask_max_masks=args.get("freq_mask_max_masks", 0),
    )
    test_ds = SpeechDataset(data["test"], split="test")

    num_workers = args.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_collate,
    )
    return train_loader, test_loader, data


# ---------------------------
# Training
# ---------------------------
def trainModel(args):
    # Safer allocator flag set (avoid unsupported keys like expandable_segments)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    logger = setup_logging(args["outputDir"])

    # TF32 (faster matmuls on Ampere+ without accuracy loss for FP32 ops)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("Enabled TF32 for faster FP32 matmuls (Ampere+ GPUs)")
        except Exception:
            pass

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

    train_loader, test_loader, loaded = getDatasetLoaders(
        args["datasetPath"], args["batchSize"], args
    )

    n_days = len(loaded["train"])
    args["nDays"] = n_days

    logger.info(f"Dataset loaded: {n_days} training days")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    with open(os.path.join(args["outputDir"], "args"), "wb") as f:
        pickle.dump(args, f)

    logger.info("=" * 80)
    logger.info("Model Architecture")
    logger.info("=" * 80)
    logger.info(f"Input features: {args['nInputFeatures']}")
    logger.info(f"Hidden units: {args['nUnits']}")
    logger.info(f"GRU layers: {args['nLayers']}")
    logger.info(f"Output classes: {args['nClasses']} (+ 1 blank = {args['nClasses']+1})")
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # --- CTC sanity ---
    logger.info("=" * 80)
    logger.info("CTC Sanity Checks")
    logger.info("=" * 80)
    assert loss_ctc.blank == 0
    logger.info(f"✓ CTCLoss blank index: {loss_ctc.blank}")
    with torch.no_grad():
        Xs, ys, Xl, yl, days = next(iter(train_loader))
        Xs, ys, Xl, yl, days = (
            Xs.to(device), ys.to(device), Xl.to(device), yl.to(device), days.to(device)
        )
        T_eff_s = ((Xl - args["kernelLen"]) // args["strideLen"]).clamp(min=1)
        logger.info(f"✓ T_eff calculation verified (min={T_eff_s.min().item()}, max={T_eff_s.max().item()})")
        bad = False
        for b in range(ys.size(0)):
            Lb = int(yl[b].item())
            if Lb > 0 and (ys[b, :Lb] == 0).any():
                bad = True
                break
        if bad:
            logger.warning("Found blank(0) inside valid labels—remap needed.")
        else:
            logger.info("✓ Labels verified: no blanks in valid spans (labels>=1)")
        logger.info(f"✓ Input lengths: min={Xl.min().item()}, max={Xl.max().item()}")
        logger.info(f"✓ Target lengths: min={yl.min().item()}, max={yl.max().item()}")
        logger.info(f"✓ T_eff lengths: min={T_eff_s.min().item()}, max={T_eff_s.max().item()}")

    # --- Precision (AMP) ---
    use_amp = bool(args.get("use_amp", True))
    if use_amp and torch.cuda.is_available():
        # Prefer BF16 on Ampere+, else FP16 (T4)
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                logger.info("Using mixed precision BF16 (Ampere+)")
            else:
                amp_dtype = torch.float16
                logger.info("Using mixed precision FP16 (Turing/T4)")
        except Exception:
            amp_dtype = torch.float16
            logger.info("Using mixed precision FP16")
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
        logger.info("  Safety: log_softmax + CTCLoss computed in FP32")
    else:
        amp_dtype, scaler = torch.float32, None
        logger.info("Using full precision (FP32) training")

    # --- Optimizer ---
    weight_decay = args.get("weight_decay", args.get("l2_decay", 1e-5))
    opt_name = args.get("optimizer", "adam").lower()
    peak_lr = float(args.get("peak_lr", 0.0015))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    # --- Scheduler: Warmup -> Cosine (no OneCycle) ---
    warmup_steps = int(args.get("warmup_steps", 1500))
    cosine_steps = int(args.get("cosine_T_max", args["nBatch"] - warmup_steps))
    lr_end = float(args.get("lrEnd", 1e-5))

    def warmup_lambda(step):
        if step >= warmup_steps:
            return 1.0
        return float(step) / float(max(1, warmup_steps))

    warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr_end)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, cos], milestones=[warmup_steps]
    )

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Optimizer: {opt_name.upper()}")
    logger.info(f"Peak LR: {peak_lr} | End LR: {lr_end}")
    logger.info(f"Warmup steps: {warmup_steps} | Cosine steps: {cosine_steps}")
    logger.info(f"Weight decay: {weight_decay}")
    grad_clip = float(args.get("grad_clip_norm", 1.0))
    logger.info(f"Gradient clipping: max_norm={grad_clip}")
    logger.info("CTC Decoding: Greedy")

    # --- State & bookkeeping ---
    moving_window = int(args.get("per_ma_window", 200))
    per_hist = []
    best_per = float("inf")
    start_time = time.time()

    metrics_path = os.path.join(args["outputDir"], "metrics.jsonl")
    accumulation_steps = int(args.get("gradient_accumulation_steps", 1))
    if accumulation_steps > 1:
        logger.info(f"Gradient accumulation: {accumulation_steps} steps "
                    f"(effective batch size: {args['batchSize'] * accumulation_steps})")

    train_iter = iter(train_loader)

    for step in range(int(args["nBatch"])):
        model.train()
        try:
            X, y, X_len, y_len, day = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, y, X_len, y_len, day = next(train_iter)

        # Device transfer (non-blocking)
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X_len = X_len.to(device, non_blocking=True)
        y_len = y_len.to(device, non_blocking=True)
        day = day.to(device, non_blocking=True)

        # Light additive noise (GPU-side)
        if args.get("whiteNoiseSD", 0) > 0:
            X = X + torch.randn_like(X) * args["whiteNoiseSD"]
        if args.get("constantOffsetSD", 0) > 0:
            X = X + torch.randn(X.shape[0], 1, X.shape[2], device=X.device, dtype=X.dtype) * args["constantOffsetSD"]

        # Effective time after conv stride
        T_eff = ((X_len - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)

        # Forward (AMP)
        if scaler is None:
            pred = model(X, day, lengths=T_eff)
            log_probs = pred.float().log_softmax(2)
            loss = loss_ctc(log_probs.permute(1, 0, 2), y, T_eff, y_len)
        else:
            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                pred = model(X, day, lengths=T_eff)
                # compute log_softmax in FP32 for stability
                log_probs = pred.float().log_softmax(2)
                loss = loss_ctc(log_probs.permute(1, 0, 2), y, T_eff, y_len)

        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        # Backward
        if scaler is None:
            if (step % accumulation_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if ((step + 1) % accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            if (step % accumulation_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if ((step + 1) % accumulation_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # ---- Eval every 100 steps ----
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                total_ed, total_len = 0, 0
                for Xb, yb, Xlb, ylb, dayb in test_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    Xlb = Xlb.to(device, non_blocking=True)
                    ylb = ylb.to(device, non_blocking=True)
                    dayb = dayb.to(device, non_blocking=True)

                    T_eff_b = ((Xlb - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)

                    if scaler is None:
                        pred_b = model(Xb, dayb, lengths=T_eff_b)
                        lp = pred_b.float().log_softmax(2)
                        l = loss_ctc(lp.permute(1, 0, 2), yb, T_eff_b, ylb)
                    else:
                        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                            pred_b = model(Xb, dayb, lengths=T_eff_b)
                            lp = pred_b.float().log_softmax(2)
                            l = loss_ctc(lp.permute(1, 0, 2), yb, T_eff_b, ylb)
                    losses.append(l.item())

                    # decode + PER
                    for i in range(pred_b.size(0)):
                        Te = int(T_eff_b[i].item())
                        dec = ctc_greedy_decode(pred_b[i], Te)
                        tgt = yb[i, : ylb[i]].detach().cpu().numpy()
                        m = SequenceMatcher(a=tgt.tolist(), b=dec.tolist())
                        total_ed += m.distance()
                        total_len += len(tgt)

                eval_loss = float(np.mean(losses)) if losses else 0.0
                per = (total_ed / max(1, total_len)) if total_len > 0 else 1.0

                per_hist.append(per)
                if len(per_hist) > moving_window:
                    per_hist.pop(0)
                per_ma = float(np.mean(per_hist)) if per_hist else per

                # Stats
                elapsed = time.time() - start_time
                time_per_batch = elapsed / 100.0
                start_time = time.time()

                # current LR (works for SequentialLR)
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    current_lr = optimizer.param_groups[0]["lr"]

                # CUDA mem
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_res = torch.cuda.memory_reserved() / 1e9
                else:
                    mem_alloc = mem_res = 0.0

                logger.info(
                    f"batch {step:5d} | loss: {eval_loss:7.4f} | "
                    f"per: {per:7.4f} (ma: {per_ma:7.4f}) | "
                    f"lr: {current_lr:.6f} | time: {time_per_batch:5.2f}s | "
                    f"mem: {mem_alloc:4.2f}GB/{mem_res:4.2f}GB"
                )

                # Save best by actual PER
                if per < best_per:
                    best_per = per
                    torch.save(model.state_dict(), os.path.join(args["outputDir"], "modelWeights"))
                    torch.save(model.state_dict(), os.path.join(args["outputDir"], "best_model.pt"))
                    logger.info(f"✓ New best checkpoint saved (PER: {per:.4f})")

                # metrics.jsonl
                entry = {
                    "step": step,
                    "ctc_loss": eval_loss,
                    "per": per,
                    "per_ma": per_ma,
                    "lr": current_lr,
                    "time_per_batch": time_per_batch,
                    "memory_allocated_gb": mem_alloc,
                    "memory_reserved_gb": mem_res,
                }
                # Sample decode
                if step in (0, 1000, 2000) and len(test_loader) > 0:
                    Xs, ys, Xl, yl, dd = next(iter(test_loader))
                    Xs = Xs[:1].to(device)
                    ys = ys[:1].to(device)
                    Xl = Xl[:1].to(device)
                    yl = yl[:1].to(device)
                    dd = dd[:1].to(device)
                    Te = ((Xl - model.kernelLen) // model.strideLen).clamp(min=1).to(torch.int32)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None), dtype=amp_dtype):
                        pr = model(Xs, dd, lengths=Te)
                    Tei = int(Te[0].item())
                    dec = ctc_greedy_decode(pr[0], Tei)
                    tgt = ys[0, : yl[0]].detach().cpu().numpy()
                    m = SequenceMatcher(a=tgt.tolist(), b=dec.tolist())
                    entry["sample_per"] = float(m.distance() / max(1, len(tgt)))
                    entry["sample_target_len"] = int(len(tgt))
                    entry["sample_pred_len"] = int(len(dec))

                with open(metrics_path, "a") as w:
                    w.write(json.dumps(entry) + "\n")

    # Final save
    torch.save(model.state_dict(), os.path.join(args["outputDir"], "final_model.pt"))
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best PER: {best_per:.4f}")
    logger.info("=" * 80)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = os.path.join(modelDir, "modelWeights")
    with open(os.path.join(modelDir, "args"), "rb") as f:
        args = pickle.load(f)

    n_days = args.get("nDays", None)
    if n_days is None:
        with open(args["datasetPath"], "rb") as f:
            loaded = pickle.load(f)
        n_days = len(loaded["train"])
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
