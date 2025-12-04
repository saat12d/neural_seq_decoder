# Baseline Recovery Run

**Goal**: Reproduce baseline stability by reverting to minimal configuration, then make one controlled change at a time.

## Configuration (`run8_recovery`)

### Architecture (minimal changes from baseline)
- GRU: 5 layers, 1024 units, bidirectional=True
- Stride: 4, Kernel: 32
- Dropout: 0.4 (baseline)
- **LayerNorm: Enabled** (helps stability, minimal change)
- **No input dropout** (baseline)

### Training (minimal, stable)
- **Optimizer**: Adam (not AdamW) - baseline-style
- **LR**: 0.001 constant (lower than baseline 0.02 which was too high)
- **Weight decay**: 0.0 (no weight decay, baseline-style)
- **Gradient clipping**: 5.0 (baseline-style, not 1.0)
- **AMP**: **DISABLED** (FP32 for stability)
- **Scheduler**: None (constant LR)

### Augmentation (minimal)
- **White noise**: 0.0 (turned off for recovery)
- **Constant offset**: 0.0 (turned off for recovery)
- **Time masking**: 0.0 (no SpecAugment yet)
- Gaussian smooth: 2.0 (baseline)

### Speed Optimizations (kept)
- **num_workers**: 4 (parallel data loading)
- **persistent_workers**: True (keep workers alive)
- **pin_memory**: True (faster GPU transfers)
- **non_blocking**: True (async data transfers)
- **AMP**: DISABLED (FP32 for stability, can enable later)

## Other
- **Adaptive LR**: DISABLED (baseline didn't have it)
- Batch size: 64
- Total batches: 10000

## What Changed from Recent Runs

1. ✅ **AMP → FP32**: Disabled mixed precision for stability
2. ✅ **AdamW → Adam**: Back to baseline optimizer
3. ✅ **Weight decay 1e-4 → 0.0**: No weight decay
4. ✅ **Gradient clip 1.0 → 5.0**: Baseline-style clipping
5. ✅ **Dropout 0.3 → 0.4**: Back to baseline
6. ✅ **LayerNorm: Enabled**: Keep it (helps stability, minimal change)
7. ✅ **All augmentation → Off**: Minimal for recovery
8. ✅ **Adaptive LR → Disabled**: Baseline didn't have it

## Expected Behavior

- **Stable training** (no NaN, no explosions)
- **PER should decrease steadily** from ~0.94
- **Target**: Get below 0.35 PER (then we can iterate)

## Next Steps (After Recovery)

Once we hit ≤0.35 PER, make **one change at a time**:

1. **ReduceLROnPlateau** on PER (patience 8-12 evals, factor 0.5)
2. **AdamW** with small weight decay (1e-4)
3. **SpecAugment-lite** (time_mask_prob=0.05, width=15, max_masks=1)
4. **Dropout sweep**: 0.3, 0.4, 0.5
5. **AMP** (only after stability confirmed)

## Run Command

```bash
python scripts/train_model.py
```

Output: `data/checkpoints/run8_recovery/`

