# Training Run Tracking

This document tracks each training run, what it tests, what changed, and the results.

**Baseline Repository**: https://github.com/cffan/neural_seq_decoder

---

## Run #1: `gru_ctc_reg1` (Baseline Attempt)

**Purpose**: Initial attempt with baseline settings

**Configuration**:
- LR: 0.02 (baseline, constant)
- Optimizer: Adam (likely)
- Weight decay: 1e-5
- No gradient clipping
- No mixed precision
- White noise: 0.8
- Dropout: 0.4

**Result**: ❌ **Failed** - Gradient explosions, training unstable

**Key Finding**: Baseline LR 0.02 is **way too high** for this model

---

## Run #2: `gru_ctc_reg2`

**Purpose**: First attempt to fix stability issues

**Changes from Run #1**:
- Added gradient clipping (max_norm=1.0)
- Added mixed precision training (FP16/BF16)
- Improved logging
- Fixed metric calculation (CER → PER)

**Configuration**:
- LR: 0.02 → 0.010 (reduced)
- Gradient clipping: 1.0
- Mixed precision: Enabled

**Result**: ❌ **Failed** - Still unstable, PER ~96% (metric calculation issue)

**Key Finding**: 
- Gradient clipping helps but not enough
- LR still too high
- Metric calculation was incorrect

---

## Run #3: `gru_ctc_reg3`

**Purpose**: Lower LR + cosine decay + stronger regularization

**Changes from Run #2**:
- LR: 0.005 peak, cosine decay to 5e-4
- Warmup: 2500 steps (0 → 0.005)
- Softened augmentation (noise 0.4, offset 0.1)
- Reduced time masking (width 15, max 2 masks)
- Input dropout: 0.1

**Configuration**:
- `lrStart`: 0.005
- `lrEnd`: 5e-4
- `warmup_steps`: 2500
- `cosine_T_max`: 7500
- `whiteNoiseSD`: 0.4
- `constantOffsetSD`: 0.1
- `time_mask_width`: 15
- `time_mask_max_masks`: 2

**Result**: ⚠️ **Partially successful**
- Best PER: 0.298 at step 600
- Degraded to 0.628 by step 2500
- Gradient explosions after warmup (Inf gradients)
- **Finding**: Model learns early but degrades as LR increases

---

## Run #4: `gru_ctc_reg4`

**Purpose**: Cap peak LR much lower, shorter warmup

**Changes from Run #3**:
- Peak LR: 0.002 (capped, much lower)
- Warmup: 1000 steps (shorter)
- Cosine decay: 0.002 → 0.0008
- Better BF16 detection

**Configuration**:
- `lrStart`: 0.0
- `peak_lr`: 0.002
- `lrEnd`: 0.0008
- `warmup_steps`: 1000
- `cosine_T_max`: 9000

**Result**: ⚠️ **Stable but degrading**
- Best PER: 0.298 at step 600-700
- Degraded to 0.387 by step 3000
- No gradient explosions (stable)
- **Finding**: Model degrades even with lower peak LR during cosine decay
- **Issue**: Checkpoint saved based on moving average PER, not actual PER (bug)

---

## Run #5: `gru_ctc_reg5`

**Purpose**: Constant LR to stay in stable regime

**Changes from Run #4**:
- Constant LR: 0.001 (no warmup, no decay)
- Fixed checkpoint saving (use actual PER, not moving average)

**Configuration**:
- `lrStart`: 0.001
- `lrEnd`: 0.001
- `peak_lr`: 0.001
- `warmup_steps`: 0
- Constant LR scheduler

**Result**: ✅ **Stable but slow**
- PER: 0.94 → 0.58 (step 1700)
- Stable gradients (no explosions)
- Slow learning rate
- **Finding**: Constant LR 0.001 is stable but learning is slow

---

## Run #6: `gru_ctc_reg6` (Current)

**Purpose**: Faster learning with higher constant LR + reduced regularization

**Changes from Run #5**:
- Constant LR: 0.0015 (50% higher)
- Dropout: 0.3 (reduced from 0.4)
- White noise: 0.2 (reduced from 0.4)
- Checkpoint resuming support added

**Configuration**:
- `lrStart`: 0.0015
- `lrEnd`: 0.0015
- `peak_lr`: 0.0015
- `dropout`: 0.3
- `whiteNoiseSD`: 0.2

**Result**: ⚠️ **Good initially, instability later**
- PER: 0.944 → 0.512 (best at step 4600)
- Faster learning than Run #5
- **Issue**: NaN gradients starting at step 4200
- Gradient norms exploding (max 17.0)
- **Finding**: LR 0.0015 becomes unstable after ~4000 steps

**Status**: Training stopped at step 5100 due to instability

---

## Run #7: `gru_ctc_reg7` (NEW)

**Purpose**: Test adaptive LR reduction to handle instability automatically

**Changes from Run #6**:
- **NEW**: Adaptive LR reduction enabled
  - Automatically reduces LR by 20% when instability detected
  - Triggers on: NaN gradients, grad_norm > 10, or too many skipped updates
  - Min LR: 0.0005, Max reductions: 5
- Same base config as Run #6 (LR 0.0015, dropout 0.3, noise 0.2)

**Configuration**:
- `lrStart`: 0.0015
- `lrEnd`: 0.0015
- `peak_lr`: 0.0015
- `dropout`: 0.3
- `whiteNoiseSD`: 0.2
- `adaptive_lr`: True ⭐ **NEW**
- `lr_reduction_factor`: 0.8 (reduce by 20%)
- `lr_reduction_threshold`: 10.0 (trigger if grad_norm > 10)
- `min_lr`: 0.0005
- `max_lr_reductions`: 5

**Expected Behavior**:
- Start at LR 0.0015 (fast learning)
- When NaN gradients appear (~step 4000), automatically reduce to 0.0012
- If instability continues, further reductions: 0.0010 → 0.0008 → 0.0006 → 0.0005
- Should maintain stability while maximizing learning speed

**Hypothesis**: Adaptive LR will allow us to start high (fast learning) and automatically reduce when needed (stability), giving best of both worlds.

---

## Summary of Learnings

### What Works ✅
1. **Constant LR 0.001-0.0015** (for first ~4000 steps)
2. **Gradient clipping at 1.0** (critical for stability)
3. **Mixed precision training** (FP16/BF16, ~2x speedup)
4. **Reduced regularization** (dropout 0.3, noise 0.2)
5. **LayerNorm + input dropout** (stabilizes training)
6. **AdamW with weight_decay 1e-4** (better than Adam)
7. **Checkpoint saving based on actual PER** (not moving average)

### What Doesn't Work ❌
1. **High LR (0.02)** → immediate explosions
2. **LR schedules that push LR too high** (warmup to 0.005)
3. **LR 0.0015 for extended training** (>4000 steps) → instability
4. **Too much noise (0.8)** → slows learning
5. **High dropout (0.4)** → may over-regularize

### Key Insights
- **Sweet spot LR**: 0.001-0.0015 for early training
- **LR needs to decrease** as training progresses (or use adaptive reduction)
- **Gradient clipping is essential** but not sufficient alone
- **Model learns quickly early** (PER 0.94 → 0.30 in ~600 steps) but needs careful LR management

---

## Best Results So Far

| Run | Best PER | Step | Notes |
|-----|----------|------|-------|
| #3 | 0.298 | 600 | Early learning, degraded later |
| #4 | 0.298 | 600-700 | Same as #3, stable longer |
| #6 | 0.512 | 4600 | Best sustained performance, but instability later |

**Target**: Beat 0.30 PER consistently for full 10k steps

---

## Next Steps After Run #7

1. If adaptive LR works: Fine-tune reduction factor/threshold
2. If still unstable: Try lower starting LR (0.0012) with adaptive reduction
3. If works well: Extend to full 10k steps and evaluate

---

**Last Updated**: Before Run #7

