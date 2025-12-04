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

## Run #8: `run8_recovery` (Baseline Recovery)

**Purpose**: Revert to minimal baseline configuration to reproduce stability, then make one controlled change at a time.

**Rationale**: After multiple runs with many changes, training became unstable. Need to go back to basics and verify we can reproduce baseline stability.

**Changes from Run #7**:
- **Optimizer**: AdamW → **Adam** (baseline used Adam)
- **Weight decay**: 1e-4 → **0.0** (baseline had minimal/no weight decay)
- **Gradient clipping**: 1.0 → **5.0** (baseline-style, less aggressive)
- **AMP**: Enabled → **Disabled** (FP32 for stability)
- **Dropout**: 0.3 → **0.4** (back to baseline)
- **All augmentation**: Enabled → **Disabled** (white noise 0.0, offset 0.0, time masking 0.0)
- **Adaptive LR**: Enabled → **Disabled** (baseline didn't have it)
- **LR**: 0.0015 → **0.001** (constant, lower than baseline 0.02 which was too high)

**Kept from improvements**:
- **LayerNorm**: Enabled (helps stability, minimal change)
- **Gradient clipping**: Enabled (baseline didn't have it, but we keep at 5.0 for safety)
- **Speed optimizations**: num_workers=4, persistent_workers, pin_memory, non_blocking transfers

**Configuration**:
- `lrStart`: 0.001 (constant)
- `lrEnd`: 0.001 (constant)
- `optimizer`: 'adam'
- `weight_decay`: 0.0
- `grad_clip_norm`: 5.0
- `use_amp`: False
- `dropout`: 0.4
- `use_layer_norm`: True
- `whiteNoiseSD`: 0.0
- `constantOffsetSD`: 0.0
- `time_mask_prob`: 0.0
- `adaptive_lr`: False

**Expected Behavior**:
- **Stable training** (no NaN, no explosions)
- **PER should decrease steadily** from ~0.94
- **Target**: Get below 0.35 PER, then iterate one change at a time

**Next Steps (After Recovery)**:
Once stable and ≤0.35 PER:
1. **ReduceLROnPlateau** on PER (patience 8-12 evals, factor 0.5)
2. **AdamW** with small weight decay (1e-4)
3. **SpecAugment-lite** (time_mask_prob=0.05, width=15, max_masks=1)
4. **Dropout sweep**: 0.3, 0.4, 0.5
5. **AMP** (only after stability confirmed)

---

## Run #9: `run9_clean_baseline` (Clean Baseline Fix)

**Purpose**: Strip back to dead-simple baseline, fix T_eff calculation bug, ensure stable training.

**Rationale**: Expert analysis identified that model was outputting mostly blanks/tiny sequences (PER stuck in 0.9s). Root causes:
1. T_eff calculation using float division (`/`) instead of floor division (`//`)
2. T_eff not clamped to minimum 1
3. AMP potentially hiding instability
4. Need dead-simple optimizer/LR setup

**Critical Fixes**:
- **T_eff calculation**: Fixed to use `//` (floor division) instead of `/` (float division)
- **T_eff clamping**: Added `.clamp(min=1)` to ensure T_eff >= 1
- **Applied in 3 places**: Training loop, eval loop, sample logging

**Configuration Changes**:
- **LR**: Exactly **0.001** (1e-3), constant (no scheduler)
- **AMP**: **OFF** (until PER < 0.5, stability first)
- **Gradient clipping**: **0.5** (tighter, as recommended)
- **Dropout**: **0.4** (baseline value)
- **Optimizer**: **Adam** (not AdamW)
- **Weight decay**: **0.0** (no weight decay)
- **All augmentation**: **OFF** (reproduce baseline)

**Kept**:
- LayerNorm enabled (helps stability)
- Speed optimizations (num_workers, persistent_workers, etc.)

**Configuration**:
- `lrStart`: 0.001 (constant, exactly 1e-3)
- `lrEnd`: 0.001 (constant)
- `use_amp`: False (OFF until PER < 0.5)
- `dropout`: 0.4 (baseline)
- `grad_clip_norm`: 0.5 (tighter clipping)
- `optimizer`: 'adam'
- `weight_decay`: 0.0

**Expected Behavior**:
- **Stable training** (no NaN, no explosions)
- **PER should decrease steadily** from ~0.94
- **No more blank-heavy outputs** (T_eff fix should help)
- **Target**: Get below 0.5 PER, then enable AMP and iterate

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

---

## Run #10: `run10_recovery` (Recovery with Speed Optimizations)

**Purpose**: Same clean baseline as Run #9, but with speed optimizations enabled.

**Changes from Run #9**:
- **cuDNN benchmark**: Enabled (faster RNN operations)
- **torch.compile**: Enabled (PyTorch 2.0+ model compilation)
- **Expected speedup**: ~25-40% faster training

**Configuration** (same as Run #9):
- `lrStart`: 0.001 (constant, exactly 1e-3)
- `lrEnd`: 0.001 (constant)
- `use_amp`: False (OFF until PER < 0.5)
- `dropout`: 0.4 (baseline)
- `grad_clip_norm`: 0.5 (tighter clipping)
- `optimizer`: 'adam'
- `weight_decay`: 0.0
- All augmentation: OFF
- T_eff calculation: Fixed (floor division + clamp)

**Expected Behavior**:
- **Same stability as Run #9** (no behavior changes)
- **~25-40% faster training** (speed optimizations)
- **PER should decrease steadily** from ~0.94
- **No more blank-heavy outputs** (T_eff fix)

---

---

## Run #11: `run11_plateau_break` (Break Through PER Plateau)

**Purpose**: Break through the ~36-38% PER plateau observed in Run #10

**Changes from Run #10**:
- **Prefix beam search** for CTC decoding (beam_size=20) instead of greedy
- **AdamW optimizer** (decoupled weight decay) instead of Adam
- **Lower base LR**: 8e-4 (down from 1e-3)
- **ReduceLROnPlateau scheduler** on validation PER (patience=8, factor=0.5, min_lr=1e-5)
- **Full SpecAugment**: Time masking (prob=0.10, width=40, max_masks=2) + Frequency masking (prob=0.10, width=12, max_masks=2)
- **Gradient clipping**: 1.5 (moderate, in 1.0-2.0 range)

**Configuration**:
- `optimizer`: 'adamw'
- `lrStart`: 0.0008 (8e-4)
- `lrEnd`: 0.0008 (will be reduced by scheduler)
- `weight_decay`: 1e-4
- `use_plateau_scheduler`: True
- `plateau_patience`: 8
- `plateau_factor`: 0.5
- `plateau_min_lr`: 1e-5
- `use_beam_search`: True
- `beam_size`: 20
- `time_mask_prob`: 0.10
- `time_mask_width`: 40
- `time_mask_max_masks`: 2
- `freq_mask_prob`: 0.10
- `freq_mask_width`: 12
- `freq_mask_max_masks`: 2
- `grad_clip_norm`: 1.5
- All other settings same as Run #10

**Expected Behavior**:
- **Beam search should improve PER** by 5-15% relative to greedy
- **ReduceLROnPlateau should break plateau** by reducing LR when PER stalls
- **SpecAugment should improve generalization** without slowing training much
- **Target: PER < 0.30** (closer to baseline ~20%)

**Rationale**:
- Beam search is standard for CTC and consistently beats greedy
- ReduceLROnPlateau is a robust "safety net" when progress stalls
- SpecAugment (time + freq masks) is standard for speech recognition
- Lower LR (8e-4) should help convergence without instability

---

**Last Updated**: Run #11 (Plateau Break)

