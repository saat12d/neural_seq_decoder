# Changelog: Modifications from Original Baseline

**Baseline Repository**: https://github.com/cffan/neural_seq_decoder

This document tracks all changes made from the original baseline to diagnose issues and understand what works.

---

## Summary of Changes

### **Current Status (Run #6)**
- **PER**: 0.944 → 0.513 (46% improvement, but showing instability signs)
- **Best PER**: 0.512 at step 4600
- **Issue**: NaN gradients appearing at step 4200+, gradient norms exploding (max 17.0)

---

## 1. Learning Rate Schedule Changes

### **Original Baseline** (from `config.yaml`)
- `lrStart: 0.02` (constant)
- `lrEnd: 0.02` (constant)
- No warmup, no decay

### **Our Changes**
- **Run #1-2**: Started with baseline LR 0.02 → caused gradient explosions
- **Run #3**: 
  - `lrStart: 0.005`, `lrEnd: 5e-4`
  - Warmup: 2500 steps (0 → 0.005)
  - Cosine decay: 0.005 → 5e-4 over 7500 steps
  - **Result**: Still unstable, PER degraded after warmup
- **Run #4**:
  - `lrStart: 0.0`, `peak_lr: 0.002`, `lrEnd: 0.0008`
  - Warmup: 1000 steps (0 → 0.002)
  - Cosine decay: 0.002 → 0.0008 over 9000 steps
  - **Result**: Stable but PER degraded (0.298 → 0.387)
- **Run #5**:
  - Constant LR: `0.001` (no warmup, no decay)
  - **Result**: Stable, slow learning, PER plateaued at ~0.58
- **Run #6** (Current):
  - Constant LR: `0.0015` (no warmup, no decay)
  - **Result**: Faster learning, but instability starting (NaN gradients at step 4200+)

### **Key Finding**
- Original LR 0.02 was **way too high** → immediate gradient explosions
- Sweet spot appears to be **0.001-0.0015** for stability
- LR 0.0015 becomes unstable after ~4000 steps (needs reduction or schedule)

---

## 2. Optimizer Changes

### **Original Baseline**
- Likely Adam (not explicitly specified in config.yaml)
- `l2_decay: 1e-5` (very low weight decay)

### **Our Changes**
- **Switched to AdamW** (better for weight decay)
- `weight_decay: 1e-4` (10x higher than baseline)
- **Rationale**: Better regularization, standard for modern training

---

## 3. Gradient Clipping

### **Original Baseline**
- **No gradient clipping** (likely cause of explosions)

### **Our Changes**
- **Added**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Location**: After `scaler.unscale_()` (for AMP) or after `loss.backward()` (for FP32)
- **Result**: Prevents most explosions, but NaN gradients can still occur before clipping

---

## 4. Mixed Precision Training (AMP)

### **Original Baseline**
- **Not implemented** (likely FP32 only)

### **Our Changes**
- **Added**: Mixed precision training with FP16/BF16
- **BF16 preferred** (more stable than FP16) if GPU supports it
- **GradScaler** for FP16 (not needed for BF16)
- **Result**: ~2x speedup, no accuracy loss observed

---

## 5. Data Augmentation Changes

### **Original Baseline** (from `config.yaml`)
- `whiteNoiseSD: 0.8` (high noise)
- `constantOffsetSD: 0.2`
- `gaussianSmoothWidth: 2.0`

### **Our Changes**
- **Run #3**: `whiteNoiseSD: 0.4` (reduced by 50%)
- **Run #6**: `whiteNoiseSD: 0.2` (reduced by 75% from baseline)
- **Rationale**: Too much noise was slowing learning

---

## 6. Dropout Changes

### **Original Baseline**
- `dropout: 0.4`

### **Our Changes**
- **Run #6**: `dropout: 0.3` (reduced by 25%)
- **Rationale**: Speed up learning, reduce over-regularization

---

## 7. Time Masking (SpecAugment-style)

### **Original Baseline**
- **Not implemented** (or minimal)

### **Our Changes**
- **Added**: `apply_time_mask()` function in `dataset.py`
- **Parameters**:
  - `time_mask_prob: 0.10` (10% chance to apply)
  - `time_mask_width: 15` (reduced from 20)
  - `time_mask_max_masks: 2` (allow 1-2 masks per sample)
- **Rationale**: Light augmentation, softer than aggressive masking

---

## 8. Layer Normalization

### **Original Baseline**
- **Not specified** (likely not used)

### **Our Changes**
- **Added**: `use_layer_norm: True`
- **Input dropout**: `input_dropout: 0.1`
- **Rationale**: Stabilize training, standard for RNNs

---

## 9. Metric Calculation Fixes

### **Original Baseline**
- Likely computed PER correctly (need to verify)

### **Our Changes**
- **Fixed**: Renamed "CER" to "PER" throughout (was incorrectly labeled)
- **Added**: `ctc_greedy_decode()` function (copied from `eval.py`) for consistent decoding
- **Changed**: Checkpoint saving now uses **actual PER** instead of moving average PER
  - **Reason**: Moving average can improve even as actual PER degrades

---

## 10. Training Loop Improvements

### **Original Baseline**
- Basic training loop, likely no special handling

### **Our Changes**
- **Added**:
  - NaN/Inf gradient detection and skipping
  - Skipped update tracking and logging
  - Moving average PER tracking (200-step window)
  - Best checkpoint saving based on actual PER
  - Comprehensive logging (grad norms, memory usage, sample predictions)
  - Metrics saved to JSONL file

---

## 11. DataLoader Optimizations

### **Original Baseline**
- Likely basic DataLoader setup

### **Our Changes**
- **Added**:
  - `num_workers=4` (parallel data loading)
  - `persistent_workers=True` (keep workers alive)
  - `pin_memory=True` (faster GPU transfers)
  - `non_blocking=True` for GPU transfers
- **Result**: ~2x faster data loading

---

## 12. Checkpoint Resuming

### **Original Baseline**
- **Not implemented**

### **Our Changes**
- **Added**: Resume from checkpoint functionality
  - Loads model weights from `best_model.pt`
  - Infers batch number from `metrics.jsonl`
  - Supports loading optimizer state (if available)

---

## 13. Code Quality Fixes

### **Fixes Applied**
1. **Removed redundant `torch.sum(loss)`**: CTCLoss with `reduction="mean"` already returns scalar
2. **Fixed `nDays` loading**: Now loads from saved `args` instead of hardcoded value
3. **Improved BF16 detection**: Tests autocast directly instead of just checking device capability

---

## Current Issues & Observations

### **Issue #1: Gradient Explosions at High LR**
- **Symptom**: Original LR 0.02 → immediate NaN/Inf gradients
- **Fix**: Reduced to 0.001-0.0015 range
- **Status**: Mostly resolved, but LR 0.0015 becomes unstable after ~4000 steps

### **Issue #2: PER Degradation After Warmup**
- **Symptom**: PER improves early (0.298 at step 600), then degrades (0.387 at step 3000)
- **Cause**: LR schedule pushing model into unstable regime
- **Fix**: Switched to constant LR
- **Status**: Resolved with constant LR, but learning is slower

### **Issue #3: NaN Gradients at Step 4200+ (Run #6)**
- **Symptom**: First NaN gradient at step 4200, gradient norms exploding (max 17.0)
- **Cause**: LR 0.0015 becoming too high as training progresses
- **Fix Needed**: Reduce LR to 0.0012 or implement LR decay
- **Status**: **ACTIVE ISSUE**

### **Issue #4: Checkpoint Saving Bug**
- **Symptom**: Best checkpoint saved based on moving average PER, not actual PER
- **Impact**: Saved checkpoint from degraded state (PER 0.36) instead of best (PER 0.298)
- **Fix**: Changed to save based on actual PER
- **Status**: Fixed

---

## Recommendations

### **For Next Run**
1. **Reduce LR to 0.0012** when NaN gradients appear (or implement adaptive LR reduction)
2. **Monitor gradient norms**: If consistently >10, reduce LR
3. **Consider LR decay**: After step 3000-4000, decay LR from 0.0015 → 0.001
4. **Keep current regularization**: Dropout 0.3, noise 0.2 seem to work well

### **What Works**
- ✅ Constant LR 0.001-0.0015 (for first ~4000 steps)
- ✅ Gradient clipping at 1.0
- ✅ Mixed precision training (FP16/BF16)
- ✅ Reduced regularization (dropout 0.3, noise 0.2)
- ✅ LayerNorm + input dropout
- ✅ AdamW with weight_decay 1e-4

### **What Doesn't Work**
- ❌ High LR (0.02) → immediate explosions
- ❌ LR schedules that push LR too high (warmup to 0.005)
- ❌ LR 0.0015 for extended training (>4000 steps) → instability
- ❌ Too much noise (0.8) → slows learning
- ❌ High dropout (0.4) → may be over-regularizing

---

## File Changes Summary

### **Modified Files**
1. `scripts/train_model.py` - Hyperparameter configuration
2. `src/neural_decoder/neural_decoder_trainer.py` - Core training loop, scheduler, checkpointing
3. `src/neural_decoder/dataset.py` - Time masking augmentation

### **Key Functions Added/Modified**
- `ctc_greedy_decode()` - Consistent CTC decoding
- `apply_time_mask()` - Time masking augmentation (modified to support multiple masks)
- `trainModel()` - Extensive modifications for stability and monitoring
- Checkpoint resuming logic

---

## Baseline Comparison Table

| Component | Baseline | Current (Run #6) | Status |
|-----------|----------|------------------|--------|
| LR | 0.02 (constant) | 0.0015 (constant) | ✅ Better (but needs adjustment) |
| Optimizer | Adam (likely) | AdamW | ✅ Better |
| Weight Decay | 1e-5 | 1e-4 | ✅ Better |
| Gradient Clipping | None | 1.0 | ✅ Critical fix |
| AMP | None | FP16/BF16 | ✅ Speedup |
| White Noise | 0.8 | 0.2 | ✅ Better |
| Dropout | 0.4 | 0.3 | ✅ Better |
| Time Masking | None | Prob 0.10, width 15 | ✅ Added |
| LayerNorm | None | Enabled | ✅ Added |
| Checkpoint Saving | Basic | Based on actual PER | ✅ Fixed |
| Logging | Basic | Comprehensive | ✅ Better |

---

## Next Steps

1. **Immediate**: Reduce LR to 0.0012 for Run #6 continuation
2. **Short-term**: Implement adaptive LR reduction when NaN gradients detected
3. **Long-term**: Find optimal LR schedule that maintains stability for full 10k steps

---

---

## Run #8: `run8_recovery` (Baseline Recovery)

**Purpose**: Revert to minimal baseline configuration to reproduce stability, then make one controlled change at a time.

**Rationale**: After multiple runs with many changes, training became unstable. Need to go back to basics and verify we can reproduce baseline stability.

### Configuration Changes

**Reverted to baseline-style**:
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

### Expected Outcome

- **Stable training** (no NaN, no explosions)
- **PER should decrease steadily** from ~0.94
- **Target**: Get below 0.35 PER, then iterate one change at a time

### Next Steps (After Recovery)

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

### Critical Fixes

**T_eff Calculation Bug Fix**:
- **Fixed**: Changed from `/` (float division) to `//` (floor division)
- **Added**: `.clamp(min=1)` to ensure T_eff >= 1
- **Applied in 3 places**: Training loop, eval loop, sample logging
- **Impact**: Should fix blank-heavy outputs and improve PER calculation

### Configuration Changes

**Dead-simple baseline**:
- **LR**: Exactly **0.001** (1e-3), constant (no scheduler, no warmup)
- **AMP**: **OFF** (until PER < 0.5, stability first)
- **Gradient clipping**: **0.5** (tighter, as recommended)
- **Dropout**: **0.4** (baseline value)
- **Optimizer**: **Adam** (not AdamW)
- **Weight decay**: **0.0** (no weight decay)
- **All augmentation**: **OFF** (reproduce baseline)

**Kept**:
- LayerNorm enabled (helps stability)
- Speed optimizations (num_workers, persistent_workers, etc.)

### Expected Outcome

- **Stable training** (no NaN, no explosions)
- **PER should decrease steadily** from ~0.94
- **No more blank-heavy outputs** (T_eff fix should help)
- **Target**: Get below 0.5 PER, then enable AMP and iterate

---

---

## Run #10: `run10_recovery` (Recovery with Speed Optimizations)

**Purpose**: Same clean baseline as Run #9, but with speed optimizations enabled.

**Changes from Run #9**:
- **cuDNN benchmark**: Enabled (faster RNN operations, ~5-15% speedup)
- **torch.compile**: Enabled (PyTorch 2.0+ model compilation, ~20-30% speedup)
- **Expected combined speedup**: ~25-40% faster training

**Configuration** (identical to Run #9):
- Same clean baseline settings
- T_eff calculation bug fixed
- All speed optimizations enabled (no behavior changes)

**Expected Outcome**:
- **Same stability as Run #9** (no behavior changes)
- **~25-40% faster training** (speed optimizations)
- **PER should decrease steadily** from ~0.94
- **No more blank-heavy outputs** (T_eff fix)

---

---

## Run #11: `run11_plateau_break` (Break Through PER Plateau)

**Goal**: Break through the ~36-38% PER plateau observed in Run #10

**Key Changes**:
1. **Prefix Beam Search**: Implemented `ctc_prefix_beam_search()` for evaluation (beam_size=20)
   - Replaces greedy decoding in evaluation loop
   - Expected 5-15% relative PER improvement
2. **AdamW Optimizer**: Switched from Adam to AdamW (decoupled weight decay)
   - `weight_decay: 1e-4` (standard for AdamW)
   - Better regularization than L2 in Adam
3. **ReduceLROnPlateau Scheduler**: Added scheduler on validation PER
   - `patience: 8` evals (800 steps)
   - `factor: 0.5` (reduce LR by 50%)
   - `min_lr: 1e-5`
   - Automatically reduces LR when PER plateaus
4. **Full SpecAugment**: Added frequency masking alongside time masking
   - Time masks: `prob=0.10`, `width=40`, `max_masks=2`
   - Frequency masks: `prob=0.10`, `width=12`, `max_masks=2`
   - Standard augmentation for speech recognition
5. **Lower Base LR**: Reduced from 1e-3 to 8e-4
   - Better convergence without instability
   - Will be further reduced by scheduler if needed
6. **Gradient Clipping**: Set to 1.5 (moderate, in 1.0-2.0 range)

**Implementation Details**:
- `src/neural_decoder/dataset.py`: Added `apply_frequency_mask()` function
- `src/neural_decoder/neural_decoder_trainer.py`: 
  - Added `ctc_prefix_beam_search()` function
  - Updated scheduler setup to support ReduceLROnPlateau
  - Updated evaluation to use beam search when enabled
  - Updated dataset loader to pass frequency mask parameters

**Expected Outcome**:
- **PER improvement from beam search**: 5-15% relative reduction
- **Plateau breaking**: ReduceLROnPlateau should reduce LR when PER stalls
- **Better generalization**: SpecAugment should help without slowing training
- **Target: PER < 0.30** (closer to baseline ~20%)

---

**Last Updated**: Run #11 (Plateau Break)
**Best PER Achieved**: 0.512 (step 4600, Run #6), 0.362 (step 4300, Run #10)
**Baseline PER**: Unknown (need to check original repo results)

