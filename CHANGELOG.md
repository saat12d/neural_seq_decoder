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

**Last Updated**: After Run #6, Step 5100
**Best PER Achieved**: 0.512 (step 4600, Run #6)
**Baseline PER**: Unknown (need to check original repo results)

