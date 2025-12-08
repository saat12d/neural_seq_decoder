# Run #12: `run12_baseline_hybrid` - Hybrid Baseline + Modern Features

## Run Information
- **Run Number**: 12
- **Run Name**: `run12_baseline_hybrid`
- **Purpose**: Match baseline performance by using baseline LR + augmentation, while keeping modern improvements (LayerNorm, gradient clipping, SpecAugment)
- **Date Started**: 2025-12-04

## Configuration

### Learning Rate
- **lrStart**: 0.02 (baseline value, 25x higher than Run #11)
- **lrEnd**: 0.02 (constant LR, baseline style)
- **Scheduler**: None (constant LR like baseline)
- **Warmup**: 0 steps

### Optimizer
- **Type**: Adam (baseline optimizer, not AdamW)
- **l2_decay**: 1e-5 (baseline value, very low weight decay)
- **Note**: Using `l2_decay` parameter (baseline style) instead of `weight_decay`

### Architecture
- **nUnits**: 1024
- **nLayers**: 5
- **dropout**: 0.4 (baseline value)
- **use_layer_norm**: True (kept for stability with high LR)
- **input_dropout**: 0.0
- **bidirectional**: True
- **strideLen**: 4
- **kernelLen**: 32

### Augmentation (Hybrid Approach)
**Baseline Augmentation:**
- **whiteNoiseSD**: 0.8 (baseline value, strong noise)
- **constantOffsetSD**: 0.2 (baseline value, offset augmentation)
- **gaussianSmoothWidth**: 2.0 (baseline value)

**SpecAugment (Modern):**
- **time_mask_prob**: 0.10 (10% chance per mask)
- **time_mask_width**: 40
- **time_mask_max_masks**: 2
- **freq_mask_prob**: 0.10 (10% chance per mask)
- **freq_mask_width**: 12
- **freq_mask_max_masks**: 2

**Note**: Combining both baseline augmentation (noise/offset) and SpecAugment - this is aggressive but should provide strong regularization.

### Training Configuration
- **batchSize**: 64
- **nBatch**: 10000
- **use_amp**: False (FP32 for stability with high LR)
- **grad_clip_norm**: 1.5 (critical for stability with LR=0.02)
- **num_workers**: 4
- **adaptive_lr**: False (baseline didn't have it)
- **use_plateau_scheduler**: False (baseline didn't have LR scheduler)
- **use_beam_search**: False (greedy decoding during training)

## Changes from Run #11

### Reverted to Baseline
1. **Learning Rate**: 0.0008 → **0.02** (25x increase, baseline value)
2. **Optimizer**: AdamW → **Adam** (baseline optimizer)
3. **Weight Decay**: weight_decay=1e-4 → **l2_decay=1e-5** (baseline value)
4. **Augmentation**: Added back **whiteNoiseSD=0.8** and **constantOffsetSD=0.2** (baseline values)
5. **LR Scheduler**: Removed ReduceLROnPlateau (baseline had constant LR)

### Kept from Run #11
1. **LayerNorm**: Enabled (helps stability with high LR)
2. **Gradient Clipping**: 1.5 (critical for stability with LR=0.02)
3. **SpecAugment**: Time + frequency masking (modern augmentation)
4. **FP32 Training**: No AMP (stability first)

## Expected Improvements

1. **Faster Convergence**: High LR (0.02) should allow much faster learning
2. **Better Generalization**: Hybrid augmentation (baseline noise + SpecAugment) should provide strong regularization
3. **Stability**: Gradient clipping + LayerNorm should prevent explosions that occurred in early runs
4. **Baseline Performance**: Should match or beat baseline PER (~20%)

## Potential Risks

1. **High LR Instability**: LR=0.02 caused explosions in Run #1-2, but we now have:
   - Gradient clipping (1.5)
   - LayerNorm (stabilizes training)
   - These should prevent explosions

2. **Aggressive Augmentation**: Combining noise (0.8) + offset (0.2) + SpecAugment might be too much
   - Monitor training - if unstable, consider reducing augmentation

3. **Optimizer Difference**: Adam vs AdamW might behave differently with high LR
   - Baseline used Adam successfully, so this should be fine

## Success Criteria

- **Target PER**: < 0.25 (closer to baseline ~20%)
- **Stability**: No gradient explosions, no NaN losses
- **Convergence**: Should converge faster than Run #11 due to higher LR
- **Final PER**: Should be competitive with baseline (20.46%)

## Monitoring

Watch for:
- Gradient norms (should stay < 10 with clipping at 1.5)
- Loss stability (should decrease smoothly)
- PER improvement (should drop faster than Run #11)
- Early signs of instability (NaN, Inf, exploding gradients)

## Implementation Details

- **Model**: `GRUDecoder` with LayerNorm enabled
- **Loss**: CTC Loss (blank=0)
- **Decoding**: Greedy CTC decoding during training eval
- **Checkpointing**: Best model saved based on validation PER

