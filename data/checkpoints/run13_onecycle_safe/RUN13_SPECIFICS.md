# Run #13: `run13_onecycle_safe` - Safe High LR with OneCycleLR

## Run Information
- **Run Number**: 13
- **Run Name**: `run13_onecycle_safe`
- **Purpose**: Use high LR safely with OneCycleLR scheduler, reduced noise augmentation
- **Date Started**: 2025-12-04

## Configuration

### Learning Rate Schedule (OneCycleLR)
- **Scheduler**: OneCycleLR (warmup → peak → cosine anneal)
- **max_lr**: 0.01 (conservative, half of baseline 0.02)
- **pct_start**: 0.1 (10% warmup = 1000 steps)
- **div_factor**: 25.0 (start_lr = 0.01/25 = 0.0004)
- **final_div_factor**: 1000.0 (end_lr = 0.01/1000 = 0.00001)
- **anneal_strategy**: "cos" (cosine annealing)

**LR Schedule**:
- Steps 0-1000: Warmup from 0.0004 → 0.01
- Steps 1000-10000: Cosine anneal from 0.01 → 0.00001

### Optimizer
- **Type**: Adam (baseline optimizer, not AdamW)
- **l2_decay**: 1e-5 (baseline value, very low weight decay)
- **betas**: (0.9, 0.999) - default
- **eps**: 1e-8 - default

### Architecture
- **nUnits**: 1024
- **nLayers**: 5
- **dropout**: 0.4 (baseline value)
- **use_layer_norm**: True (kept for stability with high LR)
- **input_dropout**: 0.0
- **bidirectional**: True
- **strideLen**: 4
- **kernelLen**: 32

### Augmentation (Balanced)
**Reduced Noise (from Run #12)**:
- **whiteNoiseSD**: 0.3 (reduced from 0.8 to balance with SpecAugment)
- **constantOffsetSD**: 0.1 (reduced from 0.2)
- **gaussianSmoothWidth**: 2.0 (baseline value)

**SpecAugment (Modern)**:
- **time_mask_prob**: 0.10 (10% chance per mask)
- **time_mask_width**: 40
- **time_mask_max_masks**: 2
- **freq_mask_prob**: 0.10 (10% chance per mask)
- **freq_mask_width**: 12
- **freq_mask_max_masks**: 2

**Rationale**: Baseline's whiteNoiseSD=0.8 is very strong. With SpecAugment also on, reducing noise to 0.3-0.4 balances the augmentation without overwhelming the model.

### Training Configuration
- **batchSize**: 64
- **nBatch**: 10000
- **use_amp**: False (FP32 for stability)
- **grad_clip_norm**: 1.5 (critical for stability with high LR)
- **num_workers**: 4
- **adaptive_lr**: False (OneCycleLR handles LR schedule)
- **use_beam_search**: False (greedy decoding during training)

## Changes from Run #12

### Fixed Issues
1. **LR Schedule**: Constant LR=0.02 → **OneCycleLR** (warmup to 0.01, then anneal)
2. **Noise Augmentation**: whiteNoiseSD=0.8 → **0.3** (balance with SpecAugment)
3. **Offset Augmentation**: constantOffsetSD=0.2 → **0.1** (balance with SpecAugment)

### Kept from Run #12
1. **LayerNorm**: Enabled (helps stability)
2. **Gradient Clipping**: 1.5 (critical for high LR)
3. **SpecAugment**: Time + frequency masking
4. **Adam Optimizer**: With l2_decay=1e-5 (baseline style)
5. **FP32 Training**: No AMP (stability first)

## Expected Improvements

1. **Safe High LR**: OneCycleLR allows high LR (0.01) safely with warmup and anneal
2. **Faster Convergence**: High LR should allow faster learning than Run #11 (0.0008)
3. **Better Generalization**: Balanced augmentation (reduced noise + SpecAugment) should provide regularization without overwhelming
4. **Stability**: Gradient clipping + LayerNorm + warmup should prevent explosions

## Potential Risks

1. **Max LR Still High**: 0.01 is still 12.5x higher than Run #11's 0.0008
   - Mitigation: Warmup (1000 steps) + gradient clipping (1.5) + LayerNorm
   - If unstable, can reduce max_lr to 0.008 or 0.006

2. **Augmentation Balance**: 0.3 noise + SpecAugment might still be aggressive
   - Monitor training - if unstable, can reduce further to 0.2

## Success Criteria

- **Target PER**: < 0.25 (closer to baseline ~20%)
- **Stability**: No gradient explosions, no NaN losses
- **Convergence**: Should converge faster than Run #11 due to higher LR
- **Final PER**: Should be competitive with baseline (20.46%)

## Monitoring

Watch for:
- Gradient norms (should stay < 10 with clipping at 1.5)
- Loss stability (should decrease smoothly, especially during warmup)
- PER improvement (should drop faster than Run #11)
- LR schedule (should warmup smoothly, then anneal)
- Early signs of instability (NaN, Inf, exploding gradients)

## Implementation Details

- **Model**: `GRUDecoder` with LayerNorm enabled
- **Loss**: CTC Loss (blank=0)
- **Decoding**: Greedy CTC decoding during training eval
- **Checkpointing**: Best model saved based on validation PER
- **Scheduler**: OneCycleLR steps every batch (after optimizer.step())

## OneCycleLR Benefits

1. **Warmup**: Smoothly increases LR from low to high, preventing early instability
2. **High LR Period**: Allows fast learning at peak LR (0.01)
3. **Annealing**: Gradually reduces LR for fine-tuning, preventing late-stage instability
4. **Automatic**: No manual LR tuning needed - scheduler handles everything

## Next Steps if Successful

- If stable and learning well: Can try increasing max_lr to 0.012-0.015
- If PER improves but plateaus: Can add EMA weights for evaluation
- If still unstable: Reduce max_lr to 0.008 or use warmup+cosine instead

