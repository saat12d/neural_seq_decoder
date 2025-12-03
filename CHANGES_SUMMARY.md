# Quick Reference: Changes from Baseline

**Baseline**: https://github.com/cffan/neural_seq_decoder

## Critical Changes (What Fixed Issues)

1. **LR: 0.02 → 0.0015** (13x reduction) - Original was way too high
2. **Gradient Clipping: None → 1.0** - Prevents explosions
3. **Optimizer: Adam → AdamW** - Better weight decay handling
4. **Weight Decay: 1e-5 → 1e-4** - 10x increase
5. **Checkpoint Saving: Moving Avg → Actual PER** - Fixed bug

## Current Issue (Run #6)

- **Step 4200+**: NaN gradients appearing
- **Gradient norms**: Exploding (max 17.0)
- **Fix needed**: Reduce LR to 0.0012 or implement decay

## What Works

✅ LR 0.001-0.0015 (for ~4000 steps)  
✅ Gradient clipping at 1.0  
✅ Mixed precision (FP16/BF16)  
✅ Reduced noise (0.8 → 0.2)  
✅ Reduced dropout (0.4 → 0.3)  
✅ LayerNorm + input dropout  

## What Doesn't Work

❌ LR 0.02 (baseline) → explosions  
❌ LR 0.0015 for >4000 steps → instability  
❌ High noise (0.8) → slows learning  
❌ Warmup to high LR (0.005) → degradation  

## Best Results

- **Best PER**: 0.512 (step 4600, Run #6)
- **Baseline PER**: Unknown (need to check)

See `CHANGELOG.md` for full details.

