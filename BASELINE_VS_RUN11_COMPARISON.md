# Baseline vs Run 11 Configuration Comparison

## Summary
- **Baseline PER**: 20.46% (greedy decoding)
- **Run 11 PER**: 28.30% (greedy decoding) - **8.3 percentage points worse**

## Key Configuration Differences

### 1. Learning Rate (CRITICAL DIFFERENCE)
| Parameter | Baseline | Run 11 | Impact |
|-----------|----------|--------|--------|
| `lrStart` | **0.02** | **0.0008** | **25x lower!** |
| `lrEnd` | **0.02** | **0.0008** | Constant in both |

**Analysis**: The baseline uses a learning rate that's 25x higher than Run 11. This is the most significant difference and likely the primary reason for the performance gap.

### 2. Optimizer & Weight Decay
| Parameter | Baseline | Run 11 | Impact |
|-----------|----------|--------|--------|
| Optimizer | **Adam** (implied) | **AdamW** | Different optimizer |
| Weight Decay | **l2_decay: 1e-5** | **weight_decay: 1e-4** | 10x higher in Run 11 |

**Analysis**: Baseline uses Adam with very low L2 decay (1e-5), while Run 11 uses AdamW with higher weight decay (1e-4). This could affect generalization.

### 3. Data Augmentation
| Parameter | Baseline | Run 11 | Impact |
|-----------|----------|--------|--------|
| `whiteNoiseSD` | **0.8** | **0.0** | Baseline has strong noise augmentation |
| `constantOffsetSD` | **0.2** | **0.0** | Baseline has offset augmentation |
| `time_mask_prob` | **Not present** | **0.1** | Run 11 has SpecAugment time masking |
| `freq_mask_prob` | **Not present** | **0.1** | Run 11 has SpecAugment frequency masking |

**Analysis**: Baseline uses strong white noise (0.8) and constant offset (0.2), while Run 11 uses SpecAugment instead. These are fundamentally different augmentation strategies.

### 4. Architecture Differences
| Parameter | Baseline | Run 11 | Impact |
|-----------|----------|--------|--------|
| `use_layer_norm` | **Not present (False)** | **True** | Run 11 has LayerNorm |
| `input_dropout` | **Not present (0.0)** | **0.0** | Same (no input dropout) |
| `dropout` | **0.4** | **0.4** | Same |

**Analysis**: Run 11 adds LayerNorm, which should help stability but might change the optimization dynamics.

### 5. Training Configuration
| Parameter | Baseline | Run 11 | Impact |
|-----------|----------|--------|--------|
| `grad_clip_norm` | **Not present** | **1.5** | Run 11 has gradient clipping |
| `use_amp` | **Not present (False)** | **False** | Same (FP32) |
| `use_plateau_scheduler` | **Not present (False)** | **True** | Run 11 has LR scheduler |
| `num_workers` | **Not present** | **4** | Run 11 has parallel data loading |

**Analysis**: Run 11 has additional training features (gradient clipping, LR scheduler) that weren't in the baseline.

### 6. Evaluation Differences
| Aspect | Baseline | Run 11 | Impact |
|--------|----------|--------|--------|
| T_eff calculation | `(X_len - kernelLen) / strideLen` (division) | `(X_len - kernelLen) // strideLen` (floor division) | Minor difference, both convert to int32 |
| Decoding method | Greedy (implied) | Greedy | Same |

**Note**: `eval_competition.py` uses regular division `/` then converts to int32, while our `eval.py` uses floor division `//`. This should have minimal impact since both end up as integers.

## Critical Findings

### 1. Learning Rate is the Biggest Issue
The baseline uses **LR = 0.02**, which is **25x higher** than Run 11's **LR = 0.0008**. This massive difference likely explains most of the performance gap.

**Why this matters**: 
- Higher LR allows faster convergence and better exploration of the loss landscape
- The baseline was able to train successfully with LR=0.02, suggesting the model can handle it
- Our attempts to use lower LR (0.001-0.0008) were due to stability concerns, but may have been too conservative

### 2. Augmentation Strategy is Different
- **Baseline**: Strong white noise (0.8) + constant offset (0.2) - aggressive augmentation
- **Run 11**: SpecAugment (time + frequency masking) - modern but different approach

These are fundamentally different augmentation strategies. The baseline's aggressive noise augmentation might be crucial for generalization.

### 3. Optimizer Difference
- **Baseline**: Adam with L2 decay (1e-5) - traditional approach
- **Run 11**: AdamW with weight decay (1e-4) - modern but different regularization

AdamW's decoupled weight decay might interact differently with the high LR used in the baseline.

## Recommendations

### Immediate Actions:
1. **Try higher learning rate**: Test LR = 0.01 or 0.015 (closer to baseline's 0.02) with gradient clipping
2. **Match baseline augmentation**: Try whiteNoiseSD=0.8, constantOffsetSD=0.2 instead of SpecAugment
3. **Use Adam instead of AdamW**: Match baseline optimizer with l2_decay=1e-5
4. **Remove LayerNorm**: Test without LayerNorm to match baseline exactly

### Investigation Steps:
1. Check if baseline model was trained with gradient clipping (might not be in args)
2. Verify if baseline used any LR scheduling (not in args, likely constant)
3. Test a run that matches baseline config exactly: LR=0.02, Adam, whiteNoiseSD=0.8, constantOffsetSD=0.2, no LayerNorm

## Questions to Answer
1. Why did the baseline work with LR=0.02 but our runs exploded?
   - Possible answers: Different initialization, different data preprocessing, or we were too conservative
2. Is the baseline's aggressive noise augmentation necessary?
   - Test with and without to see impact
3. Does LayerNorm help or hurt when using high LR?
   - Test both configurations

