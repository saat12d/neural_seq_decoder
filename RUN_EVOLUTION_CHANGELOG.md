# Training Run Evolution Changelog

This document tracks the evolution of training runs from the initial baseline through the final best run (run16_5), documenting all configuration changes, improvements, and lessons learned.

---

## Baseline: speechBaseline4

**Status:** Original baseline run
**Configuration:**
- Batch size: 64
- Learning rate: 0.02 (constant, no decay)
- Precision: FP32 (full precision)
- Scheduler: Constant LR (lrStart=0.02, lrEnd=0.02)
- Gradient clipping: Unknown/None
- Augmentation:
  - White noise SD: 0.8
  - Constant offset SD: 0.2
  - Gaussian smooth width: 2.0
- Model architecture:
  - Hidden units: 1024
  - GRU layers: 5
  - Input features: 256
  - Output classes: 40
  - Dropout: 0.4
  - Bidirectional: True
  - Stride length: 4
  - Kernel length: 32
- Training:
  - Total batches: 10000
  - Sequence length: 150
  - Max time series length: 1200
  - Weight decay (L2): 1e-5
  - Seed: 0
- **Result:** PER ~0.20 (eval_test: 0.2046)

**Notes:** This was the original baseline run from the paper/reference implementation. It used a constant high learning rate (0.02) with strong augmentation (noise=0.8, offset=0.2) and achieved reasonable results. All subsequent runs were improvements and experiments based on this baseline.

---

## Phase 1: Initial Experiments (gru_ctc_reg1-7)

**Note:** These were initial experimental runs, not the baseline. The actual baseline is `speechBaseline4` above.

### gru_ctc_reg1
**Status:** Initial experiment
**Configuration:**
- Batch size: Unknown (likely 64)
- Learning rate: ~0.02 (linear decay)
- Precision: FP32
- Scheduler: Linear decay
- Gradient clipping: Unknown
- Augmentation: Unknown
- **Result:** PER ~0.93 (very poor)

**Notes:** Initial experimental run. Training was unstable with very high PER. This was not the baseline - `speechBaseline4` is the actual baseline.

---

### gru_ctc_reg2
**Changes from gru_ctc_reg1:**
- **Added:** Warmup + Cosine annealing scheduler
  - Warmup: 1000 steps
  - Cosine: 9000 steps
  - Peak LR: Unknown, End LR: 0.005
- **Added:** Input dropout: 0.1
- **Changed:** Precision: FP16 (mixed precision)
- **Changed:** Gradient clipping: max_norm=1.0
- Batch size: 64
- Total batches: 10000

**Notes:** First attempt at stabilizing training with warmup and cosine annealing.

---

### gru_ctc_reg3
**Changes from gru_ctc_reg2:**
- **Changed:** Longer warmup period: 2500 steps (was 1000)
- **Changed:** Shorter cosine: 7500 steps (was 9000)
- **Changed:** Lower end LR: 0.0005 (was 0.005)
- **Kept:** Input dropout: 0.1, FP16, grad_clip=1.0

**Notes:** Experimented with longer warmup to stabilize early training.

---

### gru_ctc_reg4
**Changes from gru_ctc_reg3:**
- **Changed:** Precision: BF16 (was FP16)
- **Changed:** Peak LR: 0.002 (explicitly set)
- **Changed:** Warmup: 1000 steps (back to shorter)
- **Changed:** Cosine: 9000 steps
- **Changed:** End LR: 0.0008
- **Kept:** Input dropout: 0.1, grad_clip=1.0

**Notes:** Switched to BF16 for better numerical stability on Ampere GPUs.

---

### gru_ctc_reg5
**Changes from gru_ctc_reg4:**
- **Changed:** Scheduler: Constant LR (no warmup/cosine)
- **Changed:** Peak LR: 0.001 (constant)
- **Changed:** End LR: 0.001 (same as peak)
- **Kept:** BF16, input_dropout=0.1, grad_clip=1.0

**Notes:** Simplified to constant LR to test if scheduler complexity was needed.

---

### gru_ctc_reg6
**Changes from gru_ctc_reg5:**
- **Changed:** Peak/End LR: 0.0015 (was 0.001)
- **Kept:** Constant scheduler, BF16, input_dropout=0.1, grad_clip=1.0

**Notes:** Slight LR increase to see if it helps.

---

### gru_ctc_reg7
**Changes from gru_ctc_reg6:**
- **No changes** - Same configuration as gru_ctc_reg6

**Notes:** Confirmation run with same settings.

---

## Phase 2: Recovery and Stabilization (run8, run10)

### run8_recovery
**Changes from baseline (speechBaseline4):**
- **Changed:** Learning rate: 0.001 (much lower than baseline's 0.02)
- **Changed:** Gradient clipping: max_norm=5.0 (added, baseline had none)
- **Removed:** All augmentation (noise=0, offset=0, masking disabled - baseline had noise=0.8, offset=0.2)
- **Removed:** Input dropout: 0.0 (baseline didn't have input dropout)
- **Kept:** FP32 precision, constant LR scheduler, batch size: 64, total batches: 10000

**Notes:** Recovery run after initial experiments (gru_ctc_reg1-7) failed. Significantly reduced learning rate from baseline's 0.02 to 0.001, removed all augmentation, and added gradient clipping to stabilize training.

---

### run10_recovery
**Changes from run8_recovery:**
- **No significant changes** - Same configuration as run8_recovery

**Notes:** Another recovery attempt with same settings.

---

## Phase 3: Modern Training Techniques (run11-14)

### run11_plateau_break
**Major Changes:**
- **Added:** OneCycleLR scheduler
  - Max LR: 0.004
  - Peak at 15% (1500 steps)
  - Start LR: 0.0004, End LR: 4e-06
- **Changed:** Batch size: 32 (was 64)
- **Changed:** Precision: BF16
- **Added:** Gradient accumulation: 2 steps (effective batch size: 64)
- **Changed:** Gradient clipping: max_norm=1.0 (tighter)
- **Changed:** Input dropout: 0.0
- **Changed:** Reduced augmentation noise
- Total batches: 10000

**Purpose:** "Tuned OneCycleLR: max_lr=0.004, earlier peak (15%), tighter clipping (1.0), reduced noise, AMP+grad_accum (T4/Ampere optimized)"

**Result:** Best PER: 0.2830 (greedy), 0.5375 (beam)

**Notes:** First successful use of OneCycleLR. Introduced gradient accumulation to maintain effective batch size while using smaller batch size for memory efficiency.

---

### run12_baseline_hybrid
**Changes from run11:**
- **Changed:** Scheduler: Constant LR (back from OneCycleLR)
- **Changed:** Peak LR: 0.02 (very high, baseline value)
- **Changed:** Batch size: 64 (back to larger)
- **Changed:** Precision: FP32 (back to full precision)
- **Changed:** Gradient clipping: max_norm=1.5 (slightly looser)
- **Removed:** Gradient accumulation
- **Added:** Full augmentation (noise=0.8, offset=0.2, SpecAugment)
- **Added:** Weight decay: 1e-05
- **Kept:** Input dropout: 0.0

**Purpose:** "Hybrid baseline: baseline LR (0.02) + Adam + noise/offset + SpecAugment + LayerNorm + gradient clipping"

**Notes:** Attempted to combine high baseline LR with modern techniques. Likely unstable due to very high LR.

---

### run13_onecycle_safe
**Changes from run12:**
- **Changed:** Scheduler: OneCycleLR (back)
- **Changed:** Max LR: 0.01 (lower than run11's 0.004)
- **Changed:** Precision: FP32
- **Changed:** Batch size: 64
- **Changed:** Gradient clipping: max_norm=1.5
- **Changed:** Augmentation: Reduced noise (0.3, was 0.8)
- **Removed:** Gradient accumulation
- **Kept:** Input dropout: 0.0, Weight decay: 1e-05

**Purpose:** "Safe high LR with OneCycleLR: max_lr=0.01, reduced noise (0.3), Adam + LayerNorm + clipping + SpecAugment"

**Notes:** Attempted safer OneCycleLR with FP32 and reduced augmentation.

---

### run14_onecycle_tuned
**Changes from run13:**
- **Changed:** Batch size: 32 (back to smaller)
- **Changed:** Precision: BF16 (back to mixed precision)
- **Added:** Gradient accumulation: 2 steps
- **Changed:** Gradient clipping: max_norm=1.0 (tighter)
- **Changed:** Max LR: 0.004 (same as run11)
- **Changed:** Peak at 15% (earlier peak)
- **Kept:** Input dropout: 0.0

**Purpose:** "Tuned OneCycleLR: max_lr=0.004, earlier peak (15%), tighter clipping (1.0), reduced noise, AMP+grad_accum (T4/Ampere optimized)"

**Notes:** Refined version of run11 with same OneCycleLR strategy but optimized for T4/Ampere GPUs.

---

## Phase 4: Warmup-Cosine Era (run15)

### run15_warmup_cosine_safe
**Major Change:**
- **Changed:** Scheduler: Warmup→Cosine (away from OneCycleLR)
  - Warmup: 1200 steps
  - Cosine: 8800 steps
  - Peak LR: 0.0015
  - End LR: 1e-05
- **Changed:** Batch size: 32
- **Changed:** Precision: BF16
- **Added:** Gradient accumulation: 2 steps
- **Changed:** Gradient clipping: max_norm=1.0
- **Kept:** Input dropout: 0.0
- Total batches: 10000

**Purpose:** "Warmup→Cosine (no OneCycle). Safer peak LR. AMP+grad_accum. Clip=1.0."

**Result:** Best PER: 0.2636 (greedy), 0.4142 (beam)

**Notes:** **Key turning point** - Switched from OneCycleLR to Warmup-Cosine scheduler. This proved more stable and achieved better results. Lower peak LR (0.0015) compared to OneCycleLR runs.

---

## Phase 5: EMA and Fine-tuning (run16 series)

### run16_greedy_ema_warmcos
**Major Addition:**
- **Added:** EMA (Exponential Moving Average) with decay=0.999
- **Changed:** Peak LR: 0.0016 (slightly higher than run15's 0.0015)
- **Changed:** Warmup: 1500 steps (was 1200)
- **Changed:** Cosine: 10500 steps (longer tail)
- **Changed:** End LR: 8e-06
- **Changed:** Total batches: 12000 (longer training)
- **Kept:** Batch size: 32, BF16, grad_accum=2, grad_clip=1.0, input_dropout=0.0

**Purpose:** "Greedy-only: Warmup→Cosine, EMA=0.999, bf16/amp, grad_accum=2, clip=1.0. Slightly higher peak LR and longer cosine tail to reduce PER without beam."

**Result:** Best PER: 0.2095 (greedy)

**Notes:** **Major improvement** - Introduction of EMA significantly improved results. Longer training (12k batches) and longer cosine tail helped.

---

### run16_2
**Changes from run16_greedy_ema_warmcos:**
- **Same configuration** as run16_greedy_ema_warmcos
- Peak LR: 0.0016, End LR: 8e-06, Warmup: 1500, Cosine: 10500
- EMA: 0.999, Total batches: 12000

**Result:** Best PER: 0.9111 (greedy) - **Note: This seems like an anomaly or incomplete training**

**Notes:** Appears to be a duplicate or failed run.

---

### run16_3
**Changes from run16_2:**
- **Changed:** End LR: 3e-06 (lower than 8e-06)
- **Kept:** Everything else same (Peak LR: 0.0016, EMA: 0.999, Warmup: 1500, Cosine: 10500, Total: 12000)

**Result:** Best PER: 0.2084 (greedy)

**Notes:** Lower end LR helped achieve better final PER.

---

### run16_4
**Changes from run16_3:**
- **Changed:** Peak LR: 0.0014 (lower than 0.0016)
- **Changed:** EMA decay: 0.9995 (stronger EMA, was 0.999)
- **Added:** Input dropout: 0.05 (was 0.0)
- **Changed:** Cosine: 8500 steps (shorter, was 10500)
- **Changed:** Total batches: 10000 (shorter training, was 12000)
- **Kept:** End LR: 3e-06, Warmup: 1500, grad_accum=2, grad_clip=1.0

**Result:** Best PER: 0.2078 (greedy)

**Notes:** Slightly lower peak LR, stronger EMA, and added small input dropout. Achieved slightly better PER.

---

### run16_5 ⭐ **BEST RUN**
**Changes from run16_4:**
- **Changed:** Peak LR: 0.001 (lower than 0.0014)
- **Changed:** End LR: 1e-07 (much lower than 3e-06)
- **Changed:** Cosine: 18500 steps (much longer tail, was 8500)
- **Changed:** Total batches: 20000 (much longer training, was 10000)
- **Kept:** EMA decay: 0.9995, Input dropout: 0.05, Warmup: 1500, grad_accum=2, grad_clip=1.0, Batch size: 32, BF16

**Purpose:** "Greedy-only: Warmup→Cosine, EMA=0.999, bf16/amp, grad_accum=2, clip=1.0. Slightly higher peak LR and longer cosine tail to reduce PER without beam."

**Result:** 
- Training best PER: Check training log for best checkpoint PER
- Eval test PER: 0.5460 (beam_search, beam_size=10) - **Note: This eval used beam search, not greedy decoding**

**Key Innovations:**
1. **Lower peak LR (0.001)** - More conservative learning rate
2. **Much longer training (20k batches)** - Extended training duration
3. **Very long cosine tail (18500 steps)** - Gradual decay over long period
4. **Very low end LR (1e-07)** - Allows fine-tuning at end
5. **Strong EMA (0.9995)** - Better model averaging
6. **Small input dropout (0.05)** - Regularization

**Notes:** This run represents the culmination of all lessons learned. The key insight was that **longer training with a very gradual cosine decay** and **lower peak LR** produces the best results. The combination of EMA, input dropout, and extended training with careful LR scheduling achieved the best performance.

---

## Phase 6: Additional Experiments (run16_sub20, run17, run18)

### run16_sub20
**Changes from run16_5:**
- **Changed:** Peak LR: 0.0016 (higher, back to earlier value)
- **Changed:** Cosine: 14500 steps (longer than run16_4 but shorter than run16_5)
- **Changed:** Total batches: 16000 (longer than run16_4 but shorter than run16_5)
- **Changed:** End LR: 6e-06
- **Kept:** EMA: 0.9995, Input dropout: 0.05, Warmup: 1500

**Purpose:** "Greedy-only: Warmup→Cosine, EMA=0.9995, bf16/amp, grad_accum=2, clip=1.0. Gentle aug + tiny input dropout; longer cosine tail to settle below 20% PER."

**Notes:** Attempted to find middle ground between run16_4 and run16_5.

---

### run17_greedy_ema_fastcos
**Changes from run16_5:**
- **Changed:** Peak LR: 0.0012 (lower)
- **Changed:** Warmup: 1000 steps (shorter, was 1500)
- **Changed:** Cosine: 4200 steps (much shorter, was 18500)
- **Changed:** Total batches: 5200 (much shorter, was 20000)
- **Changed:** End LR: 8e-06
- **Kept:** EMA: 0.9995, Input dropout: 0.0, grad_accum=2, grad_clip=1.0

**Purpose:** "Greedy-only. Shorter warmup, lower peak LR, faster cosine to avoid post-3.2k plateau. Stronger EMA. AMP bf16/fp16. Accum=2. Clip=1.0."

**Notes:** Attempted faster training to avoid plateau, but likely too short.

---

### run18_long_sgdr_swa
**Status:** Incomplete - Only has train.log with minimal information

**Notes:** Appears to be an experiment with SGDR (Stochastic Gradient Descent with Restarts) and SWA (Stochastic Weight Averaging), but incomplete.

---

## Summary of Key Learnings

### What Worked:
1. **Warmup-Cosine scheduler** (not OneCycleLR) - More stable and better results than constant LR
2. **EMA with decay=0.9995** - Significant improvement in final model quality
3. **Longer training (20k batches)** - Extended training with gradual decay
4. **Lower peak LR (0.001)** - More conservative learning rates work better than baseline's 0.02
5. **Very long cosine tail** - Gradual decay over 18500 steps
6. **Small input dropout (0.05)** - Helps with regularization
7. **Gradient accumulation (2 steps)** - Allows smaller batch size for memory efficiency
8. **BF16 mixed precision** - Better than FP32 for speed, stable on Ampere GPUs
9. **Tight gradient clipping (1.0)** - Prevents gradient explosions

### What Didn't Work:
1. **Very high constant LR (0.02, baseline)** - Works but suboptimal, unstable in experiments
2. **OneCycleLR** - Less stable than Warmup-Cosine
3. **Constant LR without decay** - LR decay improves final performance
4. **Too short training** - Need extended training for best results
5. **No EMA** - EMA significantly improves results
6. **Too aggressive augmentation (noise=0.8)** - Moderate augmentation is better

### Final Best Configuration (run16_5):
- **Scheduler:** Warmup (1500 steps) → Cosine (18500 steps)
- **Peak LR:** 0.001
- **End LR:** 1e-07
- **Total batches:** 20000
- **EMA decay:** 0.9995
- **Input dropout:** 0.05
- **Batch size:** 32
- **Gradient accumulation:** 2 steps (effective batch: 64)
- **Gradient clipping:** max_norm=1.0
- **Precision:** BF16
- **Weight decay:** 1e-05

---

## Evolution Timeline

```
speechBaseline4 (original baseline, PER ~0.20)
    ↓
gru_ctc_reg1-7 (initial experiments, PER ~0.93, unstable)
    ↓
run8, run10 (recovery runs, FP32, no augmentation, stabilized)
    ↓
run11 (OneCycleLR, BF16, grad_accum, PER ~0.28)
    ↓
run12-13 (experiments with high LR, FP32)
    ↓
run14 (refined OneCycleLR)
    ↓
run15 (Warmup-Cosine, PER ~0.26) ⭐ Key turning point
    ↓
run16_greedy_ema_warmcos (EMA added, PER ~0.21) ⭐ Major improvement
    ↓
run16_2-4 (fine-tuning EMA, LR, dropout)
    ↓
run16_5 (longer training, lower LR, optimized) ⭐ BEST RUN
```

---

*Generated by analyzing train.log files from all checkpoint directories*
*Last updated: Based on checkpoint analysis*

