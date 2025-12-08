# Run #11: `run11_plateau_break` - Complete Configuration Details

**Run Number**: 11  
**Run Name**: `run11_plateau_break`  
**Purpose**: Break through the ~36-38% PER plateau observed in Run #10  
**Date Created**: 2024

---

## Objective

Break through the PER plateau at ~36-38% using:
1. Prefix beam search for better CTC decoding
2. AdamW optimizer with lower LR
3. ReduceLROnPlateau scheduler for automatic LR reduction
4. Full SpecAugment (time + frequency masking)

**Target**: PER < 0.30 (closer to baseline ~20%)

---

## Complete Configuration

### Model Architecture
- **Model**: GRU Decoder
- **Layers**: 5
- **Units**: 1024
- **Bidirectional**: True
- **Input Features**: 256
- **Output Classes**: 40 (+ 1 blank = 41)
- **Stride Length**: 4
- **Kernel Length**: 32
- **Dropout**: 0.4
- **Layer Norm**: True
- **Input Dropout**: 0.0

### Training Hyperparameters
- **Optimizer**: AdamW (decoupled weight decay)
- **Base Learning Rate**: 0.0008 (8e-4)
- **Learning Rate End**: 0.0008 (will be reduced by scheduler)
- **Weight Decay**: 1e-4
- **Gradient Clipping**: 1.5 (max_norm)
- **Batch Size**: 64
- **Total Batches**: 10000
- **Seed**: 0
- **Mixed Precision (AMP)**: False (disabled for stability)

### Learning Rate Schedule
- **Scheduler**: ReduceLROnPlateau
- **Mode**: 'min' (minimize PER)
- **Patience**: 8 evals (800 steps)
- **Factor**: 0.5 (reduce LR by 50%)
- **Min LR**: 1e-5
- **Warmup Steps**: 0 (no warmup)
- **Cosine Decay**: Disabled (using ReduceLROnPlateau)

### Data Augmentation (SpecAugment)
- **White Noise SD**: 0.0 (disabled)
- **Constant Offset SD**: 0.0 (disabled)
- **Gaussian Smooth Width**: 2.0

#### Time Masking
- **Probability**: 0.10 (10% chance per mask)
- **Max Width**: 40 time steps
- **Max Masks**: 2

#### Frequency Masking
- **Probability**: 0.10 (10% chance per mask)
- **Max Width**: 12 frequency bins
- **Max Masks**: 2

### CTC Decoding
- **Method**: Prefix Beam Search
- **Beam Size**: 20
- **Blank Index**: 0
- **Greedy Decoding**: Disabled (using beam search)

### Dataset
- **Path**: `/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc`
- **Sequence Length**: 150
- **Max Time Series Length**: 1200
- **Data Workers**: 4
- **Pin Memory**: True
- **Persistent Workers**: True

### Other Settings
- **Adaptive LR**: False (using ReduceLROnPlateau instead)
- **Output Directory**: `data/checkpoints/run11_plateau_break`

---

## Changes from Run #10

1. **Prefix Beam Search**: Replaced greedy CTC decoding with beam search (beam_size=20)
2. **AdamW Optimizer**: Switched from Adam to AdamW with weight_decay=1e-4
3. **Lower Base LR**: Reduced from 1e-3 to 8e-4
4. **ReduceLROnPlateau**: Added scheduler on validation PER (patience=8, factor=0.5)
5. **Frequency Masking**: Added frequency masks alongside time masks (full SpecAugment)
6. **Gradient Clipping**: Set to 1.5 (moderate, in 1.0-2.0 range)

---

## Expected Improvements

1. **Beam Search**: 5-15% relative PER improvement over greedy decoding
2. **ReduceLROnPlateau**: Automatically breaks plateau by reducing LR when PER stalls
3. **SpecAugment**: Improves generalization without significantly slowing training
4. **Lower LR**: Better convergence without instability

---

## Implementation Files

- **Training Script**: `scripts/train_model.py`
- **Trainer**: `src/neural_decoder/neural_decoder_trainer.py`
- **Dataset**: `src/neural_decoder/dataset.py`
- **Documentation**: `RUN_TRACKING.md`, `CHANGELOG.md`

---

## Key Code Changes

### New Functions
- `ctc_prefix_beam_search()`: Prefix beam search for CTC decoding
- `apply_frequency_mask()`: Frequency masking for SpecAugment

### Modified Functions
- `getDatasetLoaders()`: Added frequency mask parameters
- `trainModel()`: Added ReduceLROnPlateau scheduler support
- Evaluation loop: Uses beam search when `use_beam_search=True`

---

## Monitoring

- **Metrics File**: `data/checkpoints/run11_plateau_break/metrics.jsonl`
- **Log File**: `data/checkpoints/run11_plateau_break/train.log`
- **Checkpoints**: Saved based on best PER
- **Evaluation**: Every 100 steps

---

## Success Criteria

- **Primary**: PER < 0.30 (closer to baseline ~20%)
- **Secondary**: Stable training (no NaN gradients, no explosions)
- **Tertiary**: LR reductions trigger appropriately when PER plateaus

---

**Last Updated**: Run #11 Configuration Created

