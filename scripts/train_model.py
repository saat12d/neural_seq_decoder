
# ============================================================================
# Run #14: `run14_onecycle_tuned` - Tuned OneCycleLR + Light Regularization
# ============================================================================
# Purpose: Fix Run #13's mid-cycle overshoot with lower max_lr, earlier peak,
#          tighter clipping, reduced noise, and auto-enable AMP
# 
# Key Changes from Run #13:
# - Lower max_lr: 0.004 (down from 0.01) to prevent overshoot
# - Earlier peak: pct_start=0.15 (reach peak at 1500 steps, then longer decay)
# - Tighter clipping: grad_clip_norm=1.0 (down from 1.5)
# - Reduced noise: whiteNoiseSD=0.2, constantOffsetSD=0.05
# - Auto-enable AMP: Start FP32, enable at step 1500 if PER < 0.45
# - Early stop guard: Reduce max_lr to 0.003 if PER > 0.38 by step 1200
# 
# Target: PER < 0.25 (match or beat baseline ~20%)
# ============================================================================

modelName = 'run14_onecycle_tuned'
run_number = 14  # Explicitly mark as Run #14

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Store run metadata
args['run_number'] = run_number
args['run_name'] = modelName
args['run_purpose'] = 'Tuned OneCycleLR: max_lr=0.004, earlier peak (15%), tighter clipping (1.0), reduced noise, auto-AMP'

# OneCycleLR scheduler configuration (tuned to prevent overshoot)
args['use_onecycle'] = True  # Enable OneCycleLR
args['onecycle_max_lr'] = 0.004  # Lower max LR (down from 0.01) to prevent overshoot
args['onecycle_pct_start'] = 0.15  # 15% warmup (1500 steps) - reach peak earlier, then longer decay
args['onecycle_div_factor'] = 25.0  # start_lr = max_lr / 25 = 0.00016
args['onecycle_final_div_factor'] = 1e3  # end_lr = max_lr / 1000 = 0.000004

# Early stop guard: Reduce max_lr if PER > 0.38 by step 1200
args['early_stop_guard'] = True
args['early_stop_step'] = 1200
args['early_stop_per_threshold'] = 0.38
args['early_stop_reduced_max_lr'] = 0.003

# For compatibility with existing code (not used with OneCycleLR)
args['lrStart'] = 0.00016  # Initial LR (max_lr / div_factor)
args['lrEnd'] = 0.000004  # Final LR (max_lr / final_div_factor)
args['peak_lr'] = 0.004  # Max LR for OneCycleLR
args['warmup_steps'] = 0  # Not used (OneCycleLR handles warmup)
args['cosine_T_max'] = 10000  # Not used

args['nUnits'] = 1024
args['nLayers'] = 5
args['nBatch'] = 10000
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256

# Baseline architecture (keep as original)
args['dropout'] = 0.4  # Baseline value
args['use_layer_norm'] = True  # Keep LayerNorm (helps stability)
args['input_dropout'] = 0.0  # No input dropout

# Light regularization: Further reduced noise + SpecAugment
args['whiteNoiseSD'] = 0.2  # Reduced from 0.3 (light noise, balance with SpecAugment)
args['constantOffsetSD'] = 0.05  # Reduced from 0.1 (light offset)
args['gaussianSmoothWidth'] = 2.0  # Baseline value
args['time_mask_prob'] = 0.10  # SpecAugment: time masking (10% chance per mask)
args['time_mask_width'] = 40  # Time mask width (30-50 range recommended)
args['time_mask_max_masks'] = 2  # Up to 2 time masks
args['freq_mask_prob'] = 0.10  # SpecAugment: frequency masking (10% chance per mask)
args['freq_mask_width'] = 12  # Frequency mask width (8-15 range recommended)
args['freq_mask_max_masks'] = 2  # Up to 2 frequency masks (1-2 recommended)

args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True

# Baseline optimizer: Adam with l2_decay (not AdamW)
args['optimizer'] = 'adam'  # Baseline optimizer (Adam, not AdamW)
args['l2_decay'] = 1e-5  # Baseline weight decay (very low, as in baseline)

# Precision: Start FP32, auto-enable AMP at step 1500 if PER < 0.45
args['use_amp'] = False  # Start FP32 for stability
args['auto_enable_amp'] = True  # Enable auto-enable AMP
args['auto_amp_step'] = 1500  # Enable AMP at this step
args['auto_amp_per_threshold'] = 0.45  # Only enable if PER < this threshold

# Tighter gradient clipping (classic stabilizer for deep RNNs)
args['grad_clip_norm'] = 1.0  # Tighter clipping (down from 1.5)
args['num_workers'] = 4

# No other LR schedulers (using OneCycleLR)
args['use_plateau_scheduler'] = False
args['plateau_patience'] = 8  # Not used
args['plateau_factor'] = 0.5  # Not used
args['plateau_min_lr'] = 1e-5  # Not used

# Disable adaptive LR (OneCycleLR handles LR schedule)
args['adaptive_lr'] = False

# CTC decoding: Greedy every 100 steps, optional beam search every 500 steps on subset
args['use_beam_search'] = False  # Greedy for regular evals (every 100 steps)
args['beam_search_eval'] = True  # Enable beam search on subset every 500 steps
args['beam_search_interval'] = 500  # Run beam search every N steps
args['beam_search_subset_size'] = 100  # Evaluate on subset of test set
args['beam_size'] = 10  # Beam width for subset evals

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
