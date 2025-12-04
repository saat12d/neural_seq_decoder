
# ============================================================================
# Run #11: `run11_plateau_break` - Break Through PER Plateau
# ============================================================================
# Purpose: Break through the ~36-38% PER plateau observed in Run #10
# 
# Key Changes:
# - Prefix beam search for CTC decoding (beam_size=20)
# - AdamW optimizer with lower LR (8e-4)
# - ReduceLROnPlateau scheduler on validation PER
# - Full SpecAugment (time + frequency masking)
# 
# Target: PER < 0.30 (closer to baseline ~20%)
# ============================================================================

modelName = 'run11_plateau_break'
run_number = 11  # Explicitly mark as Run #11

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Store run metadata
args['run_number'] = run_number
args['run_name'] = modelName
args['run_purpose'] = 'Break through PER plateau with beam search, AdamW, ReduceLROnPlateau, and SpecAugment'
args['lrStart'] = 0.0008   # Lower LR (8e-4) for better convergence
args['lrEnd'] = 0.0008     # Same as start (will be reduced by ReduceLROnPlateau)
args['peak_lr'] = 0.0008   # For compatibility
args['warmup_steps'] = 0  # No warmup (ReduceLROnPlateau handles LR reduction)
args['cosine_T_max'] = 10000  # Not used (using ReduceLROnPlateau)

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

# SpecAugment: time + frequency masking
args['whiteNoiseSD'] = 0.0  # No noise
args['constantOffsetSD'] = 0.0  # No offset
args['gaussianSmoothWidth'] = 2.0
args['time_mask_prob'] = 0.10  # Enable time masking
args['time_mask_width'] = 40  # Wider masks
args['time_mask_max_masks'] = 2  # Up to 2 time masks
args['freq_mask_prob'] = 0.10  # Enable frequency masking
args['freq_mask_width'] = 12  # Frequency mask width
args['freq_mask_max_masks'] = 2  # Up to 2 frequency masks

args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True

# AdamW optimizer with lower LR
args['optimizer'] = 'adamw'  # AdamW (decoupled weight decay)
args['weight_decay'] = 1e-4  # Standard weight decay for AdamW

# Critical: AMP OFF until PER < 0.5 (stability first)
args['use_amp'] = False  # Turn AMP off for clean baseline
args['grad_clip_norm'] = 1.5  # Moderate clipping (1.0-2.0 range)
args['num_workers'] = 4

# ReduceLROnPlateau scheduler on validation PER
args['use_plateau_scheduler'] = True
args['plateau_patience'] = 8  # Wait 8 evals before reducing LR
args['plateau_factor'] = 0.5  # Reduce LR by 50%
args['plateau_min_lr'] = 1e-5  # Minimum LR

# Disable adaptive LR (using ReduceLROnPlateau instead)
args['adaptive_lr'] = False

# Prefix beam search for evaluation (better than greedy)
args['use_beam_search'] = True
args['beam_size'] = 20  # Beam width for CTC decoding

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
