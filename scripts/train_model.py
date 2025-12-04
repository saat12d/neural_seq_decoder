
modelName = 'run9_faster'

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Run #9: Faster learning while maintaining stability
# Changes from run8: Higher LR + AMP + slightly less dropout
args['lrStart'] = 0.0012   # 20% higher than run8 (0.001 → 0.0012)
args['lrEnd'] = 0.0012     # Constant LR
args['peak_lr'] = 0.0012   # For compatibility
args['warmup_steps'] = 0  # No warmup
args['cosine_T_max'] = 10000  # Not used (constant LR)

args['nUnits'] = 1024
args['nLayers'] = 5
args['nBatch'] = 10000
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256

# Architecture
args['dropout'] = 0.35  # Slightly reduced from 0.4 (faster learning)
args['use_layer_norm'] = True  # Keep LayerNorm (helps stability)
args['input_dropout'] = 0.0  # No input dropout

# Minimal augmentation (keep it simple)
args['whiteNoiseSD'] = 0.0  # No noise
args['constantOffsetSD'] = 0.0  # No offset
args['gaussianSmoothWidth'] = 2.0
args['time_mask_prob'] = 0.0  # No SpecAugment

args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True

# Optimizer
args['optimizer'] = 'adam'  # Keep Adam (stable)
args['weight_decay'] = 0.0  # No weight decay

# Speed + stability
args['use_amp'] = True  # Enable AMP for ~2x speedup (we know it works)
args['grad_clip_norm'] = 1.0  # Tighter clipping (better for higher LR)
args['num_workers'] = 4

# Disable adaptive LR (keep it simple)
args['adaptive_lr'] = False

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
