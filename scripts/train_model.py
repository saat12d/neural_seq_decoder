
modelName = 'run10_recovery'

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Run #10: Recovery with speed optimizations
# Same clean baseline as run9, but with cuDNN benchmark + torch.compile enabled
# Based on expert recommendations: strip back to basics, fix T_eff calculation
args['lrStart'] = 0.001   # Exactly 1e-3, constant (no scheduler)
args['lrEnd'] = 0.001     # Same as start (constant LR)
args['peak_lr'] = 0.001   # For compatibility
args['warmup_steps'] = 0  # No warmup
args['cosine_T_max'] = 10000  # Not used (constant LR)

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

# All augmentation OFF (reproduce baseline)
args['whiteNoiseSD'] = 0.0  # No noise
args['constantOffsetSD'] = 0.0  # No offset
args['gaussianSmoothWidth'] = 2.0
args['time_mask_prob'] = 0.0  # No SpecAugment

args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True

# Dead simple optimizer (baseline)
args['optimizer'] = 'adam'  # Adam (not AdamW)
args['weight_decay'] = 1e-5  # Baseline: l2_decay: 1e-5 (very low weight decay)

# Critical: AMP OFF until PER < 0.5 (stability first)
args['use_amp'] = False  # Turn AMP off for clean baseline
args['grad_clip_norm'] = 5.0  # Baseline-style clipping (not 0.5 - that was too tight)
args['num_workers'] = 4

# Disable adaptive LR (no scheduler)
args['adaptive_lr'] = False

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
