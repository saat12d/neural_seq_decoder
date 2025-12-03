
modelName = 'gru_ctc_reg7'

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
# Run #7: Constant LR at 0.0015 + Adaptive LR reduction (auto-reduces when instability detected)
args['lrStart'] = 0.0015   # Start high for fast learning
args['lrEnd'] = 0.0015     # Same as start (constant base)
args['peak_lr'] = 0.0015   # Peak LR (same as constant)
args['warmup_steps'] = 0   # No warmup - start at constant LR
args['cosine_T_max'] = 10000  # Not used (constant LR), but set for compatibility
args['nUnits'] = 1024
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.3  # Reduced from 0.4 to speed learning
# Run #7: Reduced noise to speed learning
args['whiteNoiseSD'] = 0.2  # Reduced from 0.4
args['constantOffsetSD'] = 0.1
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['weight_decay'] = 1e-4
args['optimizer'] = 'adamw'

# Run #7: Keep LayerNorm, reduce input dropout
args['use_layer_norm'] = True
args['input_dropout'] = 0.1

# Run #7: Softer time masking - reduced width, allow 1-2 masks
args['time_mask_prob'] = 0.10
args['time_mask_width'] = 15
args['time_mask_max_masks'] = 2

# Run #7: Adaptive LR reduction (NEW - automatically reduces LR when instability detected)
args['adaptive_lr'] = True  # Enable adaptive LR reduction
args['lr_reduction_factor'] = 0.8  # Reduce LR by 20% each time (0.0015 → 0.0012 → 0.0010 → ...)
args['lr_reduction_threshold'] = 10.0  # Reduce if grad_norm > this (even after clipping)
args['min_lr'] = 0.0005  # Don't reduce below this (safety floor)
args['max_lr_reductions'] = 5  # Maximum number of LR reductions (prevents over-reduction)

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)