
modelName = 'gru_ctc_reg2'

args = {}
args['outputDir'] = f'/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}'
args['datasetPath'] = '/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
# Run #3: Linear warmup → cosine decay with good defaults
args['lrStart'] = 0.02  # Base LR
args['lrEnd'] = 0.005   # eta_min (or 0.001 for stronger late training shrinkage)
args['warmup_steps'] = 1000  # Warm-up steps
args['cosine_T_max'] = 9000  # Cosine T_max
args['nUnits'] = 1024
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
# Run #3: Soften augmentation - reduce noise
args['whiteNoiseSD'] = 0.4
args['constantOffsetSD'] = 0.1
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['weight_decay'] = 1e-4
args['optimizer'] = 'adamw'

# Run #3: Keep LayerNorm, reduce input dropout
args['use_layer_norm'] = True
args['input_dropout'] = 0.1

# Run #3: Softer time masking - reduced width, allow 1-2 masks
args['time_mask_prob'] = 0.10
args['time_mask_width'] = 15
args['time_mask_max_masks'] = 2

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)