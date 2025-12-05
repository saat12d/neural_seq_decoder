# ============================================================================
# Run #15: run15_warmup_cosine_safe
# ----------------------------------------------------------------------------
# Schedule: Linear warmup -> Cosine decay (no OneCycle)
#  - warmup_steps: 1200 (tune 1200–1500)
#  - peak_lr: 0.0015  (tune 0.001–0.002)
#  - lrEnd: 1e-5
# Optimizer/Reg: Adam (l2 1e-5), clip=1.0
# Performance: AMP on (BF16 on Ampere / FP16 on T4), grad accumulation 2
# Notes: Matches standard warmup+cosine guidance from SGDR + large-batch warmup practice.
# ============================================================================

modelName = "run15_warmup_cosine_safe"
run_number = 15

args = {}
args["outputDir"] = f"/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}"
args["datasetPath"] = "/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc"

# Core training sizes
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 32
args["gradient_accumulation_steps"] = 2  # effective batch 64

# Metadata
args["run_number"] = run_number
args["run_name"] = modelName
args["run_purpose"] = "Warmup→Cosine (no OneCycle). Safer peak LR. AMP+grad_accum. Clip=1.0."

# LR schedule (no OneCycle)
args["use_amp"] = True
args["optimizer"] = "adam"        # (or "adamw" if you prefer)
args["l2_decay"] = 1e-5
args["weight_decay"] = args["l2_decay"]

# Warmup + Cosine params
args["peak_lr"] = 0.0015
args["lrEnd"] = 1e-5
args["warmup_steps"] = 1200
args["nBatch"] = 10000
args["cosine_T_max"] = args["nBatch"] - args["warmup_steps"]  # remainder auto-decay

# Model
args["nUnits"] = 1024
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["use_layer_norm"] = True
args["input_dropout"] = 0.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True
args["gaussianSmoothWidth"] = 2.0

# SpecAugment + light noise
args["whiteNoiseSD"] = 0.2
args["constantOffsetSD"] = 0.05
args["time_mask_prob"] = 0.10
args["time_mask_width"] = 40
args["time_mask_max_masks"] = 2
args["freq_mask_prob"] = 0.10
args["freq_mask_width"] = 12
args["freq_mask_max_masks"] = 2

# Train loop knobs
args["grad_clip_norm"] = 1.0
args["num_workers"] = 4
args["per_ma_window"] = 200

from neural_decoder.neural_decoder_trainer import trainModel
trainModel(args)
