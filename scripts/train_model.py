# ============================================================================
# Run #16: run16_greedy_ema_warmcos
# ----------------------------------------------------------------------------
# Schedule: Linear warmup -> Cosine decay (no OneCycle)
#  - warmup_steps: 1500
#  - peak_lr: 0.0016
#  - lrEnd: 8e-6
# Optimizer/Reg: Adam (l2 1e-5), clip=1.0
# Perf: AMP on (BF16 on Ampere / FP16 on T4), grad accumulation 2
# Extras: EMA(0.999) applied for eval/saving. Decoding is GREEDY ONLY.
# Goal: push greedy PER < 20% (no beam).
# ============================================================================

modelName = "run16_5"
run_number = 16

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
args["run_purpose"] = (
    "Greedy-only: Warmup→Cosine, EMA=0.999, bf16/amp, grad_accum=2, clip=1.0. "
    "Slightly higher peak LR and longer cosine tail to reduce PER without beam."
)

# LR schedule (no OneCycle)
args["use_amp"] = True
args["optimizer"] = "adam"
args["l2_decay"] = 1e-5
args["weight_decay"] = args["l2_decay"]

# Warmup + Cosine params (tweaked vs run15)
args["peak_lr"] = 0.0010
args["lrEnd"] = 1e-7
args["warmup_steps"] = 1500
args["nBatch"] = 20000
args["cosine_T_max"] = args["nBatch"] - args["warmup_steps"]  # auto-decay over remainder

# EMA settings
args["use_ema"] = True
args["ema_decay"] = 0.9995

# Model
args["nUnits"] = 1024
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["use_layer_norm"] = True
args["input_dropout"] = 0.05
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True
args["gaussianSmoothWidth"] = 2.0

# SpecAugment + light noise (same as run15)
args["whiteNoiseSD"] = 0.15
args["constantOffsetSD"] = 0.05
args["time_mask_prob"] = 0.08
args["time_mask_width"] = 32
args["time_mask_max_masks"] = 2
args["freq_mask_prob"] = 0.08
args["freq_mask_width"] = 10
args["freq_mask_max_masks"] = 2

# Train loop knobs
args["grad_clip_norm"] = 1.0
args["num_workers"] = 4
args["per_ma_window"] = 300
args["eval_every"] = 100  # keep tight eval cadence early

from neural_decoder.neural_decoder_trainer import trainModel
trainModel(args)
