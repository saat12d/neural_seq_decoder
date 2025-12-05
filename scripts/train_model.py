modelName = "run19_long_sgdr_swa"
run_number = 19

args = {}
args["outputDir"] = f"/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}"
args["datasetPath"] = "/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc"

# Core sizes
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 32
args["gradient_accumulation_steps"] = 2  # eff batch 64

# Metadata
args["run_number"] = run_number
args["run_name"] = modelName
args["run_purpose"] = (
    "Long run with SGDR restarts + SWA in low-LR tails; EMA for running evals. "
    "Aim: learn longer without drift, snapshot better minima."
)

# Optim/precision
args["use_amp"] = True
args["optimizer"] = "adam"
args["l2_decay"] = 2e-5
args["weight_decay"] = args["l2_decay"]

# LR schedule: Warmup -> CosineAnnealingWarmRestarts (SGDR)
args["nBatch"] = 18000
args["warmup_steps"] = 800               # quick ramp
args["peak_lr"] = 1.2e-3
args["lrEnd"] = 1e-5                     # eta_min
# SGDR cycle: start at 1200, then grow by 1.5x each restart
args["sgdr_T0"] = 1200
args["sgdr_Tmult"] = 1.5

# EMA (kept, slightly stronger)
args["use_ema"] = True
args["ema_decay"] = 0.9997

# SWA (new)
args["use_swa"] = True
args["swa_start"] = 9000                 # start averaging after halfway point
args["swa_update_every"] = 100           # add to SWA on low-LR steps
args["swa_lr"] = 5e-5                    # SWA LR for BN update (torch SWALR style)

# Model (same as run16/17)
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

# Augmentations (slightly stronger than run17; safe for long runs)
args["whiteNoiseSD"] = 0.2
args["constantOffsetSD"] = 0.05
args["time_mask_prob"] = 0.08
args["time_mask_width"] = 12
args["time_mask_max_masks"] = 2
args["freq_mask_prob"] = 0.08
args["freq_mask_width"] = 14
args["freq_mask_max_masks"] = 2

# Train loop
args["grad_clip_norm"] = 1.0
args["num_workers"] = 4
args["per_ma_window"] = 200
args["eval_every"] = 100

# Light plateau guard (in-cycle): shorten current cycle if no PER improve
args["plateau_guard_enable"] = True
args["plateau_patience_evals"] = 6   # 600 steps with eval_every=100
args["plateau_delta"] = 0.0015       # require 0.15% PER improvement to reset patience
args["plateau_shorten_factor"] = 0.7 # shrink current SGDR period to exit faster to a restart
