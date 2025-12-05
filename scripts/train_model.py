# ============================================================================
# Run #16b: run16_sub20 (tiny tweaks from run16_greedy_ema_warmcos)
# Goal: push greedy PER < 0.20 with minimal changes
# - Keep: Adam, Warmup→Cosine (no restarts), greedy-only, EMA
# - Tweaks: slightly longer run, a hair more EMA smoothing, gentler aug,
#           tiny input dropout, longer cosine tail
# ============================================================================

import os
from neural_decoder.neural_decoder_trainer import trainModel

modelName = "run-19-run16_sub20"
run_number = 16_2

args = {}
args["outputDir"] = f"/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}"
args["datasetPath"] = "/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc"

# Core training sizes
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 32
args["gradient_accumulation_steps"] = 2  # effective batch = 64

# Metadata
args["run_number"] = run_number
args["run_name"] = modelName
args["run_purpose"] = (
    "Greedy-only: Warmup→Cosine, EMA=0.9995, bf16/amp, grad_accum=2, clip=1.0. "
    "Gentle aug + tiny input dropout; longer cosine tail to settle below 20% PER."
)

# LR schedule (same style as run16; slightly longer tail)
args["use_amp"] = True
args["optimizer"] = "adam"
args["l2_decay"] = 1e-5
args["weight_decay"] = args["l2_decay"]

args["peak_lr"] = 0.0016         # same peak as run16 (stable)
args["lrEnd"] = 6e-6             # slightly lower floor to help settle
args["warmup_steps"] = 1500
args["nBatch"] = 16000           # +4k steps vs run16 to extend cosine tail
args["cosine_T_max"] = args["nBatch"] - args["warmup_steps"]

# EMA (a touch stronger smoothing than 0.999)
args["use_ema"] = True
args["ema_decay"] = 0.9995

# Model (unchanged from run16)
args["nUnits"] = 1024
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.40
args["use_layer_norm"] = True
args["input_dropout"] = 0.05     # NEW: tiny input dropout to improve robustness
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True
args["gaussianSmoothWidth"] = 2.0

# Gentler aug (slightly reduced vs run16 to avoid over-masking as LR gets small)
args["whiteNoiseSD"] = 0.15      # was 0.2
args["constantOffsetSD"] = 0.05
args["time_mask_prob"] = 0.08    # was 0.10
args["time_mask_width"] = 32     # was 40
args["time_mask_max_masks"] = 2
args["freq_mask_prob"] = 0.08    # was 0.10
args["freq_mask_width"] = 10     # was 12
args["freq_mask_max_masks"] = 2

# Train loop knobs
args["grad_clip_norm"] = 1.0
args["num_workers"] = 4
args["per_ma_window"] = 300      # longer window to stabilize best-per detection
args["eval_every"] = 100

# IMPORTANT: train on train, early-stop on VALID; do NOT peek at test until the end.
# (Use the trainer that evaluates on 'valid' and saves 'best_model.pt' by val PER.)

if __name__ == "__main__":
    os.makedirs(args["outputDir"], exist_ok=True)
    trainModel(args)
