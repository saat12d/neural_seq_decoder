# ============================================================================
# Run #14: `run14_onecycle_tuned` - Tuned OneCycleLR + Speed Optimizations
# ============================================================================
# Purpose:
#   Prevent Run #13 mid-cycle overshoot and stabilize training while improving speed
#   on T4/Ampere GPUs. Pairs with the updated neural_decoder_trainer.py you installed.
#
# Highlights:
#   - OneCycleLR: max_lr=0.004, pct_start=0.15, div=25, final_div=1e3
#   - AMP on (FP16 on T4; BF16 on Ampere if supported); CTCLoss in FP32
#   - Grad accumulation: 32 x 2 => effective batch 64
#   - Strong clipping: 1.0
#   - Reduced noise + SpecAugment
#   - torch.compile disabled (GRU on T4 is usually slower/fragile with it)
#   - Early-stop guard: reduce LR ceiling if PER too high at step 1200
#
# Notes:
#   - OneCycleLR total_steps is matched to optimizer steps (handled inside trainer)
#   - DataLoader tuned via args: num_workers, prefetch_factor
# ============================================================================

import os

# (Optional) You can uncomment to make the allocator settings global even before import
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")

from neural_decoder.neural_decoder_trainer import trainModel

modelName = "run14_onecycle_tuned"
run_number = 14

args = {}
args["outputDir"] = f"/home/bciuser/projects/neural_seq_decoder/data/checkpoints/{modelName}"
args["datasetPath"] = "/home/bciuser/projects/neural_seq_decoder/data/formatted/ptDecoder_ctc"

# Core dataset/model settings
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["nUnits"] = 1024
args["nLayers"] = 5
args["nBatch"] = 10000          # total training *iterations*
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256

# Batch + accumulation
args["batchSize"] = 32
args["gradient_accumulation_steps"] = 2       # effective batch = 32 * 2 = 64

# Run metadata (logged to file)
args["run_number"] = run_number
args["run_name"] = modelName
args["run_purpose"] = (
    "Tuned OneCycleLR: max_lr=0.004, earlier peak (15%), tighter clipping (1.0), "
    "reduced noise, AMP+grad_accum (T4/Ampere optimized)"
)

# OneCycleLR (trainer will align total_steps to optimizer steps w/ grad accumulation)
args["use_onecycle"] = True
args["onecycle_max_lr"] = 0.004
args["onecycle_pct_start"] = 0.15
args["onecycle_div_factor"] = 25.0
args["onecycle_final_div_factor"] = 1e3

# Early stop guard: shrink LR ceiling if validation PER is still high at the peak
args["early_stop_guard"] = True
args["early_stop_step"] = 1200
args["early_stop_per_threshold"] = 0.38
args["early_stop_reduced_max_lr"] = 0.003

# Legacy LR fields (not used with OneCycle, but kept for completeness/logging)
args["lrStart"] = 0.00016
args["lrEnd"] = 0.000004
args["peak_lr"] = 0.004
args["warmup_steps"] = 0
args["cosine_T_max"] = 10000

# Architecture details (baseline)
args["dropout"] = 0.4
args["use_layer_norm"] = True
args["input_dropout"] = 0.0

# Front-end (temporal conv) stride / kernel
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True
args["gaussianSmoothWidth"] = 2.0

# Regularization / augmentation
args["whiteNoiseSD"] = 0.2
args["constantOffsetSD"] = 0.05
args["time_mask_prob"] = 0.10
args["time_mask_width"] = 40
args["time_mask_max_masks"] = 2
args["freq_mask_prob"] = 0.10
args["freq_mask_width"] = 12
args["freq_mask_max_masks"] = 2

# Optimizer choice (Adam matches your baseline)
args["optimizer"] = "adam"
args["l2_decay"] = 1e-5   # very light weight decay, baseline-compatible

# Precision / performance
args["use_amp"] = True              # AMP on (FP16 on T4; BF16 on Ampere if supported)
args["auto_enable_amp"] = False     # unnecessary since AMP is already enabled
args["use_torch_compile"] = False   # off for GRU on T4

# Clipping
args["grad_clip_norm"] = 1.0

# Dataloader tuning
args["num_workers"] = 8
args["prefetch_factor"] = 2

# Scheduler selection
args["use_plateau_scheduler"] = False

# Adaptive LR disabled (OneCycle schedule drives LR)
args["adaptive_lr"] = False

# Decoding during eval
args["use_beam_search"] = False           # greedy for routine evals (fast)
args["beam_search_eval"] = False         # run beam on a small subset every N steps
args["beam_search_interval"] = 500
args["beam_search_subset_size"] = 100
args["beam_size"] = 10

if __name__ == "__main__":
    os.makedirs(args["outputDir"], exist_ok=True)
    trainModel(args)
