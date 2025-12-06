# Paper Figures and Tables

This directory contains the generated figures and tables for the paper, created by `scripts/generate_paper_figures.py`.

## Generated Files

### Figure 1: Baseline vs Clean Reproduction Table
- **CSV**: `figure1_baseline_table.csv`
- **LaTeX**: `figure1_baseline_table.tex`
- **Content**: Comparison of original baseline (speechBaseline4) with clean reproduction runs (run8_recovery, run10_recovery)
- **Columns**: Run, Scheduler, Peak LR, End LR, Precision, Grad Clip, Augmentation, Best Valid PER, Step of Best, Test PER

### Figure 2: PER vs Step Line Plot
- **PNG**: `figure2_per_vs_step.png`
- **Content**: PER over training steps for different LR schedules
- **Runs included**:
  - gru_ctc_reg5 (Constant 1e-3)
  - gru_ctc_reg6 (Constant 1.5e-3)
  - run15_warmup_cosine_safe (Warmup→Cosine)
  - run16_greedy_ema_warmcos (Warmup→Cosine + EMA)
  - run16_3 (Warmup→Cosine + EMA, lower end LR)
  - run16_4 (Warmup→Cosine + EMA + input dropout)
- **Features**: Best PER points marked, inset table with config details

### Figure 3: Regularization Ablation Table
- **CSV**: `figure3_regularization_table.csv`
- **LaTeX**: `figure3_regularization_table.tex`
- **Content**: Ablation study of regularization techniques
- **Runs**: speechBaseline4, run15_warmup_cosine_safe, run16_greedy_ema_warmcos, run16_4
- **Columns**: Run, Clip, EMA Decay, Input Dropout, Best Valid PER, Step of Best

### Figure 4: Augmentation Bar Chart
- **PNG**: `figure4_augmentation_bars.png`
- **Content**: Comparison of different augmentation strategies
- **Runs**: speechBaseline4 (heavy noise), run11_plateau_break (reduced noise), run15_warmup_cosine_safe (light SpecAugment)

### Figure 5: Decoding Cost vs PER
- **Status**: Not yet implemented (requires running eval.py with different decoders)
- **Note**: This figure requires manual evaluation runs with different beam sizes

### Figure 6: Final Recipe Summary Table
- **CSV**: `figure6_final_recipe_table.csv`
- **LaTeX**: `figure6_final_recipe_table.tex`
- **Content**: Complete configuration of the best run (run16_5)
- **Columns**: All hyperparameters and results

## Usage

Generate all figures:
```bash
python scripts/generate_paper_figures.py --output paper_figures
```

Generate specific figures:
```bash
python scripts/generate_paper_figures.py --figures 1 2 3 4 6 --output paper_figures
```

## Data Sources

The script reads from:
- `data/checkpoints/<run_name>/metrics.jsonl` - Training metrics
- `data/checkpoints/<run_name>/train.log` - Training configuration
- `data/checkpoints/<run_name>/eval*.json` - Evaluation results
- `data/checkpoints/<run_name>/args` - Saved training arguments (pickle)

## Notes

- **speechBaseline4**: Uses known baseline configuration (no metrics.jsonl available)
- **Best PER**: Extracted from metrics.jsonl (minimum PER at evaluation steps)
- **Test PER**: Extracted from eval*.json files (prefers greedy decoding)
- **Scheduler detection**: Parsed from train.log files
- **Augmentation**: Extracted from train.log or uses known baseline values

## Figure 5 (Decoding Cost)

To generate Figure 5, you need to run evaluation with different decoders:

```bash
# Greedy
python scripts/eval.py --modelPath data/checkpoints/run16_4/ --split test --out paper_figures/run16_4_greedy.json

# Beam k=10
python scripts/eval.py --modelPath data/checkpoints/run16_4/ --split test --out paper_figures/run16_4_beam10.json --use_beam_search --beam_size 10

# Beam k=20
python scripts/eval.py --modelPath data/checkpoints/run16_4/ --split test --out paper_figures/run16_4_beam20.json --use_beam_search --beam_size 20

# Repeat for run16_5
```

Then update the script to parse these results and generate the figure.

