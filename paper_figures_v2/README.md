# Paper Figures and Tables (Revised Spec)

This directory contains figures and tables generated according to the revised paper spec.

## Generated Files

### Figure 1: Baseline vs. Reproduction Table
- **CSV**: `figure1_baseline_table.csv`
- **LaTeX**: `figure1_baseline_table.tex`
- **Content**: Compact 3-row table comparing original baseline (speechBaseline4) with reproduction runs (run8_recovery, run10_recovery)
- **Columns**: Run, Scheduler, Peak LR, End LR, Precision, Grad Clip, Best Valid PER, Test PER
- **Note**: speechBaseline4 uses test PER since no metrics.jsonl available

### Figure 2: LR & Scheduler Multi-Panel Figure
- **PNG**: `figure2_lr_scheduler_panels.png`
- **Content**: 3-panel figure
  - **Panel A**: Constant-LR sweep (1e-3, 1.5e-3) - PER vs step
  - **Panel B**: Warmup→Cosine variants (Run15, Run16, 16_3, 16_4, 16_5) - PER vs step
  - **Panel C**: LR(t) profiles for best two schedules (run16_4, run16_5)
- **Features**: Compact layout, all panels in one figure

### Figure 3: Optimizer & Weight Decay Table
- **CSV**: `figure3_optimizer_table.csv`
- **LaTeX**: `figure3_optimizer_table.tex`
- **Content**: Comparison of Adam vs AdamW (if available) under same schedule
- **Columns**: Run, Optimizer, Weight Decay, Schedule, Best Val PER, Test PER
- **Note**: Only includes runs where optimizer info is available

### Figure 4: Regularization Ablation
- **Table CSV**: `figure4_regularization_table.csv`
- **Plot PNG**: `figure4_regularization_plot.png`
- **Content**: 
  - Table: Ablation of clipping, EMA, and dropout
  - Plot: EMA decay sweep (0, 0.999, 0.9995) showing best val PER bars
- **Runs**: speechBaseline4, run15, run16_greedy_ema_warmcos, run16_4

### Figure 5: Data Augmentation Bars
- **PNG**: `figure5_augmentation.png`
- **Content**: Bar chart comparing heavy baseline noise vs gentle noise + SpecAugment
- **Runs**: speechBaseline4 (heavy), run11_plateau_break (gentle), run15 (gentle + SpecAugment)
- **Fix Applied**: Now correctly uses test PER (0.2046) instead of competition PER (2.799) for speechBaseline4
- **Note**: Includes caption about over-masking slowing convergence

### Figure 6: Decoding Comparison
- **PNG**: `figure6_decoding.png`
- **Content**: Small bar chart comparing Greedy vs Beam search (if eval results available)
- **Note**: Requires eval results with different beam sizes to be meaningful

### Figure 7: Best Configuration Table
- **CSV**: `figure7_best_config_table.csv`
- **LaTeX**: `figure7_best_config_table.tex`
- **Content**: Concise "recipe" table for Run16_5 with all key hyperparameters
- **Columns**: All hyperparams + Best Val PER + Test PER

### Run16_5 Training Curves
Additional detailed training curves for the best run (run16_5):
- **Loss Curve**: `run16_5_loss_curve.png` - CTC loss over training (train + validation)
- **PER Curve**: `run16_5_per_curve.png` - PER over training with best point marked
- **LR Schedule**: `run16_5_lr_schedule.png` - Learning rate schedule (log scale)
- **Combined Curves**: `run16_5_combined_curves.png` - 2x2 panel with loss, PER, LR, and loss vs PER

## Usage

Generate all paper figures:
```bash
python scripts/generate_paper_figures_v2.py --output paper_figures_v2
```

Generate specific figures:
```bash
python scripts/generate_paper_figures_v2.py --figures 1 2 3 4 5 7 --output paper_figures_v2
```

Generate run16_5 training curves:
```bash
python scripts/generate_run16_5_curves.py --output paper_figures_v2
```

## Key Fixes

1. **Augmentation Figure**: Now correctly uses test PER (0.2046) instead of competition PER (2.799) for speechBaseline4
   - The `get_test_per()` method now explicitly skips competition set evaluations
   - Prefers test split with greedy decoding

2. **Baseline Table**: Uses test PER as proxy for "Best Valid PER" when metrics.jsonl is not available

3. **Multi-panel Layout**: Figure 2 now has 3 compact panels in one figure

## Data Sources

- `data/checkpoints/<run_name>/metrics.jsonl` - Training metrics
- `data/checkpoints/<run_name>/train.log` - Training configuration
- `data/checkpoints/<run_name>/eval*.json` - Evaluation results (prefers test split, skips competition)

## Notes

- **speechBaseline4**: Uses known baseline configuration and test PER (no metrics.jsonl)
- **Competition Set**: All eval results from competition set are explicitly excluded
- **Test PER**: Preferred over validation PER when available for consistency
- **Greedy Decoding**: Preferred when multiple decoding methods are available

