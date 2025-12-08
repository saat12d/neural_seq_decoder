#!/usr/bin/env python3

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RunDataLoader:
    """Loads data from checkpoint directories."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = self.checkpoint_dir.name
        
    def load_metrics_jsonl(self) -> List[Dict[str, Any]]:
        """Load metrics from metrics.jsonl file."""
        metrics_file = self.checkpoint_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []
        
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return metrics
    
    def load_args(self) -> Optional[Dict[str, Any]]:
        """Load training arguments from args pickle file."""
        args_file = self.checkpoint_dir / "args"
        if not args_file.exists():
            return None
        
        try:
            with open(args_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load args from {self.checkpoint_dir}: {e}")
        return None
    
    def load_eval_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all eval_*.json files."""
        eval_results = {}
        for eval_file in self.checkpoint_dir.glob("eval*.json"):
            try:
                with open(eval_file, 'r') as f:
                    eval_results[eval_file.stem] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {eval_file}: {e}")
        return eval_results
    
    def get_best_per(self, greedy_only: bool = True) -> Tuple[Optional[float], Optional[int]]:
        """Get best PER and step from metrics.jsonl."""
        metrics = self.load_metrics_jsonl()
        if not metrics:
            return None, None
        
        best_per = float('inf')
        best_step = None
        
        for entry in metrics:
            if 'step' not in entry:
                continue
            
            # Get PER (handle both 'per' and 'cer' fields)
            per = None
            if 'per' in entry:
                per = entry['per']
            elif 'cer' in entry:
                per = entry['cer']
            
            if per is not None and per < best_per:
                best_per = per
                best_step = entry['step']
        
        return best_per if best_per != float('inf') else None, best_step
    
    def get_config_from_log(self) -> Dict[str, Any]:
        """Extract config from train.log file."""
        log_file = self.checkpoint_dir / "train.log"
        if not log_file.exists():
            return {}
        
        config = {}
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract scheduler (check in order of specificity)
            if "Using constant LR" in content or "constant LR (no warmup, no decay)" in content:
                config['scheduler'] = "Constant"
            elif "OneCycleLR" in content:
                config['scheduler'] = "OneCycleLR"
            elif "Warmup→Cosine" in content or ("Warmup steps:" in content and "Cosine steps:" in content):
                config['scheduler'] = "WarmupCosine"
            elif "Warmup steps:" in content:
                config['scheduler'] = "WarmupCosine"
            else:
                config['scheduler'] = "Unknown"
            
            # Extract LR
            match = re.search(r'Peak LR: ([\d.e-]+)', content)
            if match:
                config['peak_lr'] = match.group(1)
            
            match = re.search(r'End LR: ([\d.e-]+)', content)
            if match:
                config['end_lr'] = match.group(1)
            
            # Extract precision
            if "BF16" in content or "bfloat16" in content:
                config['precision'] = "BF16"
            elif "FP16" in content or "float16" in content:
                config['precision'] = "FP16"
            elif "FP32" in content or "full precision" in content:
                config['precision'] = "FP32"
            
            # Extract gradient clipping
            match = re.search(r'Gradient clipping: max_norm=([\d.]+)', content)
            if match:
                config['grad_clip'] = float(match.group(1))
            elif "Gradient clipping" in content and "DISABLED" in content:
                config['grad_clip'] = None
            else:
                config['grad_clip'] = "Unknown"
            
            # Extract EMA
            match = re.search(r'EMA enabled: decay=([\d.]+)', content)
            if match:
                config['ema_decay'] = float(match.group(1))
            else:
                config['ema_decay'] = None
            
            # Extract input dropout
            match = re.search(r'Input dropout: ([\d.]+)', content)
            if match:
                config['input_dropout'] = float(match.group(1))
            else:
                config['input_dropout'] = 0.0
            
            # Extract augmentation
            match = re.search(r'White noise SD: ([\d.]+)', content)
            if match:
                config['noise_sd'] = float(match.group(1))
            else:
                config['noise_sd'] = None
            
            match = re.search(r'Constant offset SD: ([\d.]+)', content)
            if match:
                config['offset_sd'] = float(match.group(1))
            else:
                config['offset_sd'] = None
            
            # Extract warmup/cosine steps
            match = re.search(r'Warmup steps: (\d+)', content)
            if match:
                config['warmup_steps'] = int(match.group(1))
            
            match = re.search(r'Cosine steps: (\d+)', content)
            if match:
                config['cosine_steps'] = int(match.group(1))
        
        return config


def figure1_baseline_table(output_dir: Path):
    """Figure 1: Baseline vs clean reproduction table."""
    print("Generating Figure 1: Baseline vs clean reproduction table...")
    
    runs = [
        ("speechBaseline4", {"peak_lr": "0.02", "end_lr": "0.02", "scheduler": "Constant",
                            "precision": "FP32", "grad_clip": "None", "noise_sd": "0.8", "offset_sd": "0.2"}),
        ("run8_recovery", {}),
        ("run10_recovery", {}),
    ]
    
    data = []
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, defaults in runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            print(f"  Warning: {run_name} not found")
            continue
        
        loader = RunDataLoader(str(run_dir))
        config = loader.get_config_from_log()
        best_per, best_step = loader.get_best_per()
        eval_results = loader.load_eval_results()
        
        # For speechBaseline4, use known baseline config and get PER from eval
        if run_name == "speechBaseline4":
            # Use known baseline configuration
            config = {
                'scheduler': 'Constant',
                'peak_lr': '0.02',
                'end_lr': '0.02',
                'precision': 'FP32',
                'grad_clip': None,
                'noise_sd': 0.8,
                'offset_sd': 0.2,
            }
            # Get PER from eval (no metrics.jsonl for baseline)
            for eval_name, eval_data in eval_results.items():
                if 'avg_PER' in eval_data and 'test' in eval_name.lower():
                    if eval_data.get('decoding_method') == 'greedy':
                        best_per = eval_data['avg_PER']
                        break
                    elif best_per is None:
                        best_per = eval_data['avg_PER']
        
        # Get test PER from eval files (prefer greedy)
        test_per = None
        test_per_greedy = None
        for eval_name, eval_data in eval_results.items():
            if 'avg_PER' in eval_data and 'test' in eval_name.lower():
                if eval_data.get('decoding_method') == 'greedy':
                    test_per_greedy = eval_data['avg_PER']
                elif test_per is None:
                    test_per = eval_data['avg_PER']
        
        test_per = test_per_greedy if test_per_greedy is not None else test_per
        
        # Fill in defaults if not in config
        for key, value in defaults.items():
            if key not in config or config[key] is None:
                config[key] = value
        
        # Format augmentation string
        noise_val = config.get('noise_sd')
        offset_val = config.get('offset_sd')
        if noise_val is None:
            noise_val = defaults.get('noise_sd', '?')
        if offset_val is None:
            offset_val = defaults.get('offset_sd', '?')
        aug_str = f"noise={noise_val}, offset={offset_val}"
        
        row = {
            'Run': run_name,
            'Scheduler': config.get('scheduler', 'Unknown'),
            'Peak LR': config.get('peak_lr', 'Unknown'),
            'End LR': config.get('end_lr', 'Unknown'),
            'Precision': config.get('precision', 'Unknown'),
            'Grad Clip': str(config.get('grad_clip', 'Unknown')) if config.get('grad_clip') is not None else 'None',
            'Augmentation': aug_str,
            'Best Valid PER': f"{best_per:.4f}" if best_per else "N/A",
            'Step of Best': str(best_step) if best_step else "N/A",
            'Test PER': f"{test_per:.4f}" if test_per else "N/A",
        }
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_file = output_dir / "figure1_baseline_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    # Also create LaTeX table
    latex_file = output_dir / "figure1_baseline_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def figure2_per_vs_step(output_dir: Path):
    """Figure 2: PER vs Step line plot with LR schedules."""
    print("Generating Figure 2: PER vs Step line plot...")
    
    runs = [
        ("gru_ctc_reg5", "Constant 1e-3", "dashed"),
        ("gru_ctc_reg6", "Constant 1.5e-3", "dotted"),
        ("run15_warmup_cosine_safe", "Warmup→Cosine (1.5e-3)", "solid"),
        ("run16_greedy_ema_warmcos", "Warmup→Cosine+EMA (1.6e-3)", "solid"),
        ("run16_3", "Warmup→Cosine+EMA (1.6e-3, end=3e-6)", "solid"),
        ("run16_4", "Warmup→Cosine+EMA (1.4e-3, drop=0.05)", "solid"),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    checkpoints_dir = Path("data/checkpoints")
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    legend_data = []
    
    for idx, (run_name, label, linestyle) in enumerate(runs):
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            print(f"  Warning: {run_name} not found")
            continue
        
        loader = RunDataLoader(str(run_dir))
        metrics = loader.load_metrics_jsonl()
        config = loader.get_config_from_log()
        
        if not metrics:
            print(f"  Warning: No metrics for {run_name}")
            continue
        
        steps = []
        pers = []
        
        for entry in metrics:
            if 'step' not in entry:
                continue
            
            per = None
            if 'per' in entry:
                per = entry['per']
            elif 'cer' in entry:
                per = entry['cer']
            
            if per is not None:
                steps.append(entry['step'])
                pers.append(per)
        
        if steps:
            # Optional EMA smoothing (window=5)
            if len(pers) > 5:
                pers_smooth = pd.Series(pers).ewm(span=5, adjust=False).mean().values
            else:
                pers_smooth = pers
            
            ax.plot(steps, pers_smooth, label=label, color=colors[idx], 
                   linestyle=linestyle, linewidth=1.5, alpha=0.8)
            
            # Mark best PER point
            best_idx = np.argmin(pers)
            best_step = steps[best_idx]
            best_per = pers[best_idx]
            ax.plot(best_step, best_per, 'o', color=colors[idx], markersize=6)
            ax.annotate(f'{best_per:.3f}\n@{best_step}', 
                       xy=(best_step, best_per), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, alpha=0.7)
            
            # Store legend data
            legend_data.append({
                'run': run_name,
                'peak_lr': config.get('peak_lr', '?'),
                'end_lr': config.get('end_lr', '?'),
                'warmup': config.get('warmup_steps', '?'),
                'cosine': config.get('cosine_steps', '?'),
                'ema': config.get('ema_decay', 'None'),
            })
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation PER (Greedy)', fontsize=12)
    ax.set_title('PER vs Step: LR Schedule Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add inset table with config details
    table_data = []
    for ld in legend_data:
        table_data.append([
            ld['peak_lr'], ld['end_lr'], str(ld['warmup']), 
            str(ld['cosine']), str(ld['ema'])
        ])
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Peak LR', 'End LR', 'Warmup', 'Cosine', 'EMA'],
                        cellLoc='center',
                        loc='upper right',
                        bbox=[0.02, 0.5, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(0.8, 1.2)
    
    plt.tight_layout()
    output_file = output_dir / "figure2_per_vs_step.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def figure3_regularization_table(output_dir: Path):
    """Figure 3: Regularization ablation table."""
    print("Generating Figure 3: Regularization ablation table...")
    
    runs = [
        ("speechBaseline4", {"clip": "None", "ema": "None", "input_dropout": 0.0}),
        ("run15_warmup_cosine_safe", {"clip": "1.0", "ema": "None", "input_dropout": 0.0}),
        ("run16_greedy_ema_warmcos", {"clip": "1.0", "ema": "0.999", "input_dropout": 0.0}),
        ("run16_4", {"clip": "1.0", "ema": "0.9995", "input_dropout": 0.05}),
    ]
    
    data = []
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, defaults in runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            print(f"  Warning: {run_name} not found")
            continue
        
        loader = RunDataLoader(str(run_dir))
        config = loader.get_config_from_log()
        best_per, best_step = loader.get_best_per()
        
        # Fill in defaults if not in config
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        row = {
            'Run': run_name,
            'Clip': str(config.get('grad_clip', defaults.get('clip', 'Unknown'))),
            'EMA Decay': str(config.get('ema_decay', defaults.get('ema', 'None'))),
            'Input Dropout': str(config.get('input_dropout', defaults.get('input_dropout', 0.0))),
            'Best Valid PER': f"{best_per:.4f}" if best_per else "N/A",
            'Step of Best': str(best_step) if best_step else "N/A",
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_file = output_dir / "figure3_regularization_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "figure3_regularization_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def figure4_augmentation_bars(output_dir: Path):
    """Figure 4: Augmentation bar chart."""
    print("Generating Figure 4: Augmentation bar chart...")
    
    runs = [
        ("speechBaseline4", "Heavy noise\n(noise=0.8, offset=0.2)"),
        ("run11_plateau_break", "Reduced noise\n(reduced)"),
        ("run15_warmup_cosine_safe", "Light SpecAugment\n+ gentle noise"),
    ]
    
    data = []
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, label in runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            print(f"  Warning: {run_name} not found")
            continue
        
        loader = RunDataLoader(str(run_dir))
        best_per, _ = loader.get_best_per()
        config = loader.get_config_from_log()
        
        if best_per is None:
            # Try to get from eval
            eval_results = loader.load_eval_results()
            for eval_name, eval_data in eval_results.items():
                if 'avg_PER' in eval_data:
                    best_per = eval_data['avg_PER']
                    break
        
        data.append({
            'Run': label,
            'Noise SD': config.get('noise_sd', '?'),
            'Offset SD': config.get('offset_sd', '?'),
            'Best Valid PER': best_per if best_per else None,
        })
    
    # Filter out None values
    data = [d for d in data if d['Best Valid PER'] is not None]
    
    if not data:
        print("  Warning: No valid data for augmentation chart")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    runs_labels = [d['Run'] for d in data]
    pers = [d['Best Valid PER'] for d in data]
    
    bars = ax.bar(range(len(runs_labels)), pers, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xticks(range(len(runs_labels)))
    ax.set_xticklabels(runs_labels, rotation=45, ha='right')
    ax.set_ylabel('Best Valid PER (Greedy)', fontsize=12)
    ax.set_title('Augmentation Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, per in zip(bars, pers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{per:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "figure4_augmentation_bars.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def figure5_decoding_cost(output_dir: Path):
    """Figure 5: Decoding cost vs PER (optional)."""
    print("Generating Figure 5: Decoding cost vs PER...")
    print("  Note: This requires running eval.py with different decoders")
    print("  Skipping for now - run eval.py manually and provide results")
    
    # Placeholder - would need to run eval.py with different settings
    # and parse the results
    pass


def figure6_final_recipe_table(output_dir: Path):
    """Figure 6: Final recipe summary table."""
    print("Generating Figure 6: Final recipe summary table...")
    
    run_name = "run16_5"
    checkpoints_dir = Path("data/checkpoints")
    run_dir = checkpoints_dir / run_name
    
    if not run_dir.exists():
        print(f"  Warning: {run_name} not found")
        return
    
    loader = RunDataLoader(str(run_dir))
    config = loader.get_config_from_log()
    best_per, best_step = loader.get_best_per()
    eval_results = loader.load_eval_results()
    
    # Get test PERs for different decoders
    test_per_greedy = None
    test_per_beam10 = None
    test_per_beam20 = None
    
    for eval_name, eval_data in eval_results.items():
        if 'avg_PER' in eval_data:
            if 'beam' in eval_name.lower():
                if 'beam_size' in eval_data:
                    if eval_data['beam_size'] == 10:
                        test_per_beam10 = eval_data['avg_PER']
                    elif eval_data['beam_size'] == 20:
                        test_per_beam20 = eval_data['avg_PER']
            else:
                test_per_greedy = eval_data['avg_PER']
    
    row = {
        'Run': run_name,
        'Peak LR': config.get('peak_lr', '0.001'),
        'End LR': config.get('end_lr', '1e-7'),
        'Warmup': str(config.get('warmup_steps', 1500)),
        'Cosine': str(config.get('cosine_steps', 18500)),
        'Total Batches': '20000',
        'EMA': str(config.get('ema_decay', '0.9995')),
        'Input Dropout': str(config.get('input_dropout', '0.05')),
        'Batch (accum=2)': '32',
        'Clip': str(config.get('grad_clip', '1.0')),
        'Precision': config.get('precision', 'BF16'),
        'Best Valid PER': f"{best_per:.4f}" if best_per else "N/A",
        'Test PER (Greedy)': f"{test_per_greedy:.4f}" if test_per_greedy else "N/A",
        'Test PER (Beam k=10)': f"{test_per_beam10:.4f}" if test_per_beam10 else "N/A",
        'Test PER (Beam k=20)': f"{test_per_beam20:.4f}" if test_per_beam20 else "N/A",
    }
    
    df = pd.DataFrame([row])
    output_file = output_dir / "figure6_final_recipe_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "figure6_final_recipe_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='paper_figures', 
                       help='Output directory for figures')
    parser.add_argument('--figures', type=str, nargs='+', 
                       choices=['1', '2', '3', '4', '5', '6', 'all'],
                       default=['all'],
                       help='Which figures to generate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating figures in {output_dir}/")
    print("=" * 60)
    
    if 'all' in args.figures or '1' in args.figures:
        figure1_baseline_table(output_dir)
        print()
    
    if 'all' in args.figures or '2' in args.figures:
        figure2_per_vs_step(output_dir)
        print()
    
    if 'all' in args.figures or '3' in args.figures:
        figure3_regularization_table(output_dir)
        print()
    
    if 'all' in args.figures or '4' in args.figures:
        figure4_augmentation_bars(output_dir)
        print()
    
    if 'all' in args.figures or '5' in args.figures:
        figure5_decoding_cost(output_dir)
        print()
    
    if 'all' in args.figures or '6' in args.figures:
        figure6_final_recipe_table(output_dir)
        print()
    
    print("=" * 60)
    print("All figures generated!")


if __name__ == '__main__':
    main()

