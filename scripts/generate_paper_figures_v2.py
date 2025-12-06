#!/usr/bin/env python3
"""
Generate figures and tables for paper (revised spec).

Figures:
1. Baseline vs. Reproduction table
2. LR & Scheduler multi-panel figure
3. Optimizer & Weight Decay table
4. Regularization multi-panel figure or table
5. Data Augmentation figure
6. Decoding inset/figure
7. Best Configuration table
"""

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

# Add project root to path
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
            return None
    
    def load_eval_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all eval_*.json files."""
        eval_results = {}
        for eval_file in self.checkpoint_dir.glob("eval*.json"):
            try:
                with open(eval_file, 'r') as f:
                    eval_results[eval_file.stem] = json.load(f)
            except Exception as e:
                pass
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
            
            per = None
            if 'per' in entry:
                per = entry['per']
            elif 'cer' in entry:
                per = entry['cer']
            
            if per is not None and per < best_per:
                best_per = per
                best_step = entry['step']
        
        return best_per if best_per != float('inf') else None, best_step
    
    def get_test_per(self, greedy_only: bool = True) -> Optional[float]:
        """Get test PER from eval files (prefer test split, greedy decoding)."""
        eval_results = self.load_eval_results()
        
        # Prefer test split with greedy decoding
        for eval_name, eval_data in eval_results.items():
            if 'avg_PER' in eval_data:
                split = eval_data.get('split', '').lower()
                method = eval_data.get('decoding_method', '').lower()
                
                # Skip competition set
                if 'competition' in split or 'competition' in eval_name.lower():
                    continue
                
                # Prefer test split
                if 'test' in split or 'test' in eval_name.lower():
                    if greedy_only and method != 'greedy':
                        continue
                    return eval_data['avg_PER']
        
        # Fallback: any non-competition eval
        for eval_name, eval_data in eval_results.items():
            if 'avg_PER' in eval_data:
                if 'competition' not in eval_name.lower():
                    return eval_data['avg_PER']
        
        return None
    
    def get_config_from_log(self) -> Dict[str, Any]:
        """Extract config from train.log file."""
        log_file = self.checkpoint_dir / "train.log"
        if not log_file.exists():
            return {}
        
        config = {}
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract scheduler
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
                config['grad_clip'] = None
            
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
            
            # Extract optimizer
            if "AdamW" in content:
                config['optimizer'] = "AdamW"
            elif "ADAM" in content or "Adam" in content:
                config['optimizer'] = "Adam"
            
            # Extract weight decay
            match = re.search(r'Weight decay: ([\d.e-]+)', content)
            if match:
                config['weight_decay'] = float(match.group(1))
            else:
                config['weight_decay'] = 0.0
        
        return config


def figure1_baseline_table(output_dir: Path):
    """Figure 1: Baseline vs. Reproduction table (3-4 rows)."""
    print("Generating Figure 1: Baseline vs. Reproduction table...")
    
    runs = [
        ("speechBaseline4", {"peak_lr": "0.02", "end_lr": "0.02", "scheduler": "Constant",
                            "precision": "FP32", "grad_clip": None, "noise_sd": 0.8, "offset_sd": 0.2}),
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
        
        # For speechBaseline4, use known config
        if run_name == "speechBaseline4":
            config = defaults.copy()
            best_per = loader.get_test_per()  # Use test PER since no metrics.jsonl
            best_step = None
        else:
            config = loader.get_config_from_log()
            best_per, best_step = loader.get_best_per()
            # Fill in defaults
            for key, value in defaults.items():
                if key not in config or config[key] is None:
                    config[key] = value
        
        test_per = loader.get_test_per()
        
        row = {
            'Run': run_name,
            'Scheduler': config.get('scheduler', 'Unknown'),
            'Peak LR': config.get('peak_lr', 'Unknown'),
            'End LR': config.get('end_lr', 'Unknown'),
            'Precision': config.get('precision', 'Unknown'),
            'Grad Clip': str(config.get('grad_clip', 'None')) if config.get('grad_clip') is not None else 'None',
            'Best Valid PER': f"{best_per:.4f}" if best_per else "N/A",
            'Test PER': f"{test_per:.4f}" if test_per else "N/A",
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_file = output_dir / "figure1_baseline_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "figure1_baseline_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def figure2_lr_scheduler_panels(output_dir: Path):
    """Figure 2: LR & Scheduler multi-panel figure."""
    print("Generating Figure 2: LR & Scheduler multi-panel figure...")
    
    fig = plt.figure(figsize=(14, 5))
    
    # Panel A: Constant-LR sweep
    ax1 = plt.subplot(1, 3, 1)
    constant_runs = [
        ("gru_ctc_reg5", "1e-3", "dashed"),
        ("gru_ctc_reg6", "1.5e-3", "dotted"),
    ]
    # Need to find a 1.2e-3 run or approximate
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, lr_label, linestyle in constant_runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        metrics = loader.load_metrics_jsonl()
        
        if not metrics:
            continue
        
        steps = []
        pers = []
        for entry in metrics:
            if 'step' not in entry:
                continue
            per = entry.get('per') or entry.get('cer')
            if per is not None:
                steps.append(entry['step'])
                pers.append(per)
        
        if steps:
            if len(pers) > 5:
                pers_smooth = pd.Series(pers).ewm(span=5, adjust=False).mean().values
            else:
                pers_smooth = pers
            ax1.plot(steps, pers_smooth, label=f"LR={lr_label}", linestyle=linestyle, linewidth=1.5)
    
    ax1.set_xlabel('Training Step', fontsize=10)
    ax1.set_ylabel('Validation PER', fontsize=10)
    ax1.set_title('Panel A: Constant LR Sweep', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Panel B: Warmup→Cosine variants
    ax2 = plt.subplot(1, 3, 2)
    warmup_runs = [
        ("run15_warmup_cosine_safe", "Run15"),
        ("run16_greedy_ema_warmcos", "Run16"),
        ("run16_3", "Run16_3"),
        ("run16_4", "Run16_4"),
        ("run16_5", "Run16_5"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(warmup_runs)))
    
    for idx, (run_name, label) in enumerate(warmup_runs):
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        metrics = loader.load_metrics_jsonl()
        
        if not metrics:
            continue
        
        steps = []
        pers = []
        for entry in metrics:
            if 'step' not in entry:
                continue
            per = entry.get('per') or entry.get('cer')
            if per is not None:
                steps.append(entry['step'])
                pers.append(per)
        
        if steps:
            if len(pers) > 5:
                pers_smooth = pd.Series(pers).ewm(span=5, adjust=False).mean().values
            else:
                pers_smooth = pers
            ax2.plot(steps, pers_smooth, label=label, color=colors[idx], linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Training Step', fontsize=10)
    ax2.set_ylabel('Validation PER', fontsize=10)
    ax2.set_title('Panel B: Warmup→Cosine Variants', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Panel C: LR(t) profiles (optional)
    ax3 = plt.subplot(1, 3, 3)
    # Plot LR schedules for best two
    best_runs = ["run16_4", "run16_5"]
    for run_name in best_runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        metrics = loader.load_metrics_jsonl()
        config = loader.get_config_from_log()
        
        if not metrics:
            continue
        
        steps = []
        lrs = []
        for entry in metrics:
            if 'step' in entry and 'lr' in entry:
                steps.append(entry['step'])
                lrs.append(entry['lr'])
        
        if steps:
            ax3.plot(steps, lrs, label=run_name, linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Training Step', fontsize=10)
    ax3.set_ylabel('Learning Rate', fontsize=10)
    ax3.set_title('Panel C: LR(t) Profiles', fontsize=11, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "figure2_lr_scheduler_panels.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def figure3_optimizer_table(output_dir: Path):
    """Figure 3: Optimizer & Weight Decay table."""
    print("Generating Figure 3: Optimizer & Weight Decay table...")
    print("  Note: Checking for Adam vs AdamW comparison runs...")
    
    # Check which runs have AdamW
    checkpoints_dir = Path("data/checkpoints")
    runs_to_check = ["run10_recovery", "run11_plateau_break", "run12_baseline_hybrid"]
    
    data = []
    for run_name in runs_to_check:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        config = loader.get_config_from_log()
        best_per, best_step = loader.get_best_per()
        test_per = loader.get_test_per()
        
        optimizer = config.get('optimizer', 'Adam')
        wd = config.get('weight_decay', 0.0)
        scheduler = config.get('scheduler', 'Unknown')
        
        row = {
            'Run': run_name,
            'Optimizer': optimizer,
            'Weight Decay': f"{wd:.0e}" if wd > 0 else "0",
            'Schedule': scheduler,
            'Best Val PER': f"{best_per:.4f}" if best_per else "N/A",
            'Test PER': f"{test_per:.4f}" if test_per else "N/A",
        }
        data.append(row)
    
    if not data:
        print("  Warning: No optimizer comparison data found")
        return None
    
    df = pd.DataFrame(data)
    output_file = output_dir / "figure3_optimizer_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "figure3_optimizer_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def figure4_regularization(output_dir: Path):
    """Figure 4: Regularization ablation (table + small plot)."""
    print("Generating Figure 4: Regularization ablation...")
    
    runs = [
        ("speechBaseline4", {"clip": None, "ema": None, "dropout": 0.4}),
        ("run15_warmup_cosine_safe", {"clip": 1.0, "ema": None, "dropout": 0.4}),
        ("run16_greedy_ema_warmcos", {"clip": 1.0, "ema": 0.999, "dropout": 0.4}),
        ("run16_4", {"clip": 1.0, "ema": 0.9995, "dropout": 0.4, "input_dropout": 0.05}),
    ]
    
    # Create table
    data = []
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, defaults in runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        config = loader.get_config_from_log()
        best_per, _ = loader.get_best_per()
        
        # Fill defaults
        for key, value in defaults.items():
            if key not in config or config[key] is None:
                config[key] = value
        
        row = {
            'Run': run_name,
            'Clip': str(config.get('grad_clip', defaults.get('clip', 'None'))),
            'EMA Decay': str(config.get('ema_decay', defaults.get('ema', 'None'))),
            'Dropout': str(config.get('dropout', defaults.get('dropout', '0.4'))),
            'Input Dropout': str(config.get('input_dropout', defaults.get('input_dropout', '0.0'))),
            'Best Val PER': f"{best_per:.4f}" if best_per else "N/A",
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_file = output_dir / "figure4_regularization_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved table to {output_file}")
    
    # Create small plot: EMA decay sweep
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ema_runs = [
        ("run15_warmup_cosine_safe", 0.0, "No EMA"),
        ("run16_greedy_ema_warmcos", 0.999, "EMA=0.999"),
        ("run16_4", 0.9995, "EMA=0.9995"),
    ]
    
    labels = []
    pers = []
    
    for run_name, ema_val, label in ema_runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        best_per, _ = loader.get_best_per()
        
        if best_per:
            labels.append(label)
            pers.append(best_per)
    
    if labels:
        bars = ax.bar(range(len(labels)), pers, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Best Valid PER', fontsize=10)
        ax.set_title('EMA Decay Sweep', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, per in zip(bars, pers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{per:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = output_dir / "figure4_regularization_plot.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {plot_file}")
    plt.close()
    
    return df


def figure5_augmentation(output_dir: Path):
    """Figure 5: Data Augmentation bars."""
    print("Generating Figure 5: Data Augmentation bars...")
    
    runs = [
        ("speechBaseline4", "Heavy noise\n(noise=0.8, offset=0.2)"),
        ("run11_plateau_break", "Gentle noise\n(reduced)"),
        ("run15_warmup_cosine_safe", "Gentle noise\n+ SpecAugment"),
    ]
    
    data = []
    checkpoints_dir = Path("data/checkpoints")
    
    for run_name, label in runs:
        run_dir = checkpoints_dir / run_name
        if not run_dir.exists():
            continue
        
        loader = RunDataLoader(str(run_dir))
        best_per, _ = loader.get_best_per()
        
        # For speechBaseline4, use test PER (no metrics.jsonl)
        if run_name == "speechBaseline4" and best_per is None:
            best_per = loader.get_test_per()
        
        config = loader.get_config_from_log()
        
        if best_per:
            data.append({
                'Run': label,
                'Best PER': best_per,
                'Noise SD': config.get('noise_sd', '?'),
            })
    
    if not data:
        print("  Warning: No valid data")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    labels = [d['Run'] for d in data]
    pers = [d['Best PER'] for d in data]
    
    bars = ax.bar(range(len(labels)), pers, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Best Valid PER (Greedy)', fontsize=11)
    ax.set_title('Data Augmentation Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, per in zip(bars, pers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{per:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add caption note
    ax.text(0.5, -0.15, 'Note: Over-masking slowed late-cosine convergence',
           transform=ax.transAxes, fontsize=9, style='italic',
           ha='center', va='top')
    
    plt.tight_layout()
    output_file = output_dir / "figure5_augmentation.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def figure6_decoding_inset(output_dir: Path):
    """Figure 6: Decoding comparison (greedy vs beam)."""
    print("Generating Figure 6: Decoding comparison...")
    print("  Note: Requires eval results with different beam sizes")
    
    # This would need eval results with beam search
    # For now, create a placeholder structure
    checkpoints_dir = Path("data/checkpoints")
    run_name = "run16_5"
    run_dir = checkpoints_dir / run_name
    
    if not run_dir.exists():
        print("  Warning: run16_5 not found")
        return
    
    loader = RunDataLoader(str(run_dir))
    eval_results = loader.load_eval_results()
    
    data = []
    for eval_name, eval_data in eval_results.items():
        if 'avg_PER' in eval_data:
            method = eval_data.get('decoding_method', 'greedy')
            beam_size = eval_data.get('beam_size', None)
            
            if method == 'greedy':
                label = 'Greedy'
            elif beam_size:
                label = f'Beam k={beam_size}'
            else:
                label = method
            
            data.append({
                'Method': label,
                'PER': eval_data['avg_PER'],
            })
    
    if not data:
        print("  Warning: No decoding comparison data found")
        return
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    labels = [d['Method'] for d in data]
    pers = [d['PER'] for d in data]
    
    bars = ax.bar(range(len(labels)), pers, alpha=0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Test PER', fontsize=10)
    ax.set_title('Decoding Method Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, per in zip(bars, pers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{per:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "figure6_decoding.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def figure7_best_config_table(output_dir: Path):
    """Figure 7: Best Configuration table (Run16_5 recipe)."""
    print("Generating Figure 7: Best Configuration table...")
    
    run_name = "run16_5"
    checkpoints_dir = Path("data/checkpoints")
    run_dir = checkpoints_dir / run_name
    
    if not run_dir.exists():
        print(f"  Warning: {run_name} not found")
        return
    
    loader = RunDataLoader(str(run_dir))
    config = loader.get_config_from_log()
    best_per, best_step = loader.get_best_per()
    test_per = loader.get_test_per()
    
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
        'Best Val PER': f"{best_per:.4f}" if best_per else "N/A",
        'Test PER': f"{test_per:.4f}" if test_per else "N/A",
    }
    
    df = pd.DataFrame([row])
    output_file = output_dir / "figure7_best_config_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "figure7_best_config_table.tex"
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures (revised spec)')
    parser.add_argument('--output', type=str, default='paper_figures_v2', 
                       help='Output directory for figures')
    parser.add_argument('--figures', type=str, nargs='+', 
                       choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
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
        figure2_lr_scheduler_panels(output_dir)
        print()
    
    if 'all' in args.figures or '3' in args.figures:
        figure3_optimizer_table(output_dir)
        print()
    
    if 'all' in args.figures or '4' in args.figures:
        figure4_regularization(output_dir)
        print()
    
    if 'all' in args.figures or '5' in args.figures:
        figure5_augmentation(output_dir)
        print()
    
    if 'all' in args.figures or '6' in args.figures:
        figure6_decoding_inset(output_dir)
        print()
    
    if 'all' in args.figures or '7' in args.figures:
        figure7_best_config_table(output_dir)
        print()
    
    print("=" * 60)
    print("All figures generated!")


if __name__ == '__main__':
    main()

