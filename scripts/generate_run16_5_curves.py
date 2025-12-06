#!/usr/bin/env python3
"""
Generate loss and training curves for run16_5 (best run).

Uses the existing plotting infrastructure to create:
1. CTC Loss over training steps
2. PER over training steps
3. Learning rate schedule
4. Train vs Test comparison
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.plot_training_results import CheckpointDataLoader, TrainingPlotter


def load_run16_5_data():
    """Load all data for run16_5."""
    checkpoint_dir = Path("data/checkpoints/run16_5")
    loader = CheckpointDataLoader(str(checkpoint_dir))
    return loader.get_all_data()


def plot_loss_curve(data, output_dir: Path):
    """Plot CTC loss over training steps."""
    print("Generating loss curve...")
    
    metrics = data.get('metrics', [])
    if not metrics:
        print("  Warning: No metrics found")
        return
    
    steps = []
    losses = []
    train_losses = []
    
    for entry in metrics:
        if 'step' not in entry:
            continue
        
        steps.append(entry['step'])
        
        if 'ctc_loss' in entry:
            losses.append(entry['ctc_loss'])
        else:
            losses.append(None)
        
        if 'train_loss_avg' in entry:
            train_losses.append(entry['train_loss_avg'])
        else:
            train_losses.append(None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot test/validation loss
    valid_steps = [s for s, l in zip(steps, losses) if l is not None]
    valid_losses = [l for l in losses if l is not None]
    
    if valid_steps:
        # Optional smoothing
        if len(valid_losses) > 5:
            valid_losses_smooth = pd.Series(valid_losses).ewm(span=5, adjust=False).mean().values
        else:
            valid_losses_smooth = valid_losses
        
        ax.plot(valid_steps, valid_losses_smooth, label='Validation Loss', 
               color='#1f77b4', linewidth=2, alpha=0.8)
        ax.plot(valid_steps, valid_losses, color='#1f77b4', linewidth=1, 
               alpha=0.3, linestyle='--')  # Raw data in background
    
    # Plot train loss if available
    train_valid_steps = [s for s, l in zip(steps, train_losses) if l is not None]
    train_valid_losses = [l for l in train_losses if l is not None]
    
    if train_valid_steps:
        if len(train_valid_losses) > 5:
            train_valid_losses_smooth = pd.Series(train_valid_losses).ewm(span=5, adjust=False).mean().values
        else:
            train_valid_losses_smooth = train_valid_losses
        
        ax.plot(train_valid_steps, train_valid_losses_smooth, label='Train Loss (avg)', 
               color='#ff7f0e', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('CTC Loss', fontsize=12)
    ax.set_title('Training and Validation Loss - Run16_5', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "run16_5_loss_curve.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def plot_per_curve(data, output_dir: Path):
    """Plot PER over training steps."""
    print("Generating PER curve...")
    
    metrics = data.get('metrics', [])
    if not metrics:
        print("  Warning: No metrics found")
        return
    
    steps = []
    pers = []
    per_mas = []
    
    for entry in metrics:
        if 'step' not in entry:
            continue
        
        steps.append(entry['step'])
        
        if 'per' in entry:
            pers.append(entry['per'])
        elif 'cer' in entry:
            pers.append(entry['cer'])
        else:
            pers.append(None)
        
        if 'per_ma' in entry:
            per_mas.append(entry['per_ma'])
        else:
            per_mas.append(None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PER
    valid_steps = [s for s, p in zip(steps, pers) if p is not None]
    valid_pers = [p for p in pers if p is not None]
    
    if valid_steps:
        if len(valid_pers) > 5:
            valid_pers_smooth = pd.Series(valid_pers).ewm(span=5, adjust=False).mean().values
        else:
            valid_pers_smooth = valid_pers
        
        ax.plot(valid_steps, valid_pers_smooth, label='PER', 
               color='#2ca02c', linewidth=2, alpha=0.8)
        ax.plot(valid_steps, valid_pers, color='#2ca02c', linewidth=1, 
               alpha=0.3, linestyle='--')
    
    # Plot PER moving average if available
    ma_valid_steps = [s for s, p in zip(steps, per_mas) if p is not None]
    ma_valid_pers = [p for p in per_mas if p is not None]
    
    if ma_valid_steps:
        ax.plot(ma_valid_steps, ma_valid_pers, label='PER (Moving Avg)', 
               color='#9467bd', linewidth=1.5, alpha=0.7, linestyle=':')
    
    # Mark best PER
    if valid_pers:
        best_idx = np.argmin(valid_pers)
        best_step = valid_steps[best_idx]
        best_per = valid_pers[best_idx]
        ax.plot(best_step, best_per, 'o', color='red', markersize=8)
        ax.annotate(f'Best: {best_per:.4f}\n@ step {best_step}', 
                   xy=(best_step, best_per), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('PER (Phoneme Error Rate)', fontsize=12)
    ax.set_title('PER Over Training - Run16_5', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "run16_5_per_curve.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def plot_lr_schedule(data, output_dir: Path):
    """Plot learning rate schedule."""
    print("Generating LR schedule...")
    
    metrics = data.get('metrics', [])
    if not metrics:
        print("  Warning: No metrics found")
        return
    
    steps = []
    lrs = []
    
    for entry in metrics:
        if 'step' in entry and 'lr' in entry:
            steps.append(entry['step'])
            lrs.append(entry['lr'])
    
    if not steps:
        print("  Warning: No LR data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, lrs, color='#d62728', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule - Run16_5', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotations for key points
    if len(lrs) > 0:
        ax.text(0.02, 0.98, f'Start LR: {lrs[0]:.2e}\nEnd LR: {lrs[-1]:.2e}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "run16_5_lr_schedule.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def plot_combined_training_curves(data, output_dir: Path):
    """Plot combined training curves in one figure."""
    print("Generating combined training curves...")
    
    metrics = data.get('metrics', [])
    if not metrics:
        print("  Warning: No metrics found")
        return
    
    steps = []
    losses = []
    pers = []
    lrs = []
    train_losses = []
    
    for entry in metrics:
        if 'step' not in entry:
            continue
        
        steps.append(entry['step'])
        
        losses.append(entry.get('ctc_loss'))
        pers.append(entry.get('per') or entry.get('cer'))
        lrs.append(entry.get('lr'))
        train_losses.append(entry.get('train_loss_avg'))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Loss
    ax1 = axes[0, 0]
    valid_steps = [s for s, l in zip(steps, losses) if l is not None]
    valid_losses = [l for l in losses if l is not None]
    if valid_steps:
        if len(valid_losses) > 5:
            valid_losses_smooth = pd.Series(valid_losses).ewm(span=5, adjust=False).mean().values
        else:
            valid_losses_smooth = valid_losses
        ax1.plot(valid_steps, valid_losses_smooth, label='Validation', color='#1f77b4', linewidth=2)
    
    train_valid_steps = [s for s, l in zip(steps, train_losses) if l is not None]
    train_valid_losses = [l for l in train_losses if l is not None]
    if train_valid_steps:
        if len(train_valid_losses) > 5:
            train_valid_losses_smooth = pd.Series(train_valid_losses).ewm(span=5, adjust=False).mean().values
        else:
            train_valid_losses_smooth = train_valid_losses
        ax1.plot(train_valid_steps, train_valid_losses_smooth, label='Train', 
               color='#ff7f0e', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Training Step', fontsize=10)
    ax1.set_ylabel('CTC Loss', fontsize=10)
    ax1.set_title('Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Top right: PER
    ax2 = axes[0, 1]
    valid_steps = [s for s, p in zip(steps, pers) if p is not None]
    valid_pers = [p for p in pers if p is not None]
    if valid_steps:
        if len(valid_pers) > 5:
            valid_pers_smooth = pd.Series(valid_pers).ewm(span=5, adjust=False).mean().values
        else:
            valid_pers_smooth = valid_pers
        ax2.plot(valid_steps, valid_pers_smooth, color='#2ca02c', linewidth=2)
        
        # Mark best
        if valid_pers:
            best_idx = np.argmin(valid_pers)
            best_step = valid_steps[best_idx]
            best_per = valid_pers[best_idx]
            ax2.plot(best_step, best_per, 'ro', markersize=6)
            ax2.annotate(f'{best_per:.4f}', xy=(best_step, best_per),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Training Step', fontsize=10)
    ax2.set_ylabel('PER', fontsize=10)
    ax2.set_title('Phoneme Error Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Learning Rate
    ax3 = axes[1, 0]
    lr_steps = [s for s, lr in zip(steps, lrs) if lr is not None]
    lr_values = [lr for lr in lrs if lr is not None]
    if lr_steps:
        ax3.plot(lr_steps, lr_values, color='#d62728', linewidth=2)
    ax3.set_xlabel('Training Step', fontsize=10)
    ax3.set_ylabel('Learning Rate', fontsize=10)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Bottom right: Loss vs PER (dual axis)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    if valid_steps and valid_losses:
        ax4.plot(valid_steps, valid_losses_smooth, label='Loss', color='#1f77b4', linewidth=2)
        ax4.set_ylabel('Loss', fontsize=10, color='#1f77b4')
        ax4.tick_params(axis='y', labelcolor='#1f77b4')
    
    if valid_steps and valid_pers:
        ax4_twin.plot(valid_steps, valid_pers_smooth, label='PER', color='#2ca02c', linewidth=2)
        ax4_twin.set_ylabel('PER', fontsize=10, color='#2ca02c')
        ax4_twin.tick_params(axis='y', labelcolor='#2ca02c')
    
    ax4.set_xlabel('Training Step', fontsize=10)
    ax4.set_title('Loss vs PER', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "run16_5_combined_curves.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training curves for run16_5')
    parser.add_argument('--output', type=str, default='paper_figures_v2', 
                       help='Output directory')
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading run16_5 data...")
    data = load_run16_5_data()
    
    if not data.get('metrics'):
        print("Error: No metrics found for run16_5")
        return
    
    print(f"Found {len(data['metrics'])} metric entries")
    print("=" * 60)
    
    plot_loss_curve(data, output_dir)
    plot_per_curve(data, output_dir)
    plot_lr_schedule(data, output_dir)
    plot_combined_training_curves(data, output_dir)
    
    print("=" * 60)
    print("All curves generated!")


if __name__ == '__main__':
    import argparse
    main()

