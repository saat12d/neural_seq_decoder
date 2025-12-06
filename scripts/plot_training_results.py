#!/usr/bin/env python3
"""
Comprehensive plotting script for training checkpoint data.

This script analyzes all run folders in data/checkpoints and creates various graphs:
- CTC loss over time
- PER (Phoneme Error Rate) over time
- Train vs Test validation graphs
- Learning rate schedules
- Memory usage
- Gradient norms
- And more...

Supports multiple data formats:
- metrics.jsonl (step-by-step metrics)
- trainingStats (pickle file with per-epoch stats)
- eval_*.json (evaluation results)
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CheckpointDataLoader:
    """Loads and parses data from checkpoint directories."""
    
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
    
    def load_training_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Load trainingStats pickle file."""
        stats_file = self.checkpoint_dir / "trainingStats"
        if not stats_file.exists():
            return None
        
        try:
            with open(stats_file, 'rb') as f:
                data = pickle.load(f)
                # Ensure it's a dict with numpy arrays
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"Warning: Could not load trainingStats from {self.checkpoint_dir}: {e}")
        return None
    
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
        for eval_file in self.checkpoint_dir.glob("eval_*.json"):
            try:
                with open(eval_file, 'r') as f:
                    eval_results[eval_file.stem] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {eval_file}: {e}")
        return eval_results
    
    def get_all_data(self) -> Dict[str, Any]:
        """Load all available data from this checkpoint."""
        return {
            'run_name': self.run_name,
            'metrics': self.load_metrics_jsonl(),
            'training_stats': self.load_training_stats(),
            'args': self.load_args(),
            'eval_results': self.load_eval_results(),
        }


class TrainingPlotter:
    """Creates various plots from checkpoint data."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_ctc_loss(self, data_dicts: List[Dict[str, Any]], 
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None) -> Figure:
        """Plot CTC loss over training steps."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for data in data_dicts:
            metrics = data.get('metrics', [])
            if not metrics:
                continue
            
            run_name = data.get('run_name', 'unknown')
            steps = []
            ctc_losses = []
            
            for entry in metrics:
                if 'step' in entry and 'ctc_loss' in entry:
                    steps.append(entry['step'])
                    ctc_losses.append(entry['ctc_loss'])
            
            if steps:
                ax.plot(steps, ctc_losses, label=run_name, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('CTC Loss', fontsize=12)
        ax.set_title('CTC Loss Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per(self, data_dicts: List[Dict[str, Any]],
                figsize: Tuple[int, int] = (12, 6),
                save_path: Optional[str] = None) -> Figure:
        """Plot PER (Phoneme Error Rate) over training steps."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for data in data_dicts:
            metrics = data.get('metrics', [])
            if not metrics:
                continue
            
            run_name = data.get('run_name', 'unknown')
            steps = []
            pers = []
            per_mas = []
            
            for entry in metrics:
                if 'step' in entry:
                    steps.append(entry['step'])
                    # Handle both 'per' and 'cer' fields (cer was used in older runs)
                    if 'per' in entry:
                        pers.append(entry['per'])
                    elif 'cer' in entry:
                        pers.append(entry['cer'])
                    else:
                        continue
                    
                    if 'per_ma' in entry:
                        per_mas.append(entry['per_ma'])
            
            if steps and pers:
                ax.plot(steps, pers, label=f'{run_name} (PER)', alpha=0.7, linewidth=1.5)
                if per_mas and len(per_mas) == len(steps):
                    ax.plot(steps, per_mas, label=f'{run_name} (PER MA)', 
                           alpha=0.5, linewidth=1, linestyle='--')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('PER (Phoneme Error Rate)', fontsize=12)
        ax.set_title('PER Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_train_vs_test(self, data_dicts: List[Dict[str, Any]],
                          figsize: Tuple[int, int] = (14, 6),
                          save_path: Optional[str] = None) -> Figure:
        """Plot train vs test metrics comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        for data in data_dicts:
            run_name = data.get('run_name', 'unknown')
            metrics = data.get('metrics', [])
            training_stats = data.get('training_stats')
            
            # Plot from metrics.jsonl (test metrics during training)
            if metrics:
                steps = []
                test_losses = []
                test_pers = []
                
                for entry in metrics:
                    if 'step' in entry:
                        steps.append(entry['step'])
                        if 'ctc_loss' in entry:
                            test_losses.append(entry['ctc_loss'])
                        if 'per' in entry:
                            test_pers.append(entry['per'])
                        elif 'cer' in entry:
                            test_pers.append(entry['cer'])
                
                if steps and test_losses:
                    ax1.plot(steps, test_losses, label=f'{run_name} (Test)', 
                            alpha=0.7, linewidth=1.5)
                
                if steps and test_pers:
                    ax2.plot(steps, test_pers, label=f'{run_name} (Test)', 
                            alpha=0.7, linewidth=1.5)
            
            # Plot from trainingStats (per-epoch data)
            if training_stats:
                epochs = np.arange(len(training_stats.get('testLoss', [])))
                if 'testLoss' in training_stats:
                    ax1.plot(epochs, training_stats['testLoss'], 
                            label=f'{run_name} (Epochs)', 
                            alpha=0.7, linewidth=1.5, linestyle=':')
                if 'testCER' in training_stats:
                    ax2.plot(epochs, training_stats['testCER'], 
                            label=f'{run_name} (Epochs)', 
                            alpha=0.7, linewidth=1.5, linestyle=':')
            
            # Plot train loss if available
            if metrics:
                steps = []
                train_losses = []
                
                for entry in metrics:
                    if 'step' in entry and 'train_loss_avg' in entry:
                        steps.append(entry['step'])
                        train_losses.append(entry['train_loss_avg'])
                
                if steps and train_losses:
                    ax1.plot(steps, train_losses, label=f'{run_name} (Train)', 
                            alpha=0.7, linewidth=1.5, linestyle='--')
        
        ax1.set_xlabel('Training Step / Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Train vs Test Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        
        ax2.set_xlabel('Training Step / Epoch', fontsize=12)
        ax2.set_ylabel('PER', fontsize=12)
        ax2.set_title('Test PER Over Training', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_rate(self, data_dicts: List[Dict[str, Any]],
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> Figure:
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for data in data_dicts:
            metrics = data.get('metrics', [])
            if not metrics:
                continue
            
            run_name = data.get('run_name', 'unknown')
            steps = []
            lrs = []
            
            for entry in metrics:
                if 'step' in entry and 'lr' in entry:
                    steps.append(entry['step'])
                    lrs.append(entry['lr'])
            
            if steps:
                ax.plot(steps, lrs, label=run_name, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gradient_norms(self, data_dicts: List[Dict[str, Any]],
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None) -> Figure:
        """Plot gradient norms over training."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for data in data_dicts:
            metrics = data.get('metrics', [])
            if not metrics:
                continue
            
            run_name = data.get('run_name', 'unknown')
            steps = []
            grad_norms_avg = []
            grad_norms_max = []
            
            for entry in metrics:
                if 'step' in entry:
                    steps.append(entry['step'])
                    if 'grad_norm_avg' in entry:
                        grad_norms_avg.append(entry['grad_norm_avg'])
                    if 'grad_norm_max' in entry:
                        grad_norms_max.append(entry['grad_norm_max'])
            
            if steps and grad_norms_avg:
                ax.plot(steps, grad_norms_avg, label=f'{run_name} (Avg)', 
                       alpha=0.7, linewidth=1.5)
            if steps and grad_norms_max:
                ax.plot(steps, grad_norms_max, label=f'{run_name} (Max)', 
                       alpha=0.7, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norms Over Training', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_memory_usage(self, data_dicts: List[Dict[str, Any]],
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> Figure:
        """Plot memory usage over training."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for data in data_dicts:
            metrics = data.get('metrics', [])
            if not metrics:
                continue
            
            run_name = data.get('run_name', 'unknown')
            steps = []
            mem_alloc = []
            mem_reserved = []
            
            for entry in metrics:
                if 'step' in entry:
                    steps.append(entry['step'])
                    if 'memory_allocated_gb' in entry:
                        mem_alloc.append(entry['memory_allocated_gb'])
                    if 'memory_reserved_gb' in entry:
                        mem_reserved.append(entry['memory_reserved_gb'])
            
            if steps and mem_alloc:
                ax.plot(steps, mem_alloc, label=f'{run_name} (Allocated)', 
                       alpha=0.7, linewidth=1.5)
            if steps and mem_reserved:
                ax.plot(steps, mem_reserved, label=f'{run_name} (Reserved)', 
                       alpha=0.7, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Memory (GB)', fontsize=12)
        ax.set_title('GPU Memory Usage Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_eval_comparison(self, data_dicts: List[Dict[str, Any]],
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[str] = None) -> Figure:
        """Compare evaluation results across runs."""
        fig, ax = plt.subplots(figsize=figsize)
        
        run_names = []
        avg_pers = []
        
        for data in data_dicts:
            eval_results = data.get('eval_results', {})
            run_name = data.get('run_name', 'unknown')
            
            # Try to find avg_PER in eval results
            for eval_name, eval_data in eval_results.items():
                if 'avg_PER' in eval_data:
                    run_names.append(f"{run_name}\n({eval_name})")
                    avg_pers.append(eval_data['avg_PER'])
                    break
        
        if run_names:
            bars = ax.bar(range(len(run_names)), avg_pers, alpha=0.7)
            ax.set_xticks(range(len(run_names)))
            ax.set_xticklabels(run_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Average PER', fontsize=12)
            ax.set_title('Evaluation PER Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, per) in enumerate(zip(bars, avg_pers)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{per:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_all(self, data_dicts: List[Dict[str, Any]], 
                prefix: str = "training") -> None:
        """Generate all plots."""
        print(f"Generating plots for {len(data_dicts)} run(s)...")
        
        # CTC Loss
        self.plot_ctc_loss(data_dicts, 
                          save_path=self.output_dir / f"{prefix}_ctc_loss.png")
        print(f"  ✓ CTC Loss plot saved")
        
        # PER
        self.plot_per(data_dicts, 
                     save_path=self.output_dir / f"{prefix}_per.png")
        print(f"  ✓ PER plot saved")
        
        # Train vs Test
        self.plot_train_vs_test(data_dicts, 
                               save_path=self.output_dir / f"{prefix}_train_vs_test.png")
        print(f"  ✓ Train vs Test plot saved")
        
        # Learning Rate
        self.plot_learning_rate(data_dicts, 
                               save_path=self.output_dir / f"{prefix}_learning_rate.png")
        print(f"  ✓ Learning Rate plot saved")
        
        # Gradient Norms
        self.plot_gradient_norms(data_dicts, 
                                save_path=self.output_dir / f"{prefix}_gradient_norms.png")
        print(f"  ✓ Gradient Norms plot saved")
        
        # Memory Usage
        self.plot_memory_usage(data_dicts, 
                              save_path=self.output_dir / f"{prefix}_memory.png")
        print(f"  ✓ Memory Usage plot saved")
        
        # Eval Comparison
        self.plot_eval_comparison(data_dicts, 
                                 save_path=self.output_dir / f"{prefix}_eval_comparison.png")
        print(f"  ✓ Evaluation Comparison plot saved")
        
        print(f"\nAll plots saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training results from checkpoint directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all runs in data/checkpoints
  python scripts/plot_training_results.py
  
  # Plot specific runs
  python scripts/plot_training_results.py --runs gru_ctc_reg1 run11_plateau_break
  
  # Plot with custom output directory
  python scripts/plot_training_results.py --output plots/my_analysis
  
  # Plot only specific graphs
  python scripts/plot_training_results.py --graphs ctc_loss per train_vs_test
        """
    )
    
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='data/checkpoints',
        help='Directory containing checkpoint folders (default: data/checkpoints)'
    )
    
    parser.add_argument(
        '--runs',
        type=str,
        nargs='+',
        default=None,
        help='Specific run names to plot (default: all runs)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    
    parser.add_argument(
        '--graphs',
        type=str,
        nargs='+',
        choices=['all', 'ctc_loss', 'per', 'train_vs_test', 'learning_rate', 
                'gradient_norms', 'memory', 'eval_comparison'],
        default=['all'],
        help='Which graphs to generate (default: all)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='training',
        help='Prefix for output filenames (default: training)'
    )
    
    args = parser.parse_args()
    
    # Find all checkpoint directories
    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    # Get list of runs to process
    if args.runs:
        run_dirs = [checkpoints_dir / run for run in args.runs]
        run_dirs = [d for d in run_dirs if d.exists() and d.is_dir()]
    else:
        run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print(f"Error: No checkpoint directories found in {checkpoints_dir}")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} checkpoint directory(ies)")
    
    # Load data from all runs
    data_dicts = []
    for run_dir in run_dirs:
        loader = CheckpointDataLoader(run_dir)
        data = loader.get_all_data()
        if data['metrics'] or data['training_stats']:
            data_dicts.append(data)
            print(f"  ✓ Loaded {data['run_name']}")
        else:
            print(f"  ⚠ Skipped {data['run_name']} (no metrics found)")
    
    if not data_dicts:
        print("Error: No data found in any checkpoint directory")
        sys.exit(1)
    
    # Create plots
    plotter = TrainingPlotter(output_dir=args.output)
    
    if 'all' in args.graphs:
        plotter.plot_all(data_dicts, prefix=args.prefix)
    else:
        # Plot individual graphs
        if 'ctc_loss' in args.graphs:
            plotter.plot_ctc_loss(data_dicts, 
                                 save_path=plotter.output_dir / f"{args.prefix}_ctc_loss.png")
            print(f"  ✓ CTC Loss plot saved")
        
        if 'per' in args.graphs:
            plotter.plot_per(data_dicts, 
                            save_path=plotter.output_dir / f"{args.prefix}_per.png")
            print(f"  ✓ PER plot saved")
        
        if 'train_vs_test' in args.graphs:
            plotter.plot_train_vs_test(data_dicts, 
                                      save_path=plotter.output_dir / f"{args.prefix}_train_vs_test.png")
            print(f"  ✓ Train vs Test plot saved")
        
        if 'learning_rate' in args.graphs:
            plotter.plot_learning_rate(data_dicts, 
                                      save_path=plotter.output_dir / f"{args.prefix}_learning_rate.png")
            print(f"  ✓ Learning Rate plot saved")
        
        if 'gradient_norms' in args.graphs:
            plotter.plot_gradient_norms(data_dicts, 
                                       save_path=plotter.output_dir / f"{args.prefix}_gradient_norms.png")
            print(f"  ✓ Gradient Norms plot saved")
        
        if 'memory' in args.graphs:
            plotter.plot_memory_usage(data_dicts, 
                                     save_path=plotter.output_dir / f"{args.prefix}_memory.png")
            print(f"  ✓ Memory Usage plot saved")
        
        if 'eval_comparison' in args.graphs:
            plotter.plot_eval_comparison(data_dicts, 
                                        save_path=plotter.output_dir / f"{args.prefix}_eval_comparison.png")
            print(f"  ✓ Evaluation Comparison plot saved")
        
        print(f"\nPlots saved to {plotter.output_dir}/")


if __name__ == '__main__':
    main()

