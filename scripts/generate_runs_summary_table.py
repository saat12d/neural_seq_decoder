#!/usr/bin/env python3

import json
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


class RunDataLoader:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = self.checkpoint_dir.name
        
    def load_metrics_jsonl(self):
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
    
    def get_best_per(self):
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
    
    def get_test_per(self):
        eval_results = {}
        for eval_file in self.checkpoint_dir.glob("eval*.json"):
            try:
                with open(eval_file, 'r') as f:
                    eval_results[eval_file.stem] = json.load(f)
            except Exception:
                pass
        
        for eval_name, eval_data in eval_results.items():
            if 'avg_PER' in eval_data:
                split = eval_data.get('split', '').lower()
                method = eval_data.get('decoding_method', '').lower()
                
                if 'competition' in split or 'competition' in eval_name.lower():
                    continue
                
                if 'test' in split or 'test' in eval_name.lower():
                    if method == 'greedy':
                        return eval_data['avg_PER']
        
        for eval_name, eval_data in eval_results.items():
            if 'avg_PER' in eval_data:
                if 'competition' not in eval_name.lower():
                    return eval_data['avg_PER']
        
        return None
    
    def get_config_from_log(self):
        log_file = self.checkpoint_dir / "train.log"
        if not log_file.exists():
            return {}
        
        config = {}
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract EMA
            match = re.search(r'EMA enabled: decay=([\d.]+)', content)
            if match:
                config['ema_decay'] = float(match.group(1))
            else:
                config['ema_decay'] = None
            
            # Extract scheduler info
            if "Warmup→Cosine" in content or ("Warmup steps:" in content and "Cosine steps:" in content):
                match = re.search(r'Warmup steps: (\d+)', content)
                warmup = int(match.group(1)) if match else None
                match = re.search(r'Cosine steps: (\d+)', content)
                cosine = int(match.group(1)) if match else None
                match = re.search(r'Peak LR: ([\d.e-]+)', content)
                peak_lr = match.group(1) if match else None
                match = re.search(r'End LR: ([\d.e-]+)', content)
                end_lr = match.group(1) if match else None
                
                if warmup and cosine:
                    config['scheduler'] = f"Warmup→Cosine (warmup={warmup}, cosine={cosine})"
                    if peak_lr and end_lr:
                        config['scheduler'] += f", LR={peak_lr}→{end_lr}"
                else:
                    config['scheduler'] = "Warmup→Cosine"
            elif "OneCycleLR" in content:
                config['scheduler'] = "OneCycleLR"
            elif "constant LR" in content:
                config['scheduler'] = "Constant"
            else:
                config['scheduler'] = "Unknown"
            
            # Extract input dropout
            match = re.search(r'Input dropout: ([\d.]+)', content)
            if match:
                config['input_dropout'] = float(match.group(1))
            else:
                config['input_dropout'] = 0.0
        
        return config


def generate_summary_table(output_dir: Path):
    print("Generating runs summary table...")
    
    runs = [
        {
            'run': 'speechBaseline4',
            'change': 'Baseline',
            'notes': 'Constant LR=0.02, FP32, heavy augmentation',
            'val_per': None,
            'test_per': None,
        },
        {
            'run': 'run11_plateau_break',
            'change': 'OneCycleLR + AMP',
            'notes': 'OneCycleLR max_lr=0.004, BF16, grad_accum=2',
            'val_per': None,
            'test_per': None,
        },
        {
            'run': 'run15_warmup_cosine_safe',
            'change': 'Warmup→Cosine (no EMA)',
            'notes': 'Warmup 1200→Cosine 8800, peak LR=1.5e-3, no EMA',
            'val_per': None,
            'test_per': None,
        },
        {
            'run': 'run16_greedy_ema_warmcos',
            'change': 'EMA 0.999',
            'notes': 'Warmup→Cosine + EMA=0.999, peak LR=1.6e-3',
            'val_per': None,
            'test_per': None,
        },
        {
            'run': 'run16_4',
            'change': 'EMA 0.9995 + input_dropout',
            'notes': 'EMA=0.9995, input_dropout=0.05, peak LR=1.4e-3',
            'val_per': None,
            'test_per': None,
        },
        {
            'run': 'run16_5',
            'change': 'Long-tail + EMA 0.9995',
            'notes': '20k batches, cosine=18500, EMA=0.9995, input_dropout=0.05, peak LR=0.001',
            'val_per': None,
            'test_per': None,
        },
    ]
    
    checkpoints_dir = Path("data/checkpoints")
    
    for run_info in runs:
        run_name = run_info['run']
        run_dir = checkpoints_dir / run_name
        
        if not run_dir.exists():
            print(f"  Warning: {run_name} not found")
            continue
        
        loader = RunDataLoader(str(run_dir))
        val_per, _ = loader.get_best_per()
        
        if run_name == "speechBaseline4" and val_per is None:
            val_per = loader.get_test_per()
        
        test_per = loader.get_test_per()
        
        run_info['val_per'] = val_per
        run_info['test_per'] = test_per
    
    data = []
    for run_info in runs:
        data.append({
            'Run': run_info['run'],
            'Change': run_info['change'],
            'Val PER': f"{run_info['val_per']:.4f}" if run_info['val_per'] else "N/A",
            'Test PER': f"{run_info['test_per']:.4f}" if run_info['test_per'] else "N/A",
            'Notes': run_info['notes'],
        })
    
    df = pd.DataFrame(data)
    
    output_file = output_dir / "runs_summary_table.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    
    latex_file = output_dir / "runs_summary_table.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l l c c p{5cm}}\n")
        f.write("\\toprule\n")
        f.write("Run & Change & Val PER & Test PER & Notes \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            run = row['Run']
            change = row['Change']
            val_per = row['Val PER']
            test_per = row['Test PER']
            notes = row['Notes'].replace('&', '\\&')
            f.write(f"{run} & {change} & {val_per} & {test_per} & {notes} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary of key training runs showing progression from baseline to best configuration.}\n")
        f.write("\\label{tab:runs_summary}\n")
        f.write("\\end{table}\n")
    
    print(f"  ✓ Saved LaTeX to {latex_file}")
    
    md_file = output_dir / "runs_summary_table.md"
    with open(md_file, 'w') as f:
        f.write("# Key Training Runs Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    
    print(f"  ✓ Saved Markdown to {md_file}")
    
    print("\n" + "=" * 80)
    print("Runs Summary Table:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='paper_figures_v2', 
                       help='Output directory')
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_summary_table(output_dir)


if __name__ == '__main__':
    main()


