"""
generate_plots.py
Reads the results from the experiments/results directory and generates comparison plots.
"""

import os
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results_dir", type=str, default="experiments/results", help="Directory containing result folders")
    parser.add_argument("--attack", type=str, default="scaling", help="Attack type to filter and plot")
    parser.add_argument("--out_dir", type=str, default="experiments/plots", help="Output directory for plots")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all config.json files in subdirectories
    config_paths = glob.glob(os.path.join(args.results_dir, "*", "config.json"))
    
    experiments = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        if cfg.get('attack') != args.attack:
            continue
            
        csv_path = os.path.join(os.path.dirname(config_path), "rounds.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        defense = cfg.get('defense', 'unknown')
        
        # Add metadata columns
        df['Defense'] = defense.capitalize() if defense != 'proposed' else 'Proposed (Ours)'
        experiments.append(df)
        
    if not experiments:
        print(f"No results found for attack: {args.attack}")
        return
        
    # Combine all dataframes
    all_data = pd.concat(experiments, ignore_index=True)
    
    # 1. Plot Test Accuracy over Rounds
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_data, x='round', y='test_accuracy', hue='Defense', marker='o', linewidth=2)
    plt.title(f"Test Accuracy vs. Rounds under {args.attack.capitalize()} Attack (Non-IID)")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    acc_plot_path = os.path.join(args.out_dir, f"accuracy_vs_rounds_{args.attack}.png")
    plt.savefig(acc_plot_path, dpi=300)
    print(f"Saved accuracy plot to {acc_plot_path}")
    
    # 2. Plot Test Loss over Rounds
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_data, x='round', y='test_loss', hue='Defense', marker='o', linewidth=2)
    plt.title(f"Test Loss vs. Rounds under {args.attack.capitalize()} Attack (Non-IID)")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(args.out_dir, f"loss_vs_rounds_{args.attack}.png")
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Saved loss plot to {loss_plot_path}")

if __name__ == "__main__":
    main()
