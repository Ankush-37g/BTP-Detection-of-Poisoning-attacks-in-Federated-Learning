"""
Partition Analysis Script — Phase 2

Analyses and visualizes how Dirichlet(alpha) Non-IID partitioning affects
the distribution of training data across FL clients.

Runs analysis for:
    - IID (baseline)
    - Non-IID with alpha ∈ {10.0, 1.0, 0.5, 0.1}

Outputs saved to results/partitions/:
    - per_client_counts_<type>.csv  — sample counts per client per class
    - label_dist_<type>.png         — stacked bar chart
    - heatmap_<type>.png            — class fraction heatmap
    - summary.csv                   — heterogeneity statistics across alpha values

Usage:
    cd BTP-FL-Poisoning
    python experiments/run_partition_analysis.py --dataset MNIST --num_clients 10
    python experiments/run_partition_analysis.py --dataset MNIST --num_clients 10 --alpha 0.1 0.5 1.0 10.0
"""

import argparse
import csv
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.utils.reproducibility import set_seed
from proposed.data.datasets import load_dataset, get_num_classes
from proposed.data.partitioner import (
    iid_partition,
    dirichlet_partition,
    partition_summary,
    make_client_datasets,
)
from proposed.evaluation.visualize import (
    plot_client_label_distribution,
    plot_partition_heatmap,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse and visualize FL data partitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[10.0, 1.0, 0.5, 0.1],
                        help="Dirichlet alpha values to analyse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "experiments", "results", "partitions"))
    return parser.parse_args()


def analyse_partition(dataset, partition, summary, partition_tag,
                      num_classes, results_dir, alpha=None, partition_type="iid"):
    """Run analysis and save outputs for one partition configuration."""
    tag_dir = os.path.join(results_dir, partition_tag)
    os.makedirs(tag_dir, exist_ok=True)

    # ── Save per-client counts CSV ─────────────────────────────────────────────
    csv_path = os.path.join(tag_dir, "per_client_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["client_id", "total_samples"] + [f"class_{c}" for c in range(num_classes)]
        writer.writerow(header)
        for cid in sorted(summary.keys()):
            row = [cid, summary[cid]["num_samples"]]
            for c in range(num_classes):
                row.append(summary[cid]["class_counts"].get(c, 0))
            writer.writerow(row)
    print(f"  [CSV]  {csv_path}")

    # ── Compute heterogeneity statistics ──────────────────────────────────────
    sample_sizes = [summary[cid]["num_samples"] for cid in sorted(summary.keys())]
    class_counts = {
        cid: summary[cid]["class_counts"] for cid in sorted(summary.keys())
    }
    class_fractions = {
        cid: summary[cid]["class_fractions"] for cid in sorted(summary.keys())
    }

    # Earth Mover's Distance proxy: std of per-client dominant-class fraction
    dom_fracs = [max(summary[cid]["class_fractions"].values())
                 for cid in sorted(summary.keys())]
    heterogeneity_score = float(np.std(dom_fracs))

    print(f"  Samples — min: {min(sample_sizes):,}  max: {max(sample_sizes):,}  "
          f"mean: {np.mean(sample_sizes):,.0f}  std: {np.std(sample_sizes):.1f}")
    print(f"  Heterogeneity score (std of dominant-class fraction): {heterogeneity_score:.4f}")

    # ── Label distribution bar chart ──────────────────────────────────────────
    bar_path = os.path.join(tag_dir, "label_distribution.png")
    plot_client_label_distribution(
        class_counts_per_client=class_counts,
        num_classes=num_classes,
        alpha=alpha,
        partition_type=partition_type,
        save_path=bar_path,
    )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap_path = os.path.join(tag_dir, "heatmap.png")
    plot_partition_heatmap(
        class_fractions_per_client=class_fractions,
        num_classes=num_classes,
        alpha=alpha,
        partition_type=partition_type,
        save_path=heatmap_path,
    )

    return {
        "partition_type": partition_type,
        "alpha": alpha,
        "min_samples": min(sample_sizes),
        "max_samples": max(sample_sizes),
        "mean_samples": float(np.mean(sample_sizes)),
        "std_samples": float(np.std(sample_sizes)),
        "heterogeneity_score": heterogeneity_score,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("PARTITION ANALYSIS")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Clients:     {args.num_clients}")
    print(f"  Alpha vals:  {args.alpha}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output:      {args.results_dir}")
    print("=" * 60)

    # Load dataset
    train_dataset, _ = load_dataset(args.dataset, data_root=args.data_root)
    num_classes = get_num_classes(args.dataset)

    all_stats = []

    # ── IID baseline ──────────────────────────────────────────────────────────
    print("\n── IID Partition ──")
    set_seed(args.seed)
    partition_iid = iid_partition(train_dataset, args.num_clients)
    summary_iid = partition_summary(train_dataset, partition_iid)
    stats = analyse_partition(
        train_dataset, partition_iid, summary_iid,
        partition_tag="iid",
        num_classes=num_classes,
        results_dir=args.results_dir,
        alpha=None,
        partition_type="iid",
    )
    all_stats.append(stats)

    # ── Non-IID per alpha ─────────────────────────────────────────────────────
    for alpha in sorted(args.alpha, reverse=True):
        print(f"\n── Non-IID  α = {alpha} ──")
        set_seed(args.seed)    # same seed → same permutation, different proportions
        partition_noniid = dirichlet_partition(train_dataset, args.num_clients, alpha=alpha)
        summary_noniid = partition_summary(train_dataset, partition_noniid)
        tag = f"noniid_alpha_{alpha}".replace(".", "_")
        stats = analyse_partition(
            train_dataset, partition_noniid, summary_noniid,
            partition_tag=tag,
            num_classes=num_classes,
            results_dir=args.results_dir,
            alpha=alpha,
            partition_type="noniid",
        )
        all_stats.append(stats)

    # ── Summary CSV across all configs ────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"\n[Summary] {summary_path}")
    print("\n── Heterogeneity Summary ──")
    print(f"{'Type':<25} {'Alpha':>6} {'Min':>6} {'Max':>6} {'Mean':>8} {'HetScore':>10}")
    print("-" * 65)
    for s in all_stats:
        alpha_str = str(s["alpha"]) if s["alpha"] is not None else " IID"
        print(f"{'IID' if s['partition_type']=='iid' else 'Non-IID':<25} "
              f"{alpha_str:>6} "
              f"{s['min_samples']:>6.0f} "
              f"{s['max_samples']:>6.0f} "
              f"{s['mean_samples']:>8.0f} "
              f"{s['heterogeneity_score']:>10.4f}")

    print("\n✓ Partition analysis complete.")


if __name__ == "__main__":
    main()
