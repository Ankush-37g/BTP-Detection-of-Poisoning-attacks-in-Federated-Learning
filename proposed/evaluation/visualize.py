"""
Visualization utilities for FL experiments.

All plotting functions follow these conventions:
    - Accept pre-computed data (numpy arrays, dicts) rather than dataset objects.
      This keeps plots reproducible from saved CSVs without re-running experiments.
    - Save to a specified output path rather than calling plt.show(), so they
      work correctly in headless/script environments.
    - Return the figure object so callers can modify it before saving if needed.
    - Use a consistent colour scheme across the project.

Plots implemented here:
    1. plot_client_label_distribution  — stacked bar chart per client per alpha
    2. plot_accuracy_curve             — test accuracy vs round
    3. plot_loss_curve                 — test/client loss vs round
    4. plot_partition_heatmap          — class fraction heatmap (clients × classes)
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Project-wide style ─────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10", 10)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ── 1. Client label distribution ───────────────────────────────────────────────

def plot_client_label_distribution(
    class_counts_per_client: Dict[int, Dict[int, int]],
    num_classes: int = 10,
    alpha: Optional[float] = None,
    partition_type: str = "iid",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked bar chart showing the class composition of each client's dataset.

    Args:
        class_counts_per_client: {client_id: {class_label: count}}
        num_classes:  Total number of classes.
        alpha:        Dirichlet alpha (used for title). None if IID.
        partition_type: "iid" or "noniid"
        save_path:    If given, saves the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    num_clients = len(class_counts_per_client)
    client_ids = sorted(class_counts_per_client.keys())

    # Build matrix: rows=clients, cols=classes
    matrix = np.zeros((num_clients, num_classes), dtype=int)
    for row_idx, cid in enumerate(client_ids):
        for cls, cnt in class_counts_per_client[cid].items():
            matrix[row_idx, int(cls)] = cnt

    fig, ax = plt.subplots(figsize=(max(10, num_clients * 0.9), 5))

    bottoms = np.zeros(num_clients)
    for cls in range(num_classes):
        values = matrix[:, cls]
        ax.bar(
            range(num_clients),
            values,
            bottom=bottoms,
            color=PALETTE[cls % len(PALETTE)],
            label=f"Class {cls}",
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += values

    # Labels and legend
    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f"C{cid}" for cid in client_ids], fontsize=8)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")

    if partition_type == "iid":
        title = "Client Label Distribution — IID"
    else:
        title = f"Client Label Distribution — Non-IID  (Dirichlet α = {alpha})"
    ax.set_title(title, fontweight="bold")

    ax.legend(
        loc="upper right",
        ncol=5,
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.8,
        fontsize=8,
    )
    ax.set_xlim(-0.6, num_clients - 0.4)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [Plot] Saved → {save_path}")

    return fig


# ── 2. Partition heatmap ───────────────────────────────────────────────────────

def plot_partition_heatmap(
    class_fractions_per_client: Dict[int, Dict[int, float]],
    num_classes: int = 10,
    alpha: Optional[float] = None,
    partition_type: str = "iid",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of class fractions per client (clients × classes).

    Rows = clients, Columns = class labels, Cell = fraction of that client's
    data belonging to that class. A perfectly IID partition would show all
    cells with identical values (~0.1 for 10 classes).
    """
    client_ids = sorted(class_fractions_per_client.keys())
    num_clients = len(client_ids)

    matrix = np.zeros((num_clients, num_classes))
    for row_idx, cid in enumerate(client_ids):
        for cls, frac in class_fractions_per_client[cid].items():
            matrix[row_idx, int(cls)] = frac

    fig, ax = plt.subplots(figsize=(10, max(4, num_clients * 0.5)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        annot=(num_clients <= 15),   # only annotate if readable
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Class Fraction"},
        xticklabels=[f"C{c}" for c in range(num_classes)],
        yticklabels=[f"Client {cid}" for cid in client_ids],
    )

    if partition_type == "iid":
        title = "Class Fraction Heatmap — IID"
    else:
        title = f"Class Fraction Heatmap — Non-IID  (α = {alpha})"
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Client")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [Plot] Saved → {save_path}")

    return fig


# ── 3. Accuracy curve ──────────────────────────────────────────────────────────

def plot_accuracy_curve(
    rounds: List[int],
    accuracies: List[float],
    label: str = "FedAvg",
    title: str = "Global Test Accuracy vs Round",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Line plot of global test accuracy over FL rounds.

    Can be called multiple times with the same `ax` to overlay multiple
    experiment curves on one figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(rounds, [a * 100 for a in accuracies], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if standalone:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            print(f"  [Plot] Saved → {save_path}")

    return fig


# ── 4. Loss curve ──────────────────────────────────────────────────────────────

def plot_loss_curve(
    rounds: List[int],
    losses: List[float],
    label: str = "Test Loss",
    title: str = "Loss vs Round",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of loss over FL rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, losses, marker="s", linewidth=2, color="tomato", label=label)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [Plot] Saved → {save_path}")

    return fig
