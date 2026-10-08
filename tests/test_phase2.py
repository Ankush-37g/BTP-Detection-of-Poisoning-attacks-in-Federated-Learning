"""
Phase 2 unit tests — Non-IID partitioning properties.

Tests verify:
    1. Dirichlet partition covers all training samples
    2. Lower alpha produces more heterogeneous distributions (higher std)
    3. Heterogeneity score (std of dominant-class fraction) increases as alpha decreases
    4. make_client_datasets returns proper Subset objects
    5. partition_summary has correct structure and sums
    6. Visualization functions run without errors and produce files

Run with:
    cd BTP-FL-Poisoning
    .venv/bin/python -m pytest tests/test_phase2.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.utils.reproducibility import set_seed
from proposed.data.partitioner import (
    iid_partition,
    dirichlet_partition,
    partition_summary,
    make_client_datasets,
)
from proposed.evaluation.visualize import (
    plot_client_label_distribution,
    plot_partition_heatmap,
    plot_accuracy_curve,
    plot_loss_curve,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mnist_like():
    """200-sample dummy dataset with 10 balanced classes (20 each)."""
    set_seed(0)
    n, num_classes = 200, 10
    images = torch.randn(n, 1, 28, 28)
    labels = torch.tensor([i % num_classes for i in range(n)])
    ds = TensorDataset(images, labels)
    ds.targets = labels
    return ds


# ── 1. Non-IID coverage and correctness ──────────────────────────────────────

class TestNonIIDCoverage:

    def test_all_samples_assigned(self, mnist_like):
        set_seed(42)
        partition = dirichlet_partition(mnist_like, num_clients=10, alpha=0.5)
        total = sum(len(v) for v in partition.values())
        assert total == len(mnist_like)

    def test_no_index_duplicates(self, mnist_like):
        set_seed(42)
        partition = dirichlet_partition(mnist_like, num_clients=10, alpha=0.5)
        all_idx = np.concatenate(list(partition.values()))
        assert len(all_idx) == len(set(all_idx))

    def test_correct_num_clients(self, mnist_like):
        set_seed(42)
        for k in [5, 10, 20]:
            partition = dirichlet_partition(mnist_like, num_clients=k, alpha=1.0)
            assert len(partition) == k

    def test_alpha_1_less_skewed_than_alpha_01(self, mnist_like):
        """Higher alpha → lower heterogeneity → lower std of dominant class fractions."""
        set_seed(42)
        p_high = dirichlet_partition(mnist_like, num_clients=5, alpha=1.0)
        p_low  = dirichlet_partition(mnist_like, num_clients=5, alpha=0.1)

        def dom_std(partition, dataset):
            s = partition_summary(dataset, partition)
            return np.std([max(v["class_fractions"].values()) for v in s.values()])

        assert dom_std(p_low, mnist_like) >= dom_std(p_high, mnist_like)


# ── 2. partition_summary structure ───────────────────────────────────────────

class TestPartitionSummary:

    def test_summary_keys(self, mnist_like):
        set_seed(0)
        partition = dirichlet_partition(mnist_like, num_clients=5, alpha=0.5)
        summary = partition_summary(mnist_like, partition)
        for cid, stats in summary.items():
            assert "num_samples" in stats
            assert "class_counts" in stats
            assert "class_fractions" in stats

    def test_class_fractions_sum_to_one(self, mnist_like):
        set_seed(0)
        partition = iid_partition(mnist_like, num_clients=5)
        summary = partition_summary(mnist_like, partition)
        for cid, stats in summary.items():
            total_frac = sum(stats["class_fractions"].values())
            assert abs(total_frac - 1.0) < 1e-6, \
                f"Client {cid} fractions sum to {total_frac}, expected 1.0"

    def test_class_counts_match_num_samples(self, mnist_like):
        set_seed(0)
        partition = dirichlet_partition(mnist_like, num_clients=5, alpha=0.5)
        summary = partition_summary(mnist_like, partition)
        for cid, stats in summary.items():
            count_total = sum(stats["class_counts"].values())
            assert count_total == stats["num_samples"]


# ── 3. make_client_datasets ────────────────────────────────────────────────────

class TestMakeClientDatasets:

    def test_returns_subsets(self, mnist_like):
        from torch.utils.data import Subset
        set_seed(0)
        partition = iid_partition(mnist_like, num_clients=5)
        subsets = make_client_datasets(mnist_like, partition)
        for cid, ds in subsets.items():
            assert isinstance(ds, Subset)

    def test_subset_lengths_match_partition(self, mnist_like):
        set_seed(0)
        partition = dirichlet_partition(mnist_like, num_clients=5, alpha=0.5)
        subsets = make_client_datasets(mnist_like, partition)
        for cid in partition:
            assert len(subsets[cid]) == len(partition[cid])


# ── 4. Visualization functions ────────────────────────────────────────────────

class TestVisualization:

    def _make_counts(self, num_clients=5, num_classes=10):
        """Build fake class counts for testing."""
        counts = {}
        for cid in range(num_clients):
            counts[cid] = {c: np.random.randint(5, 50) for c in range(num_classes)}
        return counts

    def _make_fractions(self, num_clients=5, num_classes=10):
        """Build fake class fractions (summing to 1 per client)."""
        fracs = {}
        for cid in range(num_clients):
            raw = np.random.dirichlet(np.ones(num_classes))
            fracs[cid] = {c: float(raw[c]) for c in range(num_classes)}
        return fracs

    def test_label_dist_plot_saves_file(self, tmp_path):
        counts = self._make_counts()
        path = str(tmp_path / "test_dist.png")
        fig = plot_client_label_distribution(
            class_counts_per_client=counts,
            num_classes=10,
            alpha=0.5,
            partition_type="noniid",
            save_path=path,
        )
        assert os.path.exists(path)
        assert fig is not None

    def test_heatmap_saves_file(self, tmp_path):
        fracs = self._make_fractions()
        path = str(tmp_path / "test_heatmap.png")
        fig = plot_partition_heatmap(
            class_fractions_per_client=fracs,
            num_classes=10,
            alpha=0.5,
            partition_type="noniid",
            save_path=path,
        )
        assert os.path.exists(path)

    def test_accuracy_curve_saves_file(self, tmp_path):
        path = str(tmp_path / "test_acc.png")
        fig = plot_accuracy_curve(
            rounds=list(range(5)),
            accuracies=[0.5, 0.7, 0.8, 0.85, 0.9],
            save_path=path,
        )
        assert os.path.exists(path)

    def test_loss_curve_saves_file(self, tmp_path):
        path = str(tmp_path / "test_loss.png")
        fig = plot_loss_curve(
            rounds=list(range(5)),
            losses=[2.0, 1.5, 1.0, 0.7, 0.5],
            save_path=path,
        )
        assert os.path.exists(path)
