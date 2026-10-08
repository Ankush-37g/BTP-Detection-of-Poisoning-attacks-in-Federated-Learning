"""
Phase 1 unit tests.

Tests:
    1. IID partitioning
    2. Non-IID Dirichlet partitioning
    3. Model forward pass (LeNet5, SimpleCNN)
    4. Model penultimate representation extraction
    5. FedAvg aggregation (weighted and uniform)
    6. FL client training (one round)
    7. FL server evaluation
    8. Full mini FL experiment (2 clients, 2 rounds, MNIST)

Run with:
    cd BTP-FL-Poisoning
    .venv/bin/python -m pytest tests/test_phase1.py -v
"""

import os
import sys
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.utils.reproducibility import set_seed
from proposed.utils.device import get_device
from proposed.data.partitioner import (
    iid_partition,
    dirichlet_partition,
    partition_summary,
    make_client_datasets,
)
from proposed.models.lenet import LeNet5, SimpleCNN, build_model
from proposed.aggregation.fedavg import fedavg
from proposed.federated.client import FLClient
from proposed.federated.server import FLServer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_dataset():
    """A small dummy dataset (100 samples, 10 classes) for fast tests."""
    set_seed(0)
    images = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    ds = TensorDataset(images, labels)
    ds.targets = labels  # needed by partitioner
    return ds


@pytest.fixture
def device():
    return get_device()


# ── 1. IID Partitioning ───────────────────────────────────────────────────────

class TestIIDPartition:

    def test_partition_covers_all_indices(self, dummy_dataset):
        set_seed(42)
        partition = iid_partition(dummy_dataset, num_clients=5)
        all_indices = np.concatenate(list(partition.values()))
        # Each index should appear at most once
        assert len(all_indices) == len(set(all_indices))

    def test_correct_number_of_clients(self, dummy_dataset):
        partition = iid_partition(dummy_dataset, num_clients=5)
        assert len(partition) == 5

    def test_roughly_equal_sizes(self, dummy_dataset):
        partition = iid_partition(dummy_dataset, num_clients=5)
        sizes = [len(v) for v in partition.values()]
        # All clients should have the same count (100 // 5 = 20)
        assert all(s == 20 for s in sizes)

    def test_client_ids_are_integers(self, dummy_dataset):
        partition = iid_partition(dummy_dataset, num_clients=4)
        assert all(isinstance(cid, int) for cid in partition.keys())


# ── 2. Non-IID Dirichlet Partitioning ────────────────────────────────────────

class TestDirichletPartition:

    def test_correct_number_of_clients(self, dummy_dataset):
        set_seed(42)
        partition = dirichlet_partition(dummy_dataset, num_clients=5, alpha=0.5)
        assert len(partition) == 5

    def test_all_samples_assigned(self, dummy_dataset):
        set_seed(42)
        partition = dirichlet_partition(dummy_dataset, num_clients=5, alpha=0.5)
        all_indices = np.concatenate(list(partition.values()))
        assert len(all_indices) == len(dummy_dataset)

    def test_no_duplicate_indices(self, dummy_dataset):
        set_seed(42)
        partition = dirichlet_partition(dummy_dataset, num_clients=5, alpha=0.5)
        all_indices = np.concatenate(list(partition.values()))
        assert len(all_indices) == len(set(all_indices))

    def test_low_alpha_is_more_skewed(self, dummy_dataset):
        """Low alpha should produce more unequal client sizes."""
        set_seed(42)
        p_high = dirichlet_partition(dummy_dataset, num_clients=5, alpha=100.0)
        p_low = dirichlet_partition(dummy_dataset, num_clients=5, alpha=0.1)
        std_high = np.std([len(v) for v in p_high.values()])
        std_low = np.std([len(v) for v in p_low.values()])
        assert std_low >= std_high  # lower alpha → higher variance

    def test_partition_summary_structure(self, dummy_dataset):
        set_seed(42)
        partition = dirichlet_partition(dummy_dataset, num_clients=3, alpha=0.5)
        summary = partition_summary(dummy_dataset, partition)
        assert len(summary) == 3
        for cid, stats in summary.items():
            assert "num_samples" in stats
            assert "class_counts" in stats
            assert "class_fractions" in stats


# ── 3. Model Forward Pass ─────────────────────────────────────────────────────

class TestModels:

    def test_lenet5_forward(self, device):
        model = LeNet5(num_classes=10).to(device)
        x = torch.randn(4, 1, 28, 28).to(device)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"

    def test_simplecnn_forward(self, device):
        model = SimpleCNN(num_classes=10).to(device)
        x = torch.randn(4, 1, 28, 28).to(device)
        out = model(x)
        assert out.shape == (4, 10)

    def test_lenet5_penultimate_dim(self, device):
        model = LeNet5(num_classes=10).to(device)
        x = torch.randn(4, 1, 28, 28).to(device)
        rep = model.get_penultimate_representation(x)
        assert rep.shape == (4, LeNet5.PENULTIMATE_DIM), \
            f"Expected (4, {LeNet5.PENULTIMATE_DIM}), got {rep.shape}"

    def test_simplecnn_penultimate_dim(self, device):
        model = SimpleCNN(num_classes=10).to(device)
        x = torch.randn(4, 1, 28, 28).to(device)
        rep = model.get_penultimate_representation(x)
        assert rep.shape == (4, SimpleCNN.PENULTIMATE_DIM)

    def test_build_model_factory(self, device):
        m = build_model("LeNet5", num_classes=10)
        assert isinstance(m, LeNet5)
        m = build_model("SimpleCNN", num_classes=10)
        assert isinstance(m, SimpleCNN)

    def test_build_model_raises_on_unknown(self):
        with pytest.raises(ValueError):
            build_model("UnknownModel")


# ── 4. FedAvg Aggregation ─────────────────────────────────────────────────────

class TestFedAvg:

    def _make_state_dict(self, value: float) -> dict:
        return {"layer": torch.full((3,), value, dtype=torch.float32)}

    def test_weighted_average(self):
        """FedAvg should weight by num_samples."""
        updates = [
            (self._make_state_dict(1.0), 100),   # weight = 100/150
            (self._make_state_dict(4.0), 50),    # weight = 50/150
        ]
        agg = fedavg(updates)
        expected = (1.0 * 100 + 4.0 * 50) / 150  # = 2.0
        assert torch.allclose(agg["layer"], torch.full((3,), expected), atol=1e-5)

    def test_uniform_average(self):
        """Uniform weighting should produce simple mean."""
        updates = [
            (self._make_state_dict(2.0), 100),
            (self._make_state_dict(4.0), 50),
        ]
        agg = fedavg(updates, uniform_weights=True)
        expected = (2.0 + 4.0) / 2  # = 3.0
        assert torch.allclose(agg["layer"], torch.full((3,), expected), atol=1e-5)

    def test_single_client(self):
        """With one client, aggregation returns the client's own weights."""
        sd = self._make_state_dict(5.0)
        agg = fedavg([(sd, 100)])
        assert torch.allclose(agg["layer"], sd["layer"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fedavg([])

    def test_output_is_on_cpu(self):
        updates = [(self._make_state_dict(1.0), 10)]
        agg = fedavg(updates)
        for v in agg.values():
            assert v.device.type == "cpu"


# ── 5. FL Client Training ─────────────────────────────────────────────────────

class TestFLClient:

    def test_client_returns_correct_types(self, dummy_dataset, device):
        set_seed(0)
        model = LeNet5(num_classes=10)
        client = FLClient(
            client_id=0,
            dataset=dummy_dataset,
            model=model,
            local_epochs=1,
            batch_size=16,
            device=device,
        )
        global_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        local_weights, num_samples, avg_loss, delta_w = client.train(global_weights)

        assert isinstance(local_weights, dict)
        assert isinstance(num_samples, int)
        assert isinstance(avg_loss, float)
        assert isinstance(delta_w, dict)

    def test_delta_w_equals_diff(self, dummy_dataset, device):
        """delta_w must equal local_weights - global_weights."""
        set_seed(0)
        model = LeNet5(num_classes=10)
        client = FLClient(0, dummy_dataset, model, local_epochs=1, device=device)
        global_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        local_weights, _, _, delta_w = client.train(global_weights)

        for key in delta_w:
            expected = local_weights[key] - global_weights[key]
            assert torch.allclose(delta_w[key], expected, atol=1e-6), \
                f"delta_w mismatch for key {key}"

    def test_num_samples_matches_dataset(self, dummy_dataset, device):
        set_seed(0)
        model = LeNet5(num_classes=10)
        client = FLClient(0, dummy_dataset, model, device=device)
        global_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        _, num_samples, _, _ = client.train(global_weights)
        assert num_samples == len(dummy_dataset)


# ── 6. Full Mini FL Experiment ────────────────────────────────────────────────

class TestMiniExperiment:

    def test_accuracy_is_valid_range(self, dummy_dataset, device):
        """Run 2 rounds on a tiny dataset and check accuracy is in [0, 1]."""
        set_seed(42)
        model = LeNet5(num_classes=10)
        clients = [
            FLClient(i, dummy_dataset, model, local_epochs=1, device=device)
            for i in range(2)
        ]
        server = FLServer(
            global_model=model,
            clients=clients,
            test_dataset=dummy_dataset,
            clients_per_round=2,
            device=device,
        )
        for rnd in range(2):
            record = server.run_round(rnd)
            assert 0.0 <= record["test_accuracy"] <= 1.0
            assert record["test_loss"] >= 0.0

    def test_history_length(self, dummy_dataset, device):
        set_seed(0)
        model = LeNet5(num_classes=10)
        clients = [FLClient(i, dummy_dataset, model, device=device) for i in range(2)]
        server = FLServer(model, clients, dummy_dataset, device=device)
        for rnd in range(3):
            server.run_round(rnd)
        assert len(server.history) == 3
