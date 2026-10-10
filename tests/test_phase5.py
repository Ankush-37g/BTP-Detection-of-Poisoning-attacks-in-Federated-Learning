"""
Phase 5 unit tests — Clustering-based state-of-the-art baselines.

Tests verify:
    1. Linear CKA computation correctly identifies highly correlated vs uncorrelated matrices.
    2. FedCVG correctly clusters by update norms and filters large-norm attackers (like scaling attackers).
    3. FedCC correctly clusters by CKA and filters attackers with dissimilar representations.

Run with:
    cd BTP-FL-Poisoning
    .venv/bin/python -m pytest tests/test_phase5.py -v
"""

import os
import sys

import pytest
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.aggregation.baselines import fedcvg, fedcc, linear_cka


# ── 1. Linear CKA ─────────────────────────────────────────────────────────────

class TestLinearCKA:
    def test_cka_identical_matrices(self):
        X = torch.randn(10, 5)
        # Identical matrices should have CKA = 1.0 (or very close)
        score = linear_cka(X, X)
        assert torch.isclose(torch.tensor(score), torch.tensor(1.0), atol=1e-5)

    def test_cka_orthogonal_matrices(self):
        # We need at least 3 rows to have meaningful variance after centering
        X = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
        Y = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, -2.0]])
        # X variance is purely between row 1 and 2. Y variance is purely between 1,2 and 3.
        # They should have very low CKA.
        score = linear_cka(X, Y)
        assert score < 1e-5

    def test_cka_scaled_matrices(self):
        X = torch.randn(10, 5)
        Y = X * 5.0
        # CKA is invariant to isotropic scaling
        score = linear_cka(X, Y)
        assert torch.isclose(torch.tensor(score), torch.tensor(1.0), atol=1e-5)


# ── 2. FedCVG (Norm-based KMeans) ──────────────────────────────────────────────

class TestFedCVG:
    def test_fedcvg_filters_scaling_attack(self):
        # Global weights
        global_w = {"layer": torch.tensor([1.0, 1.0])}
        
        # 3 benign clients with small updates (norms ~ 0.1)
        b1 = ({"layer": torch.tensor([1.1, 1.0])}, 10)
        b2 = ({"layer": torch.tensor([1.0, 1.1])}, 10)
        b3 = ({"layer": torch.tensor([0.9, 1.0])}, 10)
        
        # 2 attackers with massive updates (scaling attack)
        a1 = ({"layer": torch.tensor([10.0, 10.0])}, 10)
        a2 = ({"layer": torch.tensor([10.5, 9.5])}, 10)
        
        client_updates = [b1, b2, b3, a1, a2]
        
        # FedCVG should cluster them into {b1,b2,b3} and {a1,a2}
        # It assumes the larger cluster (3 benign) is safe
        agg = fedcvg(client_updates, global_weights=global_w)
        
        # The aggregated value should be the average of b1, b2, b3
        # b1+b2+b3 = [3.0, 3.1] / 3 = [1.0, 1.0333]
        expected = torch.tensor([1.0, 1.033333])
        assert torch.allclose(agg["layer"], expected, atol=1e-4)

    def test_fedcvg_requires_global_weights(self):
        with pytest.raises(ValueError):
            fedcvg([({"layer": torch.tensor([1.0])}, 10)])


# ── 3. FedCC (CKA-based KMeans) ────────────────────────────────────────────────

class TestFedCC:
    def test_fedcc_filters_dissimilar_features(self):
        # We need at least 3 rows to avoid CKA collapsing to 1.0
        global_w = {
            "k1": torch.tensor([0.0]),
            "k2": torch.tensor([0.0]),
            "pl_layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), # keys[-4]
            "k4": torch.tensor([0.0]),
            "k5": torch.tensor([0.0]),
            "k6": torch.tensor([0.0])
        }
        
        # Benign clients have PLR identical to global
        b1_pl = global_w["pl_layer"].clone()
        b2_pl = global_w["pl_layer"].clone()
        b3_pl = global_w["pl_layer"].clone()
        
        # Attackers have completely orthogonal/dissimilar PLR
        a1_pl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
        a2_pl = torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        
        def make_client(pl):
            return {
                "k1": torch.tensor([0.0]),
                "k2": torch.tensor([0.0]),
                "pl_layer": pl,
                "k4": torch.tensor([0.0]),
                "k5": torch.tensor([0.0]),
                "k6": torch.tensor([0.0])
            }
            
        client_updates = [
            (make_client(b1_pl), 10),
            (make_client(b2_pl), 10),
            (make_client(b3_pl), 10),
            (make_client(a1_pl), 10),
            (make_client(a2_pl), 10)
        ]
        
        agg = fedcc(client_updates, global_weights=global_w)
        
        # The aggregated PLR should be the average of b1, b2, b3
        expected_pl = (b1_pl + b2_pl + b3_pl) / 3.0
        assert torch.allclose(agg["pl_layer"], expected_pl, atol=1e-4)
