"""
Unit tests for the proposed Phase 6/7 defense mechanisms.
"""

import os
import sys

import pytest
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.aggregation.proposed import extract_features, proposed_defense


class TestProposedDefense:
    def test_feature_extraction_dimensions(self):
        # 3 clients, mock architecture
        global_w = {
            "k1": torch.tensor([0.0]),
            "k2": torch.tensor([0.0]),
            "pl_layer": torch.randn(5, 5), # keys[-4]
            "k4": torch.tensor([0.0]),
            "k5": torch.tensor([0.0]),
            "k6": torch.tensor([0.0])
        }
        
        def make_client():
            return {
                "k1": torch.tensor([0.1]),
                "k2": torch.tensor([0.1]),
                "pl_layer": torch.randn(5, 5),
                "k4": torch.tensor([0.1]),
                "k5": torch.tensor([0.1]),
                "k6": torch.tensor([0.1])
            }
            
        client_updates = [
            (make_client(), 10),
            (make_client(), 10),
            (make_client(), 10)
        ]
        
        features = extract_features(client_updates, global_w)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 4) # 3 clients, 4 features
        
        # Check that features are finite (no NaN or inf)
        assert np.isfinite(features).all()

    def test_proposed_defense_filters_attackers(self):
        global_w = {
            "k1": torch.tensor([0.0]),
            "k2": torch.tensor([0.0]),
            "pl_layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "k4": torch.tensor([0.0]),
            "k5": torch.tensor([0.0]),
            "k6": torch.tensor([0.0])
        }
        
        # Benign clients (identical to global + slight noise)
        b1_pl = global_w["pl_layer"].clone()
        b2_pl = global_w["pl_layer"].clone()
        b3_pl = global_w["pl_layer"].clone()
        
        # Attacker clients (massive norm, completely different representation)
        a1_pl = torch.tensor([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])
        a2_pl = torch.tensor([[10.0, -10.0], [10.0, -10.0], [10.0, -10.0]])
        
        def make_client(pl, noise_scale=0.0):
            return {
                "k1": torch.randn(1) * noise_scale,
                "k2": torch.randn(1) * noise_scale,
                "pl_layer": pl,
                "k4": torch.randn(1) * noise_scale,
                "k5": torch.randn(1) * noise_scale,
                "k6": torch.randn(1) * noise_scale
            }
            
        client_updates = [
            (make_client(b1_pl, 0.01), 10),
            (make_client(b2_pl, 0.01), 10),
            (make_client(b3_pl, 0.01), 10),
            (make_client(a1_pl, 10.0), 10),
            (make_client(a2_pl, 10.0), 10)
        ]
        
        agg = proposed_defense(client_updates, global_weights=global_w)
        
        # The aggregated PLR should be the average of b1, b2, b3 (close to global_w)
        # It should definitely not be dominated by the 10.0 magnitude of the attackers
        assert torch.allclose(agg["pl_layer"], global_w["pl_layer"], atol=1e-1)
