"""
Phase 3 unit tests — Poisoning Attacks.

Tests verify:
    1. Sign-flip attack correctly inverts updates.
    2. Scaling attack correctly scales updates.
    3. Noise attack adds variance.
    4. FLClient applies the attack when flagged as malicious.
    5. FLClient delta_w correctly reflects the poisoned weights.

Run with:
    cd BTP-FL-Poisoning
    .venv/bin/python -m pytest tests/test_phase3.py -v
"""

import os
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.utils.reproducibility import set_seed
from proposed.attacks.poisoning import (
    apply_attack,
    sign_flip_attack,
    scaling_attack,
    noise_attack,
)
from proposed.federated.client import FLClient
from proposed.models.lenet import SimpleCNN


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def global_and_local():
    """Returns a mock global and local state dict."""
    set_seed(42)
    global_weights = {
        "layer1.weight": torch.ones(2, 2) * 2.0,
        "layer1.bias": torch.ones(2) * 1.0,
    }
    # Local weights have moved away from global
    local_weights = {
        "layer1.weight": torch.ones(2, 2) * 3.0,  # Delta = +1.0
        "layer1.bias": torch.ones(2) * 0.5,       # Delta = -0.5
    }
    return global_weights, local_weights


# ── 1. Attack Functions ────────────────────────────────────────────────────────

class TestAttacks:

    def test_sign_flip_attack(self, global_and_local):
        g, l = global_and_local
        poisoned = sign_flip_attack(l, g)
        
        # Original delta = l - g
        # Poisoned should be g - delta = 2g - l
        expected_weight = 2.0 * 2.0 - 3.0  # = 1.0
        expected_bias = 2.0 * 1.0 - 0.5    # = 1.5
        
        assert torch.allclose(poisoned["layer1.weight"], torch.full((2, 2), expected_weight))
        assert torch.allclose(poisoned["layer1.bias"], torch.full((2,), expected_bias))

    def test_scaling_attack(self, global_and_local):
        g, l = global_and_local
        scale_factor = 10.0
        poisoned = scaling_attack(l, g, scale_factor=scale_factor)
        
        # Poisoned = g + scale_factor * (l - g)
        expected_weight = 2.0 + 10.0 * (3.0 - 2.0)  # = 12.0
        expected_bias = 1.0 + 10.0 * (0.5 - 1.0)    # = -4.0
        
        assert torch.allclose(poisoned["layer1.weight"], torch.full((2, 2), expected_weight))
        assert torch.allclose(poisoned["layer1.bias"], torch.full((2,), expected_bias))

    def test_noise_attack(self, global_and_local):
        g, l = global_and_local
        set_seed(42)
        poisoned = noise_attack(l, std=1.0)
        
        # Should not be exactly equal to local
        assert not torch.allclose(poisoned["layer1.weight"], l["layer1.weight"])

    def test_apply_attack_router(self, global_and_local):
        g, l = global_and_local
        
        # Test 'none'
        out_none = apply_attack(l, g, "none")
        assert torch.allclose(out_none["layer1.weight"], l["layer1.weight"])
        
        # Test 'sign_flip'
        out_sf = apply_attack(l, g, "sign_flip")
        assert torch.allclose(out_sf["layer1.weight"], torch.full((2, 2), 1.0))
        
        # Test 'scaling'
        out_sc = apply_attack(l, g, "scaling", scale_factor=5.0)
        # 2.0 + 5.0 * (3.0 - 2.0) = 7.0
        assert torch.allclose(out_sc["layer1.weight"], torch.full((2, 2), 7.0))


# ── 2. FLClient Integration ────────────────────────────────────────────────────

class TestFLClientAttacks:

    @pytest.fixture
    def setup_client(self):
        set_seed(42)
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))
        ds = TensorDataset(images, labels)
        model = SimpleCNN(num_classes=10)
        global_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        return ds, model, global_weights

    def test_client_clean(self, setup_client):
        ds, model, global_weights = setup_client
        client = FLClient(
            client_id=0, dataset=ds, model=model,
            is_malicious=False, attack_type="none"
        )
        local_weights, _, _, delta_w = client.train(global_weights)
        
        # delta_w should perfectly match local_weights - global_weights
        for k in local_weights:
            expected_delta = local_weights[k] - global_weights[k]
            assert torch.allclose(delta_w[k], expected_delta, atol=1e-5)

    def test_client_sign_flip(self, setup_client):
        ds, model, global_weights = setup_client
        
        # Run clean first to know what the clean local update would be
        clean_client = FLClient(0, ds, model)
        clean_weights, _, _, _ = clean_client.train(global_weights)
        
        # Run malicious
        set_seed(42)  # Reset seed so local training produces the exact same clean update
        malicious_client = FLClient(
            client_id=0, dataset=ds, model=model,
            is_malicious=True, attack_type="sign_flip"
        )
        mal_weights, _, _, delta_w = malicious_client.train(global_weights)
        
        # Verify delta_w = mal_weights - global_weights
        for k in mal_weights:
            expected_delta = mal_weights[k] - global_weights[k]
            assert torch.allclose(delta_w[k], expected_delta, atol=1e-5)
            
            # Verify the poison: mal = 2*global - clean
            expected_poison = 2.0 * global_weights[k] - clean_weights[k]
            assert torch.allclose(mal_weights[k], expected_poison, atol=1e-5)

    def test_client_scaling(self, setup_client):
        ds, model, global_weights = setup_client
        
        clean_client = FLClient(0, ds, model)
        clean_weights, _, _, _ = clean_client.train(global_weights)
        
        set_seed(42)
        malicious_client = FLClient(
            client_id=0, dataset=ds, model=model,
            is_malicious=True, attack_type="scaling",
            attack_kwargs={"scale_factor": 10.0}
        )
        mal_weights, _, _, delta_w = malicious_client.train(global_weights)
        
        for k in mal_weights:
            expected_delta = mal_weights[k] - global_weights[k]
            assert torch.allclose(delta_w[k], expected_delta, atol=1e-5)
            
            # expected_poison = global + 10 * (clean - global)
            clean_delta = clean_weights[k] - global_weights[k]
            expected_poison = global_weights[k] + 10.0 * clean_delta
            assert torch.allclose(mal_weights[k], expected_poison, atol=1e-4)
