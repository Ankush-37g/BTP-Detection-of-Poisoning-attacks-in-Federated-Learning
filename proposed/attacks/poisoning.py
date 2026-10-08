"""
Poisoning attacks for Federated Learning.

This module implements untargeted and targeted model poisoning attacks.
Attacks are applied by modifying the client's local weights *after* local training,
before they are sent to the server.

Implemented attacks:
    1. Sign-flip: Inverts the direction of the client's gradient update.
                  w_poison = w_global - (w_local - w_global)
    2. Scaling:   Scales the gradient update to dominate the aggregation.
                  w_poison = w_global + scale_factor * (w_local - w_global)
    3. Noise:     Adds Gaussian noise to the local weights.

Design decisions:
    - Attacks are functions that take the locally trained weights and the
      global weights, and return poisoned weights.
    - They operate on CPU state_dicts to avoid device mismatch issues.
    - Backdoor (targeted data poisoning) is deferred or implemented separately
      if needed, as it requires modifying the client's dataset during training.
      For now, we focus on model poisoning attacks which directly perturb weights.
"""

from typing import Dict
import torch

StateDict = Dict[str, torch.Tensor]


def apply_attack(
    local_weights: StateDict,
    global_weights: StateDict,
    attack_type: str,
    **kwargs,
) -> StateDict:
    """
    Apply a poisoning attack to the local weights.

    Args:
        local_weights:  The client's weights after local training.
        global_weights: The server's global weights from the start of the round.
        attack_type:    One of "none", "sign_flip", "scaling", "noise".
        **kwargs:       Attack-specific parameters (e.g., scale_factor).

    Returns:
        Poisoned StateDict.
    """
    attack_type = attack_type.lower()
    if attack_type == "none":
        return local_weights
    
    if attack_type == "sign_flip":
        return sign_flip_attack(local_weights, global_weights)
    
    if attack_type == "scaling":
        scale_factor = kwargs.get("scale_factor", 10.0)
        return scaling_attack(local_weights, global_weights, scale_factor)
    
    if attack_type == "noise":
        std = kwargs.get("std", 0.1)
        return noise_attack(local_weights, std)
    
    raise ValueError(f"Unknown attack type: {attack_type}")


def sign_flip_attack(local_weights: StateDict, global_weights: StateDict) -> StateDict:
    """
    Invert the direction of the client's update.
    
    Δw = w_local - w_global
    w_poison = w_global - Δw = 2 * w_global - w_local
    """
    poisoned = {}
    for k in local_weights:
        # Avoid in-place operations that might affect original tensors
        poisoned[k] = 2.0 * global_weights[k].cpu() - local_weights[k].cpu()
    return poisoned


def scaling_attack(
    local_weights: StateDict, 
    global_weights: StateDict, 
    scale_factor: float
) -> StateDict:
    """
    Scale the client's update by a large factor to dominate aggregation.
    
    Δw = w_local - w_global
    w_poison = w_global + scale_factor * Δw
    """
    poisoned = {}
    for k in local_weights:
        delta = local_weights[k].cpu() - global_weights[k].cpu()
        poisoned[k] = global_weights[k].cpu() + scale_factor * delta
    return poisoned


def noise_attack(local_weights: StateDict, std: float) -> StateDict:
    """Add Gaussian noise to the local weights."""
    poisoned = {}
    for k in local_weights:
        noise = torch.randn_like(local_weights[k].cpu()) * std
        poisoned[k] = local_weights[k].cpu() + noise
    return poisoned
