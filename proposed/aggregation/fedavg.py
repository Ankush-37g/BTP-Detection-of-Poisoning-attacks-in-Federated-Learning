"""
FedAvg aggregation.

Standard federated averaging as defined in:
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.

Aggregation rule:
    w_global = Σ_i  (n_i / N) * w_i

where n_i = number of training samples for client i and N = Σ n_i.

Design decisions:
    - Aggregation takes a list of (state_dict, num_samples) pairs.
    - It is purely a function — it does NOT modify the server model in place.
      The caller is responsible for loading the returned state_dict.
    - If uniform_weights=True, all clients are weighted equally (1/K),
      ignoring sample counts. This is useful for ablation experiments.
    - This function is intentionally kept separate from the server class
      to make it independently testable and reusable.
"""

import copy
from typing import List, Tuple, Dict, Optional

import torch


StateDict = Dict[str, torch.Tensor]


def fedavg(
    client_updates: List[Tuple[StateDict, int]],
    uniform_weights: bool = False,
) -> StateDict:
    """
    Federated Averaging aggregation.

    Args:
        client_updates:  List of (state_dict, num_samples) tuples.
                         state_dict must be on CPU for safe aggregation.
        uniform_weights: If True, weight all clients equally.
                         If False (default), weight by num_samples.

    Returns:
        Aggregated state_dict (on CPU). The caller should load this into
        the server model and move it to the appropriate device.

    Raises:
        ValueError: If client_updates is empty.
    """
    if not client_updates:
        raise ValueError("fedavg received an empty list of client updates.")

    weights = _compute_weights(client_updates, uniform_weights)
    return _weighted_average(client_updates, weights)


def _compute_weights(
    client_updates: List[Tuple[StateDict, int]],
    uniform: bool,
) -> List[float]:
    """Compute normalized aggregation weights."""
    if uniform:
        k = len(client_updates)
        return [1.0 / k] * k

    sample_counts = [n for _, n in client_updates]
    total = sum(sample_counts)
    return [n / total for n in sample_counts]


def _weighted_average(
    client_updates: List[Tuple[StateDict, int]],
    weights: List[float],
) -> StateDict:
    """Compute the weighted average of state dicts."""
    # Use the first client's state_dict as a template for key names and shapes
    template = client_updates[0][0]
    aggregated: StateDict = {}

    for key in template:
        # Accumulate weighted sum for each parameter tensor
        weighted_sum = torch.zeros_like(
            template[key].float(), device="cpu"
        )
        for (state_dict, _), weight in zip(client_updates, weights):
            param = state_dict[key].float().cpu()
            weighted_sum += weight * param

        # Keep the same dtype as the original parameter
        aggregated[key] = weighted_sum.to(template[key].dtype)

    return aggregated
