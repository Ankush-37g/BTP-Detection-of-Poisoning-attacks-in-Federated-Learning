"""
Baseline robust aggregation methods.

Implemented defenses:
    1. Median: Coordinate-wise median.
    2. Trimmed Mean: Coordinate-wise trimmed mean.
    3. Krum: Selects the single update minimizing distance to neighbors.
    4. Multi-Krum: Selects m updates using Krum scoring and averages them.

Reference for Krum/Multi-Krum:
    Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NIPS 2017.
Reference for Median/Trimmed Mean:
    Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018.
"""

from typing import List, Tuple, Dict, Any
import torch

StateDict = Dict[str, torch.Tensor]


def _flatten_state_dict(state_dict: StateDict) -> torch.Tensor:
    """Flatten a state_dict into a single 1D tensor."""
    tensors = [t.view(-1) for t in state_dict.values()]
    return torch.cat(tensors)


def median(client_updates: List[Tuple[StateDict, int]], **kwargs) -> StateDict:
    """
    Coordinate-wise median aggregation.
    Ignores sample counts.
    """
    if not client_updates:
        raise ValueError("Empty client updates list.")

    template = client_updates[0][0]
    aggregated: StateDict = {}

    for key in template:
        # Stack parameter tensors along a new dimension (dim=0)
        stacked = torch.stack([update[0][key].float().cpu() for update in client_updates], dim=0)
        # Compute median
        med_val, _ = torch.median(stacked, dim=0)
        aggregated[key] = med_val.to(template[key].dtype)

    return aggregated


def trimmed_mean(client_updates: List[Tuple[StateDict, int]], trim_ratio: float = 0.1, **kwargs) -> StateDict:
    """
    Coordinate-wise trimmed mean aggregation.
    
    Args:
        trim_ratio: Fraction of extreme values to trim from BOTH ends (top and bottom).
                    e.g., trim_ratio=0.1 means trim 10% highest and 10% lowest.
    """
    n_clients = len(client_updates)
    if not client_updates:
        raise ValueError("Empty client updates list.")

    trim_count = int(trim_ratio * n_clients)
    if trim_count * 2 >= n_clients:
        raise ValueError(f"trim_ratio {trim_ratio} is too large for {n_clients} clients (would trim everything).")

    template = client_updates[0][0]
    aggregated: StateDict = {}

    for key in template:
        stacked = torch.stack([update[0][key].float().cpu() for update in client_updates], dim=0)
        # Sort along client dimension
        sorted_stacked, _ = torch.sort(stacked, dim=0)
        # Slice to remove trimmed elements
        if trim_count > 0:
            trimmed = sorted_stacked[trim_count:-trim_count]
        else:
            trimmed = sorted_stacked
        # Average the remaining
        mean_val = torch.mean(trimmed, dim=0)
        aggregated[key] = mean_val.to(template[key].dtype)

    return aggregated


def krum(client_updates: List[Tuple[StateDict, int]], f: int = 1, **kwargs) -> StateDict:
    """
    Standard Krum aggregation.
    Selects the single client update with the lowest sum of squared distances 
    to its N - f - 2 nearest neighbors.
    
    Args:
        f: Number of assumed Byzantine attackers.
    """
    return multi_krum(client_updates, f=f, m=1, **kwargs)


def multi_krum(client_updates: List[Tuple[StateDict, int]], f: int = 1, m: int = 1, **kwargs) -> StateDict:
    """
    Multi-Krum aggregation.
    Scores clients using Krum scoring, selects the top m, and averages them (FedAvg style).
    
    Args:
        f: Number of assumed Byzantine attackers.
        m: Number of updates to select and average. (If m=1, this is equivalent to standard Krum).
    """
    n_clients = len(client_updates)
    if n_clients == 0:
        raise ValueError("Empty client updates list.")
    
    if n_clients - f - 2 < 0:
        raise ValueError(f"Too many assumed attackers (f={f}) for {n_clients} clients. Need N - f - 2 >= 0.")
    
    if m > n_clients:
        raise ValueError(f"Cannot select {m} updates from {n_clients} total clients.")

    num_neighbors = n_clients - f - 2

    # Flatten updates for distance calculation
    flat_updates = torch.stack([_flatten_state_dict(update[0]).float().cpu() for update in client_updates])
    
    # Compute pairwise squared Euclidean distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    sq_norms = torch.sum(flat_updates ** 2, dim=1, keepdim=True)
    distances = sq_norms + sq_norms.T - 2 * torch.mm(flat_updates, flat_updates.T)
    # Clamp to avoid very small negative values due to numerical instability
    distances = torch.clamp(distances, min=0.0)
    
    # Set self-distance to infinity to ignore it when finding nearest neighbors
    distances.fill_diagonal_(float('inf'))
    
    scores = []
    for i in range(n_clients):
        # Sort distances to neighbors
        sorted_dists, _ = torch.sort(distances[i])
        # Sum the (N - f - 2) smallest distances (which correspond to the nearest neighbors)
        score = torch.sum(sorted_dists[:num_neighbors]).item()
        scores.append(score)
        
    # Get indices of the m lowest scores
    scores_tensor = torch.tensor(scores)
    _, top_m_indices = torch.topk(scores_tensor, k=m, largest=False)
    
    # Select the top m client updates
    selected_updates = [client_updates[idx] for idx in top_m_indices]
    
    # Average them using standard FedAvg logic
    from proposed.aggregation.fedavg import fedavg
    # For standard Krum (m=1), FedAvg just returns that single client's weights
    return fedavg(selected_updates, uniform_weights=False)

