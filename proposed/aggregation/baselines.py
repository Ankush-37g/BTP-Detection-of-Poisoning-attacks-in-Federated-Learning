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
import numpy as np
from sklearn.cluster import KMeans

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


def fedcvg(client_updates: List[Tuple[StateDict, int]], global_weights: StateDict = None, **kwargs) -> StateDict:
    """
    FedCVG-style Stage 1 detection baseline.
    Computes the L2 norm of the update (w_local - w_global) for each client.
    Uses K-Means (K=2) on the update norms.
    Assumes the cluster with the smaller median norm (or larger size if medians are close) is benign, 
    but a standard heuristic is to assume the larger cluster is benign.
    We will use the cluster size heuristic (larger cluster = benign), which is standard for unsupervised FL anomaly detection.
    """
    if global_weights is None:
        raise ValueError("fedcvg requires global_weights to compute update norms.")
        
    n_clients = len(client_updates)
    if n_clients <= 2:
        from proposed.aggregation.fedavg import fedavg
        return fedavg(client_updates, uniform_weights=False)
        
    # 1. Compute update norms
    flat_global = _flatten_state_dict(global_weights).float().cpu()
    norms = []
    
    for update, _ in client_updates:
        flat_local = _flatten_state_dict(update).float().cpu()
        delta_w = flat_local - flat_global
        norm = torch.norm(delta_w, p=2).item()
        norms.append([norm])  # 2D array for KMeans
        
    norms_arr = np.array(norms)
    
    # 2. KMeans Clustering (K=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(norms_arr)
    
    # 3. Identify benign cluster
    # Heuristic: the larger cluster is benign (assuming > 50% benign clients)
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    
    benign_label = 0 if count_0 >= count_1 else 1
    
    # If the clusters are equally sized, fallback to the one with the smaller mean norm
    if count_0 == count_1:
        mean_0 = np.mean(norms_arr[labels == 0])
        mean_1 = np.mean(norms_arr[labels == 1])
        benign_label = 0 if mean_0 < mean_1 else 1
        
    # 4. Filter and aggregate
    selected_updates = [client_updates[i] for i in range(n_clients) if labels[i] == benign_label]
    
    from proposed.aggregation.fedavg import fedavg
    return fedavg(selected_updates, uniform_weights=False)

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two 2D matrices.
    Using the efficient formulation: HSIC(X, Y) = ||X_c^T Y_c||_F^2
    """
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)
        
    # Center columns
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)
    
    hsic_xy = torch.norm(torch.mm(X_c.t(), Y_c), p='fro') ** 2
    hsic_xx = torch.norm(torch.mm(X_c.t(), X_c), p='fro') ** 2
    hsic_yy = torch.norm(torch.mm(Y_c.t(), Y_c), p='fro') ** 2
    
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return (hsic_xy / torch.sqrt(hsic_xx * hsic_yy)).item()


def fedcc(client_updates: List[Tuple[StateDict, int]], global_weights: StateDict = None, **kwargs) -> StateDict:
    """
    FedCC-style Stage 1 detection baseline.
    Computes Linear CKA between the client's penultimate layer weight matrix and the global model's.
    Uses K-Means (K=2) on the 1D CKA similarity scores.
    The cluster with the HIGHER mean CKA similarity to the global model is considered benign.
    """
    if global_weights is None:
        raise ValueError("fedcc requires global_weights to compute CKA similarities.")
        
    n_clients = len(client_updates)
    if n_clients <= 2:
        from proposed.aggregation.fedavg import fedavg
        return fedavg(client_updates, uniform_weights=False)
        
    # 1. Identify penultimate weight layer (typically 4th from last key in standard networks like LeNet)
    keys = list(global_weights.keys())
    penultimate_key = keys[-4]
    
    global_pl = global_weights[penultimate_key].float().cpu()
    
    # 2. Compute CKA scores
    cka_scores = []
    for update, _ in client_updates:
        local_pl = update[penultimate_key].float().cpu()
        score = linear_cka(local_pl, global_pl)
        cka_scores.append([score])
        
    cka_arr = np.array(cka_scores)
    
    # 3. KMeans Clustering (K=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cka_arr)
    
    # 4. Identify benign cluster
    mean_0 = np.mean(cka_arr[labels == 0])
    mean_1 = np.mean(cka_arr[labels == 1])
    
    benign_label = 0 if mean_0 > mean_1 else 1
    
    # 5. Filter and aggregate
    selected_updates = [client_updates[i] for i in range(n_clients) if labels[i] == benign_label]
    
    from proposed.aggregation.fedavg import fedavg
    if not selected_updates:
        selected_updates = client_updates
        
    return fedavg(selected_updates, uniform_weights=False)



