import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from proposed.aggregation.fedavg import fedavg
from proposed.aggregation.baselines import linear_cka

StateDict = Dict[str, torch.Tensor]


def _flatten_state_dict(state_dict: StateDict) -> torch.Tensor:
    """Flattens a PyTorch state_dict into a single 1D tensor."""
    return torch.cat([v.flatten() for v in state_dict.values()])


def extract_features(
    client_updates: List[Tuple[StateDict, int]], 
    global_weights: StateDict
) -> np.ndarray:
    """
    Phase 6: Proposed Feature Extraction.
    Extracts a multi-dimensional feature vector for each client update.
    
    Features:
    1. Update Norm (magnitude)
    2. Cosine Similarity to Mean Update (direction)
    3. CKA to Global Model (representation similarity)
    4. CKA to Mean Model (representation consensus)
    
    Returns:
        np.ndarray of shape (N_clients, 4)
    """
    n_clients = len(client_updates)
    
    flat_global = _flatten_state_dict(global_weights).float().cpu()
    
    # Pre-compute all client flat updates and delta_w
    flat_locals = []
    delta_ws = []
    for update, _ in client_updates:
        flat_local = _flatten_state_dict(update).float().cpu()
        flat_locals.append(flat_local)
        delta_ws.append(flat_local - flat_global)
        
    # Compute mean delta_w
    mean_delta_w = torch.stack(delta_ws).mean(dim=0)
    
    # Identify penultimate layer key for CKA
    keys = list(global_weights.keys())
    penultimate_key = keys[-4]
    global_pl = global_weights[penultimate_key].float().cpu()
    
    # Compute mean penultimate layer
    mean_pl = torch.stack([update[penultimate_key].float().cpu() for update, _ in client_updates]).mean(dim=0)
    
    features = []
    for i in range(n_clients):
        # 1. Update Norm
        norm = torch.norm(delta_ws[i], p=2).item()
        
        # 2. Cosine Similarity to Mean Update
        cos_sim = torch.nn.functional.cosine_similarity(
            delta_ws[i].unsqueeze(0), mean_delta_w.unsqueeze(0)
        ).item()
        
        # 3. CKA to Global Model
        local_pl = client_updates[i][0][penultimate_key].float().cpu()
        cka_global = linear_cka(local_pl, global_pl)
        
        # 4. CKA to Mean Model
        cka_mean = linear_cka(local_pl, mean_pl)
        
        features.append([norm, cos_sim, cka_global, cka_mean])
        
    return np.array(features)


def proposed_defense(
    client_updates: List[Tuple[StateDict, int]], 
    global_weights: StateDict = None, 
    **kwargs
) -> StateDict:
    """
    Phase 7: Proposed K-Means Detector & Robust Aggregator.
    Uses the 4D feature vectors from Phase 6, standardizes them, and applies K-Means.
    """
    if global_weights is None:
        raise ValueError("Proposed defense requires global_weights.")
        
    n_clients = len(client_updates)
    if n_clients <= 2:
        return fedavg(client_updates, uniform_weights=False)
        
    # Phase 6: Extract 4D features
    feature_matrix = extract_features(client_updates, global_weights)
    
    # Phase 7: Standardize features (Z-score normalization)
    # This is critical because Norms can be 100s while CKA is 0-1.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Phase 7: K-Means Clustering (K=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    
    # Identify Benign Cluster
    # We use a combined heuristic: the benign cluster typically has a smaller mean update norm 
    # AND a higher mean CKA to the global model.
    # In severe attacks, norms dominate. In subtle attacks, CKA dominates.
    # We will compute a simple "suspicion score" for each cluster.
    
    # Indices of features: 0=Norm, 1=CosSim, 2=CKA_Global, 3=CKA_Mean
    cluster_0_norm = np.mean(feature_matrix[labels == 0, 0])
    cluster_1_norm = np.mean(feature_matrix[labels == 1, 0])
    
    cluster_0_cka = np.mean(feature_matrix[labels == 0, 2])
    cluster_1_cka = np.mean(feature_matrix[labels == 1, 2])
    
    # Suspicion rule: High norm = bad, Low CKA = bad.
    # We normalize these two metrics across the two clusters to compare them.
    norm_sum = cluster_0_norm + cluster_1_norm + 1e-9
    cka_sum = cluster_0_cka + cluster_1_cka + 1e-9
    
    score_0 = (cluster_0_norm / norm_sum) - (cluster_0_cka / cka_sum)
    score_1 = (cluster_1_norm / norm_sum) - (cluster_1_cka / cka_sum)
    
    # The cluster with the lower suspicion score is benign
    benign_label = 0 if score_0 < score_1 else 1
    
    selected_updates = [client_updates[i] for i in range(n_clients) if labels[i] == benign_label]
    
    if not selected_updates:
        selected_updates = client_updates
        
    return fedavg(selected_updates, uniform_weights=False)
