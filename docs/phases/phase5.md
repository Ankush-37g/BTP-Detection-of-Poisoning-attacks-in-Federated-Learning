# Phase 5: Defense Baselines (FedCVG & FedCC)

## Goal
Implement the state-of-the-art clustering-based robust aggregation methods from the reference papers to serve as baselines for our proposed CKA-based defense.

## References
1. **FedCVG (Federated Client Verification and Gradient Memory):** Uses K-Means clustering on the L2 norm (magnitude) of client updates to filter outliers.
2. **FedCC (Federated Contrastive Clustering):** Uses Centered Kernel Alignment (CKA) similarity on the client's penultimate layer representations, followed by K-Means clustering to filter dissimilar clients.

## Implementation Details

The implementation is located in `proposed/aggregation/baselines.py`.

### 1. Architectural Changes
Both FedCVG and FedCC are **stateful** aggregators; they require knowledge of the `global_model` to compute update magnitudes ($w_{local} - w_{global}$) and similarities.
- We updated `proposed/federated/server.py` to pass `global_weights` to the aggregator.
- We updated `fedavg` to accept `**kwargs` to prevent crashing.

### 2. FedCVG (Update Norm Clustering)
- **Feature:** Computes the L2 norm of the flattened update vector for each client: `||w_local - w_global||_2`.
- **Detection:** Uses `sklearn.cluster.KMeans` (K=2) on the 1D norm values.
- **Filtering:** Adopts the standard unsupervised assumption that the cluster containing the majority of clients is the benign cluster (since attackers typically constitute $<50\%$ of the network). Only the benign cluster is aggregated using FedAvg.

### 3. FedCC (CKA Similarity Clustering)
- **Feature:** Extracts the penultimate layer's weight matrix (dynamically identified as `keys[-4]`) for each client and the global model.
- **Similarity:** Computes Linear Centered Kernel Alignment (CKA) between the client's penultimate matrix and the global penultimate matrix.
    - *Mathematical efficiency:* $HSIC(X, Y) = ||X_c^T Y_c||_F^2$
- **Detection:** Uses K-Means (K=2) on the 1D CKA scores.
- **Filtering:** The cluster with the *higher* mean CKA similarity to the global model is identified as benign, as benign clients naturally maintain representations closer to the global consensus.

### 4. Mathematical Rigor & Testing
Implemented a mathematically exact Linear CKA function in PyTorch that relies purely on Frobenius norms of centered matrices. We verified this with strict PyTest unit tests (`tests/test_phase5.py`), asserting that:
1. Identical matrices yield CKA = 1.0.
2. Orthogonal features yield CKA ~ 0.0.
3. K-Means correctly identifies and filters attackers injecting random noise or massive scaling attacks.

---

## Why these baselines fail in Non-IID settings (The BTP Thesis)

- **FedCVG (Norms):** In a highly Non-IID setting (like Dirichlet $\alpha=0.1$), benign clients with unique local data distributions will naturally produce very large update norms because they are pulling the model in drastically different directions. FedCVG's K-Means will misclassify these valuable benign clients as "attackers" and discard them.
- **FedCC (CKA):** While CKA captures structural representation, relying *solely* on it (a 1D score) makes the clustering vulnerable to edge cases where attackers perfectly mimic the global representation while poisoning the classifier layer.

Our proposed Phase 6/7 defense will combine both update-level and representation-level features into a multi-dimensional feature vector, achieving state-of-the-art robustness in severe Non-IID conditions.

---

*Phase 5 completed: 2026-10-10*
