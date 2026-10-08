# Repository Analysis: FedCVG and FedCC

**Project:** Representation-Based Unsupervised Detection of Poisoning Attacks in Federated Learning  
**Author:** BTP Student, IIIT Kottayam  
**Date:** 2026-09-30  
**Purpose:** Understanding the architecture, design decisions, and implementation of the two reference repositories before proposing the combined method.

---

## 1. FedCVG

**Repository:** https://github.com/zhangsanry/FedCVG  
**Full name:** Federated Client Verification and Gradient Memory  
**Language:** Python (PyTorch)

### 1.1 High-Level Architecture

FedCVG is a **two-stage robust federated learning algorithm**:

- **Stage 1 (Detection/Filtering):** Uses reputation mechanisms and K-Means clustering to identify and exclude malicious clients from aggregation. This stage typically uses SCAFFOLD or FedProx internally for client training.
- **Stage 2 (Virtual Aggregation):** Decouples feature extraction from classifier training. Records historical client gradients (with time-decay) to simulate "virtual participation" of unselected clients.

The two-stage design ensures that:
1. Malicious clients are identified before they can corrupt the global model.
2. Gradient history from non-selected rounds is reused (gradient memory) to improve convergence under data heterogeneity.

### 1.2 Directory Structure

```
FedCVG/
├── fedcvg.py                   # FedCVG (FedHyb) algorithm implementation
├── federated_learning.py       # Core FL framework (FedAvg, SCAFFOLD, Auror, MSGuard, etc.)
├── federated_parser.py         # CLI argument parser
├── main.py                     # Entry point for standard FL algorithms
├── main_fedcvg.py              # Entry point for FedCVG/FedHyb algorithm
├── models/
│   ├── lenet.py                # LeNet-5 (MNIST, FashionMNIST)
│   ├── cnn.py                  # CNN (CIFAR-10)
│   ├── resnet.py               # ResNet18/34/50
│   └── alexnet.py              # AlexNet
├── preprocessing/
│   └── baselines_dataloader.py # Dataset loading + IID/Non-IID partitioning
├── results/                    # Pre-saved experimental result directories
├── visualization/              # Plotting and result analysis scripts
└── requirements.txt            # Dependencies
```

### 1.3 Entry Points

| Entry Point | Purpose |
|---|---|
| `main.py` | Runs FedAvg, SCAFFOLD, FedProx, Krum, Median, TrimmedMean, Auror, MSGuard |
| `main_fedcvg.py` | Runs FedCVG (two-stage) / FedHyb |

**Example command:**
```bash
python main.py --dataset MNIST --model LeNet --num_client 10 --num_local_class 2 --num_round 100
python main_fedcvg.py --dataset CIFAR10 --model ResNet34 --num_client 20 --alpha 0.5 --num_round 100
```

### 1.4 Dataset Handling

File: `preprocessing/baselines_dataloader.py`

- Supports: MNIST, EMNIST, FashionMNIST, CIFAR10, CIFAR100, CelebA, QMNIST, IMAGENET
- IID: Uniform random partition across clients
- Non-IID: **Dirichlet distribution** with parameter `alpha`
  - Smaller `alpha` → more heterogeneous
  - e.g., `alpha=0.1` produces highly skewed distributions per client

The `divide_data()` function returns a `trainset_config` dictionary:
```python
{
    "users": [client_id_0, client_id_1, ...],
    "user_data": {
        client_id_0: Subset(trainset),
        ...
    }
}
```

### 1.5 Client Training

Located in `FederatedLearning.client_update()` in `federated_learning.py`.

Each client receives the global model, performs local training for `local_epoch` steps, and returns:
- Updated model weights (`state_dict`)
- Number of local training samples
- Average local training loss

The client update is essentially: `Δw_i = w_i^local - w_global`

Different algorithms (FedProx, SCAFFOLD, FedNova) modify the local objective, but the structure is the same.

### 1.6 Server Aggregation

Standard FedAvg:
```
w_global = Σ (n_i / N) * w_i
```
where `n_i` = number of samples for client `i`, `N` = total samples.

Additional aggregation rules supported: Krum, MultiKrum, Median, TrimmedMean.

### 1.7 Attack Implementation

Attacks are integrated inside `client_update()` based on the `attack_type` argument. The attack is applied to the client's local update (gradient/model) before submission.

Attack types observed in the codebase:
- `sign_flip`: Negates the gradient direction
- `gaussian`: Adds Gaussian noise to the update
- `targeted`: Targeted misclassification attack (backdoor-type)

The attack is applied to a fixed fraction (`attack_ratio`) of randomly selected clients.

**Important limitation:** The client itself decides whether to attack based on its client ID being in the "malicious set." This is known at the server level only for evaluation purposes.

### 1.8 Malicious Client Detection — Auror

Located in `FederatedLearning` class, method `detect_malicious_clients()`.

**Auror-style detection:**
1. Flatten all client model updates to 1D vectors.
2. Dimensionality reduction (TruncatedSVD).
3. K-Means clustering (`n_clusters` configurable).
4. Identify malicious clusters based on:
   - Cluster size ratio (small clusters = suspicious)
   - Inter-cluster distance (outlier clusters = suspicious)

**MSGuard detection** (also implemented):
1. Norm filtering: Remove clients with update norms outside `[lower_bound, upper_bound]`
2. Cosine similarity features: Compute median cosine similarity per client
3. Spectral features: SVD-based sign consistency analysis
4. Combined feature vector: `[sign_features, cosine_features, spectral_features]`
5. MeanShift clustering on the combined feature vector
6. Largest cluster = trusted; others = malicious

### 1.9 FedCVG Detection (FedHyb)

Located in `fedcvg.py`, class `FedHyb`.

**Stage 1 detection:**
- K-Means on client gradient updates (raw or reduced)
- Reputation mechanism: Clients repeatedly detected as malicious are blacklisted
- Blacklisted clients excluded from Stage 2 gradient memory

**Reputation update:**
```
reputation[client_id] -= penalty  # if detected as malicious
reputation[client_id] += bonus    # if consistently classified as benign
clients below reputation_threshold → blacklist
```

**Gradient memory (Stage 2):**
```
effective_gradient[t] = Σ decay^(t - t_last) * gradient[client]
```
where `decay` is a configurable factor (e.g., 0.9).

### 1.10 What Can Be Conceptually Reused from FedCVG

| Component | Reuse Decision |
|---|---|
| K-Means detector structure | Conceptually reuse — apply to our combined feature vector |
| Reputation mechanism | Optional — not in our Phase 1 proposal |
| Gradient memory (virtual aggregation) | Not directly reused — out of scope for Phase 1 |
| Dirichlet data partitioning | Reuse the algorithm independently |
| FedAvg aggregation | Re-implement from scratch (clean) |
| LeNet/CNN model definitions | Reuse architecture idea, implement independently |
| Attack interface structure | Reuse concept, clean re-implementation |

> **NOTE:** The FedCVG codebase is primarily in Chinese (comments). For our implementation, all code will be in English. We do NOT copy code verbatim from FedCVG.

---

## 2. FedCC

**Repository:** https://github.com/HyejunJeong/FedCC  
**Full name:** Federated Contrastive Clustering (or Federated Client Clustering via CKA)  
**Language:** Python (PyTorch)

### 2.1 High-Level Architecture

FedCC uses **penultimate-layer representations (PLR)** to compute CKA similarity between client models and the global model. Clients whose representations are dissimilar to the global model's representations are considered potentially malicious.

**Key idea:** The penultimate layer captures high-level feature representations. Malicious clients trained on poisoned data will produce different internal representations than benign clients.

### 2.2 Directory Structure

```
FedCC/
└── src/
    ├── aggregate.py        # All aggregation functions including FedCC
    ├── attacks.py          # Attack implementations (Fang, Sign-flip, Gaussian, DBA)
    ├── cka.py              # CKA implementation (linear and kernel)
    ├── models.py           # CNN, LeNet5, Alexnet, GoogleNet model definitions
    ├── options.py          # Argument parser
    ├── sampling.py         # IID and Non-IID (Dirichlet) data partitioning
    ├── update.py           # Client update (local training) and evaluation
    ├── utils.py            # Dataset loading, utility functions
    └── prelim_results_n_vis.ipynb  # Jupyter notebook for visualization
```

### 2.3 Entry Point

The main experiment appears to be run from `prelim_results_n_vis.ipynb`. There is **no standalone `main.py`** — this is a significant compatibility consideration.

**Important Compatibility Issue:**
- FedCC's `aggregate.py` hardcodes `cuda:0` in multiple places (e.g., `torch.tensor(...).to('cuda:0')`)
- This makes it **incompatible with CPU-only or MPS environments** without modification
- We will document this and use our own implementation of the FedCC concept

### 2.4 Dataset Handling

File: `src/sampling.py` and `src/utils.py`

- Supports: MNIST, CIFAR-10, CIFAR-100, FashionMNIST
- IID: `mnist_iid()`, `cifar_iid()` — uniform random partition
- Non-IID (Dirichlet): `mnist_noniid()`, `cifar_noniid()`
  - Uses: `np.random.dirichlet([alpha]*num_users, num_classes)`
  - Same mathematical approach as FedCVG

### 2.5 Client Training

File: `src/update.py`

Class `LocalUpdate`:
- Takes: global model, local dataset indices, args
- Trains locally for `local_ep` epochs
- Returns: updated `state_dict`, number of samples, loss

For malicious clients, training uses a poisoned dataset via:
```python
class PoisonedDataset(Dataset):
    # Returns poisoned (image, target_label) pairs
```

### 2.6 CKA Implementation

File: `src/cka.py`

FedCC implements both **linear CKA** and **kernel (RBF) CKA**:

**Linear CKA:**
```
Linear_CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
```
where `HSIC(X, Y) = tr(K_X_c * K_Y_c)` and `K_c = HKH` (centered kernel matrix).

**Kernel CKA (RBF):**
```
kernel_CKA(X, Y) = kernel_HSIC(X, Y) / sqrt(kernel_HSIC(X,X) * kernel_HSIC(Y,Y))
```
where the kernel is computed with `rbf(X, sigma)`:
- Sigma is computed from the **median heuristic**: `sigma = sqrt(median(pairwise_distances^2))`

**Input to CKA:**
- X, Y: matrices of shape `[n_samples, feature_dim]`
- These are **penultimate-layer activations** obtained by forward-passing a reference dataset through the model

### 2.7 Penultimate Layer Representation

In `src/aggregate.py`, function `fed_cc()`:

```python
keys = list(weights.keys())
plr = weights[keys[-4]].detach().cpu().numpy()  # keys[-4] = penultimate weight
```

**IMPORTANT:** FedCC does NOT obtain representations by forward-passing data through the model. Instead, it uses the **penultimate layer's weight matrix directly** from `state_dict()`.

This is a design choice in FedCC. Our proposed method will:
- Extract penultimate-layer **activations** (forward pass with reference data) as a more principled approach
- Document this difference explicitly

### 2.8 FedCC Aggregation (`fed_cc()`)

Steps:
1. Extract PLR from each client's model weights (`keys[-4]`)
2. Compute CKA similarity between each client's PLR and the **global model's PLR**
3. Apply K-Means (K=2) on similarity values
4. Larger cluster = benign; smaller cluster = potentially malicious
5. **Layer-wise aggregation:**
   - All layers except last two: weight by global CKA similarity
   - Penultimate layer (second-to-last): aggregate only from larger (benign) cluster
   - Last classification layer: all clients, weighted by global CKA similarity

**CRITICAL COMPATIBILITY ISSUE:** `fed_cc()` ends with:
```python
aggregated_flattened_weights_tensor = aggregated_flattened_weights.to('cuda:0')
```
This will crash on CPU/MPS without modification.

### 2.9 Attacks

File: `src/attacks.py`

- `get_malicious_updates_untargeted_mkrum()`: Fang attack (optimized to fool Multi-Krum)
- `get_malicious_updates_untargeted_med()`: Fang attack (optimized to fool Median)
- Gaussian noise attack
- Sign-flip attack
- DBA (Distributed Backdoor Attack) — multiple clients each add partial triggers

### 2.10 Other Aggregations in FedCC

| Method | Key Idea |
|---|---|
| `bulyan()` | Byzantine-tolerant; iterative Krum + coordinate-wise trimmed mean |
| `multi_krum()` | Multi-Krum: select `m - f` closest-neighbor updates |
| `krum()` | Select single best update with smallest sum-of-distances |
| `trimmed_mean()` | Coordinate-wise trimmed mean |
| `fltrust()` | Trust score = cosine similarity with global update; normalize and aggregate |
| `flare()` | MMD-based PLR similarity; neighbor frequency weighting |
| `fed_cc()` | CKA-based PLR similarity + K-Means + layer-wise aggregation |

### 2.11 What Can Be Conceptually Reused from FedCC

| Component | Reuse Decision |
|---|---|
| CKA algorithm (cka.py) | Reuse the algorithm; reimplement with CPU/MPS compatibility |
| Penultimate-layer concept | Reuse concept; use activation-based PLR (not raw weights) |
| K-Means on similarity scores | Reuse concept; extend to multi-feature vector |
| Dataset partitioning | Reuse algorithm (independent reimplementation) |
| Attack structures | Reuse concept (sign-flip, gaussian, targeted); clean reimplementation |

---

## 3. Comparison Summary

| Aspect | FedCVG | FedCC |
|---|---|---|
| Feature type | Update-level (gradient vectors, cosine similarity, norms) | Representation-level (PLR weights, CKA similarity) |
| Detection method | K-Means on update features; Reputation tracking | K-Means on 1D CKA similarity score |
| Aggregation strategy | Exclude detected malicious; gradient memory for Phase 2 | Layer-wise weighted aggregation; benign cluster only for PLR layers |
| Non-IID handling | Two-stage + SCAFFOLD/FedProx + reputation | Not explicitly designed for Non-IID |
| Datasets | MNIST, FashionMNIST, CIFAR-10 | MNIST, CIFAR-10, CIFAR-100, FashionMNIST |
| Entry point | `main.py` / `main_fedcvg.py` | Jupyter Notebook only |
| GPU dependency | CPU fallback available | Hardcoded `cuda:0` — NOT CPU/MPS compatible |
| Code language | Chinese comments | English |

---

## 4. Key Insights for the Proposed Method

### 4.1 Gap Identified

- **FedCVG** uses update-level features (norms, directions) → Works well for severe attacks but may misidentify heterogeneous benign clients in Non-IID.
- **FedCC** uses representation-level features (CKA on PLR) → Captures semantic dissimilarity but uses only a 1D similarity score.
- **Neither** combines both feature types into a joint feature vector.

### 4.2 Proposed Feature Vector (Each Client, Each Round)

```
F_i = [
    update_norm,         # ||w_i - w_global||
    cosine_similarity,   # cos(Δw_i, Δw_global)
    cka_vs_global,       # kernel_CKA(activation_i, activation_global)
    cka_vs_mean,         # kernel_CKA(activation_i, mean_activation)
]
```

### 4.3 Critical Design Decisions (Documented Here)

1. **PLR = activation, not weight**: We use a reference dataset (subset of test data) and forward-pass through each client's model to extract penultimate-layer activations. This is more principled than using raw weight values.

2. **CKA sigma**: Median heuristic (same as FedCC) for reproducibility.

3. **K-Means cluster identification**: We do NOT use ground-truth labels. The cluster with the lower mean CKA similarity to the global model is identified as suspicious (this is a principled, unsupervised criterion).

4. **FedCC PLR definition**: FedCC uses `state_dict()[keys[-4]]` (4th from last key). Our proposed method will use the layer explicitly named during model construction, making it model-architecture-aware rather than key-index-dependent.

---

## 5. Compatibility Issues Documented

| Issue | Repo | Description | Resolution |
|---|---|---|---|
| `to('cuda:0')` hardcoded | FedCC `aggregate.py` | Will crash on CPU/MPS | Use our own device-agnostic implementation |
| No `main.py` | FedCC | Only Jupyter notebook entry point | For FedCC baseline, we wrap it in a script |
| Chinese comments | FedCVG | All internal documentation in Chinese | No code change; documentation is ours |
| `batch_norm` deprecated API (potential) | Both | Some PyTorch deprecated calls possible | Check during installation; document any fixes |
| Python 3.14 compatibility | Both | Both repos target Python 3.6+ | PyTorch 2.x supports Python 3.14 |

---

## 6. Files Relevant to Our Proposed Implementation

| FedCC file | What to extract conceptually |
|---|---|
| `cka.py` | `linear_CKA()`, `kernel_CKA()`, `rbf()`, `centering()` |
| `sampling.py` | `mnist_noniid()`, `cifar_noniid()` Dirichlet logic |
| `models.py` | Architecture ideas (CNN, LeNet5) |
| `attacks.py` | Sign-flip, Gaussian structures |

| FedCVG file | What to extract conceptually |
|---|---|
| `federated_learning.py` | FedAvg aggregation, client update structure |
| `preprocessing/baselines_dataloader.py` | Dataset loading, Dirichlet partitioning |
| `models/lenet.py` | LeNet-5 architecture |

---

*This document was created during the initial inspection phase (Milestone 1) and will be updated as implementation progresses.*
