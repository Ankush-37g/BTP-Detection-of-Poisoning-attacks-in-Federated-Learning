# Phase 6 & 7: Proposed Multi-Dimensional Feature Extraction & Robust Aggregation

## Goal
Implement the core thesis of the BTP project: A robust federated learning defense that overcomes the limitations of purely statistical defenses (like Krum/Median) and purely representational defenses (like FedCC) in non-IID settings.

## Conceptual Framework

The proposed defense identifies malicious clients through unsupervised clustering of a rich, multi-dimensional feature space.

### 1. Feature Extraction (Phase 6)
For each client update $w_{local, i}$ relative to the global model $w_{global}$, we extract a 4-dimensional feature vector encompassing both update magnitude/direction and internal semantic representation:

| Feature Dimension | Extraction Method | Purpose | Vulnerable To |
|---|---|---|---|
| **$f_1$: Update Norm** | $|| w_{local, i} - w_{global} ||_2$ | Detects aggressive magnitude attacks (e.g., Scaling). | High Non-IID variance |
| **$f_2$: Cosine Sim.** | $cos(\Delta w_i, \mu_{\Delta w})$ | Detects directional poisoning (e.g., Sign-flip). | Smart attackers matching the mean |
| **$f_3$: CKA Global** | $LinearCKA(PLR_i, PLR_{global})$ | Measures representation drift from the global model. | Attackers fine-tuning only the last layer |
| **$f_4$: CKA Mean** | $LinearCKA(PLR_i, \mu_{PLR})$ | Measures representation consensus among clients. | Colluding attackers |

*Note: $PLR$ is the penultimate layer's weight matrix (dynamically extracted).*

### 2. Standardized Clustering (Phase 7)
Because norms operate on a scale of $10^1 \sim 10^3$ while CKA operates in $[0, 1]$, raw clustering would be dominated entirely by update norms. We apply **Z-score standardization** (`StandardScaler`) across the client dimension before clustering.

We apply **K-Means Clustering (K=2)** to these standardized 4D vectors.

### 3. Unsupervised Benign Identification
Instead of hardcoding sizes or assuming the larger cluster is benign, we use a novel **Suspicion Score**:
- $Score_{cluster} = \text{Normalized Mean Norm} - \text{Normalized Mean CKA_{global}}$
- A cluster is highly suspicious if it has large update norms but low semantic similarity to the global model.
- The cluster with the lower Suspicion Score is identified as benign and selected for FedAvg aggregation.

## Implementation Details

* **File:** `proposed/aggregation/proposed.py`
* **Entry Points:** `extract_features()`, `proposed_defense()`
* **Validation:** PyTest mathematically verified the exactness of the 4D feature extraction and confirmed the KMeans cluster heuristic perfectly separates severe scaling attacks.

---

*Phase 6 & 7 completed: 2026-10-10*
