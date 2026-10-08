# BTP-FL-Poisoning

## Representation-Based Unsupervised Detection of Poisoning Attacks in Federated Learning

**Author:** B.Tech Student, IIIT Kottayam  
**Guide:** [Advisor Name]  
**Year:** 2026  

---

## Project Objective

This project investigates whether combining **update-level features** (gradient norms, cosine similarity) with **representation-level features** (penultimate-layer activations, CKA similarity) into a unified client feature vector improves unsupervised malicious-client detection in Federated Learning, particularly under Non-IID data distributions.

**Research Question:**  
*Can combining update statistics with internal model representations improve unsupervised poisoning detection under heterogeneous (Non-IID) federated data?*

This is treated as a **hypothesis** to be validated experimentally, not a claim.

---

## Project Architecture

```
                    GLOBAL MODEL
                         |
              distribute to clients
                         |
          +--------------+--------------+
          |              |              |
       Client 1       Client 2       Client N
       Non-IID        Non-IID        Non-IID
          |              |              |
       Train          Train          Train
          |              |              |
       Update         Update         Update
          |              |              |
          +--------------+--------------+
                         |
                  FEATURE EXTRACTION
                         |
       +-----------------+------------------+
       |                                    |
 UPDATE-LEVEL                         REPRESENTATION
       |                                    |
  Update Norm                         Penultimate Layer
  Cosine Similarity                   Activations
       |                                   |
       |                                  CKA
       +----------------+-------------------+
                        |
               Combined Feature Vector
                        |
                   Normalization
                        |
                    K-Means (K=2)
                        |
             +----------+----------+
             |                     |
          Benign              Suspicious
             |                     |
       Keep contribution      Filter/Downweight
             |                     |
             +----------+----------+
                        |
                 Robust Aggregation
                        |
                 Updated Global Model
```

---

## Directory Structure

```
BTP-FL-Poisoning/
│
├── proposed/                  # Our implementation
│   ├── data/                  # Dataset loading and partitioning
│   ├── models/                # Model definitions (LeNet, CNN, etc.)
│   ├── attacks/               # Poisoning attack implementations
│   ├── features/              # Feature extraction (update + representation)
│   ├── detection/             # K-Means detector
│   ├── aggregation/           # FedAvg and robust aggregation
│   ├── federated/             # FL server and client
│   ├── evaluation/            # Metrics and logging
│   └── utils/                 # Shared utilities
│
├── experiments/
│   ├── configs/               # YAML experiment configurations
│   ├── scripts/               # Experiment runner scripts
│   └── results/               # Experiment outputs (git-ignored for raw)
│
├── tests/                     # Unit tests
├── notebooks/                 # Exploratory analysis
├── logs/                      # Experiment logs
├── docs/                      # Documentation
│   ├── repository_analysis.md # Analysis of reference implementations
│   ├── environment.md         # Environment setup and versions
│   └── baselines.md           # Baseline methods documentation
│
├── .venv/                     # Virtual environment (git-ignored)
├── requirements.txt           # Pinned dependencies
├── .gitignore
└── README.md                  # This file
```

---

## Design Principle

The implementation is split into two clearly separated layers:

| Layer | Contents | Purpose |
|---|---|---|
| **Common FL Infrastructure** | `proposed/data/`, `proposed/models/`, `proposed/federated/`, `proposed/attacks/` | Shared simulation framework |
| **Proposed Method** | `proposed/features/`, `proposed/detection/`, `proposed/aggregation/` | Core research contribution |

> Reference implementations (FedCVG, FedCC) are studied locally but are **not committed** to this repository. See [Reference Repositories](#reference-repositories) below.

---

## Implementation Phases

| Phase | Description | Status |
|---|---|---|
| **Milestone 1** | Environment setup, repo inspection | ✅ Complete |
| **Phase 1** | Clean FL simulation (FedAvg, IID) | 🔲 Pending |
| **Phase 2** | Non-IID data partitioning | 🔲 Pending |
| **Phase 3** | Poisoning attacks | 🔲 Pending |
| **Phase 4** | FedAvg baseline under poisoning | 🔲 Pending |
| **Phase 5** | Defense baselines (FedCVG, FedCC) | 🔲 Pending |
| **Phase 6** | Proposed feature extraction (update + CKA) | 🔲 Pending |
| **Phase 7** | K-Means detector + robust aggregation | 🔲 Pending |
| **Ablation** | Update-only vs CKA-only vs Combined | 🔲 Pending |
| **Evaluation** | Systematic experiments + plots | 🔲 Pending |

---

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run clean FL baseline (once Phase 1 is implemented)
python -m experiments.run \
    --dataset MNIST \
    --clients 10 \
    --rounds 5 \
    --partition iid \
    --attack none \
    --defense fedavg \
    --seed 42

# 3. Run Non-IID with poisoning (once Phase 3 is implemented)
python -m experiments.run \
    --dataset MNIST \
    --clients 10 \
    --rounds 20 \
    --partition noniid \
    --alpha 0.5 \
    --attack sign_flip \
    --malicious_fraction 0.2 \
    --defense proposed \
    --seed 42

# 4. Run ablation study (once Phase 6 is implemented)
python -m experiments.run_ablation --feature_set update
python -m experiments.run_ablation --feature_set cka
python -m experiments.run_ablation --feature_set combined
```

---

## Key Research Integrity Rules

1. **Never use ground-truth attack labels in the detector.** Labels are only for evaluation metrics.
2. **Never fabricate results.** All claims must be backed by experimental runs.
3. **Document every design decision** that could affect experimental validity.
4. **Do not silently change** the FedCVG or FedCC algorithms to make them perform better.
5. **If proposed method underperforms**, report it honestly and analyze why.

---

## Reference Repositories

These are studied as baselines but are **not included** in this repo. Clone locally into an `external/` folder (git-ignored) if needed:

```bash
mkdir -p external && cd external
git clone https://github.com/zhangsanry/FedCVG.git
git clone https://github.com/HyejunJeong/FedCC.git
```

See [`docs/repository_analysis.md`](docs/repository_analysis.md) for a full architectural breakdown of both.

---

## References

1. **FedCVG**: Zhang et al., [github.com/zhangsanry/FedCVG](https://github.com/zhangsanry/FedCVG)
2. **FedCC**: Jeong et al., [github.com/HyejunJeong/FedCC](https://github.com/HyejunJeong/FedCC)
3. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
4. **CKA**: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019
5. **Dirichlet Non-IID**: Hsieh et al., "Quagmire of Batch Normalization in Federated Learning", 2019

---

## Contact

IIIT Kottayam, B.Tech Computer Science and Engineering  
Final Year Project (BTP)
