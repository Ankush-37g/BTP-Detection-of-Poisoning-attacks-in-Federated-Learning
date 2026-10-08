# Phase 2 — Non-IID Data Partitioning & Analysis

**Status:** ✅ Complete  
**Date:** 2026-10-04  

---

## What This Phase Does

Phase 2 introduces and analyses **Non-IID (Non-Independent and Identically Distributed)** data partitioning — the core challenge that makes Federated Learning hard in practice.

In real FL, each client holds data from its own local distribution. A hospital might only see certain patient demographics; a phone user might only have photos of certain subjects. This label skew is modelled using a **Dirichlet(α) distribution**, where smaller α means more heterogeneous data.

Phase 2 answers:
1. **How different does client data look** across α values?
2. **Does FedAvg still converge** under Non-IID data?
3. **Why does heterogeneity matter** for poisoning detection? (Foreshadowing Phase 5–7)

---

## Why This Phase Is Needed

Without understanding the Non-IID structure:
- We can't explain why some defenses (like Krum) fail under Non-IID — benign clients with skewed data *look* like malicious clients to a naive detector.
- We can't calibrate our experiments — choosing α=0.5 requires knowing what that means visually and statistically.
- We have no baseline to compare against when attacks are introduced (Phase 3).

The **key research insight** this phase makes concrete: Under Non-IID with small α, *benign clients naturally produce diverse, dissimilar updates* — making unsupervised detection harder and more interesting.

---

## Files Created in Phase 2

### `proposed/evaluation/visualize.py`
**What:** Reusable plotting functions used throughout the project.  
**Why:** Centralising all matplotlib code avoids duplicated style settings and ensures consistent, publication-quality figures across all phases.

Functions:
| Function | Output | Use |
|---|---|---|
| `plot_client_label_distribution()` | Stacked bar chart | Shows per-client class composition |
| `plot_partition_heatmap()` | Seaborn heatmap | Shows class fraction matrix (clients × classes) |
| `plot_accuracy_curve()` | Line plot | Global accuracy vs round |
| `plot_loss_curve()` | Line plot | Loss vs round |

All functions accept a `save_path` argument and save PNG files — they **never call `plt.show()`**, so they work correctly in script/headless environments.

```python
from proposed.evaluation.visualize import plot_client_label_distribution

# class_counts = {client_id: {class_label: count}}
fig = plot_client_label_distribution(
    class_counts_per_client=class_counts,
    num_classes=10,
    alpha=0.5,
    partition_type="noniid",
    save_path="results/partitions/noniid_alpha_0_5/label_distribution.png",
)
```

---

### `experiments/run_partition_analysis.py`
**What:** Standalone script to analyse and visualize partitions for IID and multiple α values.  
**Why:** You can run this independently without running an FL experiment — useful for quickly understanding data heterogeneity before committing to a full experimental run.

```bash
python experiments/run_partition_analysis.py \
    --dataset MNIST \
    --num_clients 10 \
    --alpha 10.0 1.0 0.5 0.1 \
    --seed 42
```

**Outputs** (per partition configuration):
```
results/partitions/
├── iid/
│   ├── per_client_counts.csv
│   ├── label_distribution.png   ← stacked bar chart
│   └── heatmap.png              ← class fraction heatmap
├── noniid_alpha_10_0/
│   └── ...
├── noniid_alpha_1_0/
│   └── ...
├── noniid_alpha_0_5/
│   └── ...
├── noniid_alpha_0_1/
│   └── ...
└── summary.csv                  ← heterogeneity stats across all alpha values
```

---

### `experiments/configs/noniid.yaml`
**What:** Config for running a Non-IID FL experiment (α=0.5, 10 rounds).  

```bash
python experiments/run_fl.py --config experiments/configs/noniid.yaml
```

To explore other α values without editing the file:
```bash
python experiments/run_fl.py --config experiments/configs/noniid.yaml --alpha 0.1
python experiments/run_fl.py --config experiments/configs/noniid.yaml --alpha 1.0
```

---

### `tests/test_phase2.py`
**What:** 13 unit tests covering Non-IID properties and visualization.

```bash
python -m pytest tests/test_phase2.py -v
# Expected: 13 passed in ~4s
```

---

## How to Run Phase 2

### Activate environment
```bash
cd BTP-FL-Poisoning
source .venv/bin/activate
```

### Step 1 — Run partition analysis (generates all plots)
```bash
python experiments/run_partition_analysis.py \
    --dataset MNIST \
    --num_clients 10 \
    --alpha 10.0 1.0 0.5 0.1 \
    --seed 42
```

### Step 2 — Run Non-IID FL experiment (α=0.5, 10 rounds)
```bash
python experiments/run_fl.py --config experiments/configs/noniid.yaml
```

### Step 3 — Compare IID vs Non-IID
```bash
# IID baseline (Phase 1)
python experiments/run_fl.py --partition iid --num_rounds 10 --seed 42

# Non-IID with different alpha values
python experiments/run_fl.py --config experiments/configs/noniid.yaml --alpha 0.1 --seed 42
python experiments/run_fl.py --config experiments/configs/noniid.yaml --alpha 1.0 --seed 42
```

### Step 4 — Run unit tests
```bash
python -m pytest tests/test_phase2.py -v
```

---

## Partition Analysis Results

### Heterogeneity Summary (MNIST, 10 clients, seed=42)

| Partition | α | Min Samples | Max Samples | Std | Het. Score |
|---|:---:|---:|---:|---:|---:|
| IID | — | 6,000 | 6,000 | 0.0 | 0.0028 |
| Non-IID | 10.0 | 5,051 | 7,060 | 550.4 | 0.0236 |
| Non-IID | 1.0 | 3,782 | 8,106 | 1,308.8 | 0.0880 |
| Non-IID | 0.5 | 1,829 | 10,421 | 2,759.8 | 0.0646 |
| Non-IID | 0.1 | 409 | 16,400 | 4,510.3 | 0.1996 |

**Heterogeneity Score** = std of each client's dominant-class fraction.  
- IID: ~0.003 (all clients have ~10% of every class → max class ≈ 10% for all)
- α=0.1: ~0.200 (some clients are 80%+ one class → max class varies widely)

---

## FL Experiment Results

### Non-IID FedAvg (α=0.5, 10 clients, 10 rounds, seed=42)

| Round | Test Accuracy | Test Loss | Client Loss |
|:-----:|:-------------:|:---------:|:-----------:|
| 1 | 86.41% | 0.9886 | 0.5246 |
| 2 | 96.43% | 0.1136 | 0.1463 |
| 3 | 97.57% | 0.0730 | 0.0704 |
| 4 | 98.22% | 0.0530 | 0.0536 |
| 5 | 98.47% | 0.0462 | 0.0436 |
| 6 | 98.64% | 0.0425 | 0.0381 |
| 7 | 98.77% | 0.0381 | 0.0340 |
| 8 | 98.80% | 0.0373 | 0.0324 |
| 9 | 98.90% | 0.0344 | 0.0273 |
| 10 | **98.91%** | **0.0318** | 0.0252 |

### IID vs Non-IID Comparison (FedAvg, 5 rounds, seed=42)

| Setting | Round 1 Acc | Round 5 Acc |
|---|:---:|:---:|
| IID | 92.49% | 98.14% |
| Non-IID α=0.5 | 86.41% | 98.47% |

**Observation:** Non-IID starts ~6% lower in round 1, but catches up and even slightly exceeds IID by round 5. For MNIST with α=0.5, the heterogeneity is moderate enough that FedAvg still converges well. This **changes dramatically when attacks are introduced** (Phase 4).

---

## Key Design Decisions (BTP Viva Answers)

| Question | Answer |
|---|---|
| Why Dirichlet(α) for Non-IID? | It's the de-facto standard in FL literature (used in FedCVG, FedCC, and most ICLR/NeurIPS FL papers). It produces a continuum from IID (α→∞) to extreme heterogeneity (α→0) via one parameter. |
| Why α=0.5 as the standard? | It represents moderate, realistic heterogeneity. Many FL papers use 0.5 as their primary setting. We test 0.1, 0.5, 1.0, 10.0 for completeness. |
| What is the heterogeneity score? | Std of each client's dominant-class fraction. An IID client has ~10% per class, so max ≈ 10% for everyone (std ≈ 0). A Non-IID client with α=0.1 may have 90%+ of one class, producing high std. |
| Why doesn't Non-IID hurt FedAvg much on MNIST? | MNIST is relatively easy. The effect is more pronounced on harder datasets (CIFAR-10) or with more extreme α. More importantly, the effect is amplified under poisoning (Phase 4). |
| Why do we plot both bar charts AND heatmaps? | Bar charts show absolute sample counts (useful for understanding dataset size imbalance). Heatmaps show class fractions (useful for understanding label skew regardless of size). Both are needed for a complete picture. |
| Why is `plt.show()` never called? | The analysis runs in scripts without a display server. `matplotlib.use("Agg")` and file-saving makes it reproducible in any environment including remote servers. |

---

## Saved Plots

All plots are saved to `experiments/results/partitions/`. Key files:

| File | What It Shows |
|---|---|
| `iid/label_distribution.png` | Balanced bars — all clients have ~600 samples of each class |
| `noniid_alpha_0_1/label_distribution.png` | Highly skewed bars — clients dominated by 1–2 classes |
| `noniid_alpha_0_5/heatmap.png` | Colour intensity shows which clients "own" which classes |
| `summary.csv` | Heterogeneity statistics for all α values in one table |

---

## What Phase 3 Adds (Next)

- **Poisoning attacks:** sign-flip, scaling, and backdoor
- Malicious clients modify their updates before sending to the server
- Server receives a mix of benign and poisoned updates
- FedAvg aggregation is naive — it averages everything, including poison
- **Result:** Global model accuracy drops measurably, setting up the detection problem

---

*Phase 2 completed: 2026-10-04*
