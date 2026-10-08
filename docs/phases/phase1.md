# Phase 1 — Clean Federated Learning Simulation

**Status:** ✅ Complete  
**Date:** 2026-10-01  

---

## What This Phase Does

Phase 1 builds the **foundation of the entire project** — a clean, working Federated Learning (FL) simulation with no attacks and no detection. Everything in later phases (attacks, defenses, detection) is built on top of this.

Think of it as: *"Can 10 clients, each holding a slice of MNIST, collaboratively train a shared model using FedAvg — and does accuracy improve round by round?"*

**Answer after Phase 1: Yes. 92.5% → 98.1% in 5 rounds.**

---

## Why We Build a Clean Baseline First

Before adding attacks or defenses, we need to establish:

1. **Sanity check** — The FL training loop actually works.
2. **Reproducibility baseline** — The same seed gives the same results.
3. **Performance reference** — We know what *clean* accuracy looks like, so we can measure degradation when attacks are added.
4. **Modular foundation** — Every component (client, server, aggregator) is tested independently before being combined.

---

## Files Created in Phase 1

### `proposed/utils/device.py`
**What:** Selects the best available compute device (MPS → CUDA → CPU).  
**Why:** This machine uses Apple Silicon. MPS gives GPU-speed without CUDA. All code calls `get_device()` instead of hardcoding `'cuda'` or `'cpu'`, so the project runs on any hardware without modification.

```python
from proposed.utils.device import get_device, device_name
device = get_device()   # returns torch.device("mps") on this machine
print(device_name())    # "Apple Silicon MPS"
```

---

### `proposed/utils/reproducibility.py`
**What:** Sets seeds for Python, NumPy, and PyTorch in one call.  
**Why:** Without this, two runs with the same `--seed` can produce different results. Every experiment calls `set_seed(cfg['seed'])` as its very first step.

```python
from proposed.utils.reproducibility import set_seed
set_seed(42)   # must be called before ANY random operation
```

---

### `proposed/data/datasets.py`
**What:** Loads MNIST / FashionMNIST / CIFAR-10 with correct normalization.  
**Why:** Normalization values match both FedCVG and FedCC reference implementations, so our results are comparable. Data is auto-downloaded on first run into `data/` (git-ignored).

```python
from proposed.data.datasets import load_dataset
train_dataset, test_dataset = load_dataset("MNIST")
# train: 60,000 samples   test: 10,000 samples
```

---

### `proposed/data/partitioner.py`
**What:** Splits the training dataset across clients — IID or Non-IID (Dirichlet).  
**Why:** Federated Learning clients hold *private* data. We simulate this by giving each client a partition of the dataset. The Dirichlet parameter `alpha` controls heterogeneity (Phase 2 explores this in depth).

```python
from proposed.data.partitioner import iid_partition, dirichlet_partition, make_client_datasets

# IID — each client gets a random equal share
partition = iid_partition(train_dataset, num_clients=10)

# Non-IID — label-skewed using Dirichlet(alpha)
partition = dirichlet_partition(train_dataset, num_clients=10, alpha=0.5)

# Convert index maps to actual Subset objects
client_datasets = make_client_datasets(train_dataset, partition)
# client_datasets[0], client_datasets[1], ..., client_datasets[9]
```

**How Dirichlet partitioning works:**
- For each class c (0–9), draw proportions for 10 clients from Dirichlet(alpha).
- `alpha=10.0` → near-IID (each client sees all classes roughly equally).
- `alpha=0.5` → moderate skew (some clients are dominated by 2–3 classes).
- `alpha=0.1` → extreme skew (each client may have only 1–2 classes).

---

### `proposed/models/lenet.py`
**What:** LeNet-5 model for MNIST, with `SimpleCNN` as a lightweight alternative.  
**Why:** LeNet-5 is the standard model used in both FedCVG and FedCC for MNIST experiments. We explicitly name the penultimate layer (`fc2`, dim=84) as a class constant — this is critical for Phase 6 (CKA representation extraction).

```
Architecture (input: 1×28×28):
  conv1 (1→6, 5×5)  → ReLU → MaxPool(2×2)
  conv2 (6→16, 5×5) → ReLU → MaxPool(2×2)
  flatten (→256)
  fc1  (256→120)     → ReLU
  fc2  (120→84)      → ReLU   ← PENULTIMATE LAYER (used in Phase 6)
  fc3  (84→10)                ← CLASSIFIER HEAD
```

```python
from proposed.models.lenet import LeNet5, build_model

model = build_model("LeNet5", num_classes=10)

# Standard forward pass
logits = model(images)                       # shape: [B, 10]

# Penultimate-layer activations (used for CKA in Phase 6)
rep = model.get_penultimate_representation(images)  # shape: [B, 84]
print(LeNet5.PENULTIMATE_DIM)   # 84  ← documented as a constant
```

---

### `proposed/aggregation/fedavg.py`
**What:** FedAvg aggregation as a pure function.  
**Why:** FedAvg is the standard baseline. Implemented as a standalone function (not buried inside the server) so it can be unit-tested independently and swapped out easily in Phase 5.

**Math:**
```
w_global = Σᵢ (nᵢ / N) × wᵢ
```
where `nᵢ` = samples for client i, `N` = total samples.

```python
from proposed.aggregation.fedavg import fedavg

# client_updates = list of (state_dict, num_samples)
aggregated = fedavg(client_updates)               # weighted by sample count
aggregated = fedavg(client_updates, uniform_weights=True)  # equal weights
```

---

### `proposed/federated/client.py`
**What:** A stateless FL client — receives global weights, trains, returns results.  
**Why:** Stateless design means clients don't accumulate optimizer state between rounds they weren't selected in. This matches real FL behavior.

Each call to `client.train(global_weights)` returns:
- `local_weights` — updated model (state_dict on CPU)
- `num_samples` — how much data this client trained on
- `avg_loss` — mean training loss this round
- `delta_w` — the update: `w_local - w_global` (needed for Phase 6 feature extraction)

```python
from proposed.federated.client import FLClient

client = FLClient(
    client_id=0,
    dataset=client_datasets[0],
    model=model,
    local_epochs=1,
    batch_size=32,
    lr=0.01,
)
local_weights, num_samples, avg_loss, delta_w = client.train(global_weights)
```

---

### `proposed/federated/server.py`
**What:** FL server — manages client selection, aggregation, and evaluation.  
**Why:** The server is the central coordinator. It accepts any aggregation function via `aggregator=`, so swapping FedAvg for the proposed method in Phase 7 requires changing one argument.

```python
from proposed.federated.server import FLServer
from proposed.aggregation.fedavg import fedavg

server = FLServer(
    global_model=model,
    clients=clients,
    test_dataset=test_dataset,
    clients_per_round=10,
    aggregator=fedavg,           # swap this in Phase 5/7 for other defenses
    malicious_ids=set(),         # ground-truth labels — NEVER used for detection
)

record = server.run_round(round_idx=0)
# record = {round, test_accuracy, test_loss, avg_client_loss, selected_clients}
```

---

### `proposed/evaluation/metrics.py`
**What:** Saves per-round CSV and summary JSON for every experiment.  
**Why:** Manual result tracking is error-prone. Every run saves a `config.json` (what was run) and a `rounds.csv` (what happened), keyed by a self-describing experiment ID.

```
results/
  mnist_iid_a0.5_c10_r5_atk_none_def_fedavg_seed42/
    config.json    ← full experiment config
    rounds.csv     ← per-round: accuracy, loss, selected clients
    summary.json   ← final accuracy, best accuracy, total time
```

---

### `experiments/configs/base.yaml`
**What:** Default values for every experiment parameter.  
**Why:** No hardcoded parameters anywhere in the code. All parameters live in one place and can be overridden via CLI.

```yaml
dataset: MNIST
model: LeNet5
num_clients: 10
num_rounds: 5
partition: iid
attack: none
defense: fedavg
seed: 42
```

---

### `experiments/run_fl.py`
**What:** The single CLI entry point for running FL experiments.  
**Why:** One command runs any experiment. YAML config + CLI flags merged — CLI takes priority.

---

## How to Run Phase 1

### Activate the environment first
```bash
cd BTP-FL-Poisoning
source .venv/bin/activate
```

### Minimal experiment (uses all defaults from base.yaml)
```bash
python experiments/run_fl.py
```

### Custom experiment via CLI flags
```bash
python experiments/run_fl.py \
    --dataset MNIST \
    --model LeNet5 \
    --num_clients 10 \
    --clients_per_round 10 \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 32 \
    --partition iid \
    --attack none \
    --malicious_fraction 0.0 \
    --defense fedavg \
    --seed 42
```

### Using a config file
```bash
python experiments/run_fl.py --config experiments/configs/base.yaml
```

### Run unit tests
```bash
python -m pytest tests/test_phase1.py -v
# Expected: 25 passed in ~5 seconds
```

---

## What the Output Looks Like

```
============================================================
FEDERATED LEARNING EXPERIMENT
============================================================
  Dataset:    MNIST
  Model:      LeNet5
  Clients:    10 (per round: 10)
  Rounds:     5
  Partition:  iid
  Attack:     none  (fraction: 0.0)
  Defense:    fedavg
  Device:     Apple Silicon MPS
============================================================

[1/5] Loading dataset ...
  Train samples: 60000 | Test samples: 10000

[2/5] Partitioning data ...
  Client sample counts — min: 6000, max: 6000, mean: 6000

[3/5] Initialising model and clients ...
  Created 10 clients.

[4/5] Running 5 FL rounds ...
  Round   1 | Test Acc: 0.9249 | Test Loss: 0.2422 | Client Loss: 1.0812
  Round   2 | Test Acc: 0.9652 | Test Loss: 0.1084 | Client Loss: 0.2365
  Round   3 | Test Acc: 0.9761 | Test Loss: 0.0740 | Client Loss: 0.1481
  Round   4 | Test Acc: 0.9806 | Test Loss: 0.0609 | Client Loss: 0.1217
  Round   5 | Test Acc: 0.9814 | Test Loss: 0.0548 | Client Loss: 0.1010

[5/5] Saving results ...
Results saved to experiments/results/mnist_iid_a0.5_c10_r5_atk_none_def_fedavg_seed42/
```

---

## Verified Results (Seed=42, MNIST, IID, 10 clients, 5 rounds)

| Round | Test Accuracy | Test Loss |
|:-----:|:-------------:|:---------:|
| 1 | 92.49% | 0.2422 |
| 2 | 96.52% | 0.1084 |
| 3 | 97.61% | 0.0740 |
| 4 | 98.06% | 0.0609 |
| 5 | **98.14%** | **0.0548** |

**Interpretation:** Accuracy consistently improves. FedAvg is working correctly. This becomes the clean baseline — we expect to see degradation in Phase 4 when attacks are introduced.

---

## Key Design Decisions (BTP Viva Answers)

| Question | Answer |
|---|---|
| Why LeNet-5? | Standard model for MNIST in both reference implementations. Small enough for fast experiments, large enough to be meaningful. |
| Why is the penultimate layer explicit? | FedCC uses fragile key indexing (`keys[-4]`). We name it `fc2` and document `PENULTIMATE_DIM = 84` to avoid bugs and make Phase 6 self-documenting. |
| Why is FedAvg a pure function? | So it can be unit-tested independently (verified: 5 unit tests pass) and swapped with one argument change in the server. |
| Why is the client stateless? | Real FL clients are not selected every round. Carrying optimizer state between rounds would be unrealistic and could bias results. |
| Why is `malicious_ids` passed to the server but never to the aggregator? | **Research integrity rule.** Ground-truth labels are only for computing evaluation metrics (precision, recall, F1). The detector must never receive them as input. |
| Why `delta_w` returned from client? | Needed in Phase 6 for update norm (`‖Δwᵢ‖`) and cosine similarity computation — the two update-level features in the proposed method. |

---

## What Phase 2 Adds (Next)

- **Non-IID partitioning analysis** with multiple alpha values (0.1, 0.5, 1.0, 10.0)
- **Visualization** of client label distributions (stacked bar charts)
- **`run_partition_analysis.py`** — standalone script to inspect and save partition statistics
- **Result:** Understanding how much natural heterogeneity looks like "suspicious" updates — the core challenge that motivates the research

---

*Phase 1 completed: 2026-10-01*
