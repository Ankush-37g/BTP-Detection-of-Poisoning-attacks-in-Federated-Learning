# Phase 4 — Baseline Robust Defenses

**Status:** ✅ Complete  
**Date:** 2026-10-10  

---

## What This Phase Does

In Phase 3, we proved that standard Federated Averaging (`FedAvg`) is completely destroyed by scaling and sign-flip attacks. To properly evaluate our proposed representation-based detector (Phases 5/6), we first need to implement the standard state-of-the-art **Robust Aggregation Baselines**. 

This phase implements four classic robust aggregation algorithms:
1. **Median:** Takes the coordinate-wise median of all client updates.
2. **Trimmed Mean:** Sorts updates coordinate-wise, removes a fraction of extreme values from both ends, and averages the rest.
3. **Krum:** Computes pairwise distances between all updates. Selects the *single* update that has the smallest sum of squared distances to its neighbors (assuming $f$ attackers).
4. **Multi-Krum:** Like Krum, but selects the top $m$ updates and averages them.

---

## Why This Phase Is Needed

We need these baselines for two reasons:
1. **Benchmark Comparison:** We must prove that our proposed CKA method actually outperforms existing statistical methods.
2. **Understanding the Failure Modes:** Traditional statistical defenses (like Median and Krum) often struggle under **Non-IID data**. Because benign clients with skewed data produce updates that look "far away" from the average, distance-based methods like Krum often mistakenly reject benign clients and select attackers (or just fail to converge well). We need to observe this failure in practice.

---

## Files Created / Modified

### `proposed/aggregation/baselines.py`
**What:** Implementation of the four robust aggregators.  
**Why:** Separates the complex mathematical logic from the server loop. All functions follow the strict signature `(client_updates, **kwargs) -> StateDict`.

### `experiments/run_fl.py`
**What:** Updated the `_get_aggregator` function to use `functools.partial`.  
**Why:** The `FLServer` simply calls `self.aggregator(client_updates)`. By using `partial`, we can pass hyperparameters like `krum_f` (number of assumed attackers) and `trim_ratio` from the CLI without changing the server's internal logic.

```python
# From run_fl.py
elif defense == "krum":
    f = cfg.get("krum_f", 1)
    return functools.partial(krum, f=f)
```

### `tests/test_phase4.py`
**What:** 9 unit tests strictly verifying the math.  
**Why:** Ensuring that Trimmed Mean actually trims the correct fraction of extremes, and that Krum actually picks the geometric median point when fed controlled mock data.

---

## How to Run Phase 4

### Activate the environment
```bash
cd BTP-FL-Poisoning
source .venv/bin/activate
```

### Run tests
```bash
python -m pytest tests/test_phase4.py -v
```

### Test Krum against the Scaling Attack
*Setup: 10 clients, Non-IID $\alpha=0.5$, 3 attackers (scaling factor=10).*  
Since there are 3 attackers, we tell Krum to expect 3 attackers (`--krum_f 3`).
```bash
python experiments/run_fl.py \
    --config experiments/configs/attack_scaling.yaml \
    --defense krum \
    --krum_f 3
```

---

## Verified Output (Krum vs. Scaling Attack)

*Setup: MNIST, 10 clients, Non-IID (α=0.5), 10 rounds, 30% malicious clients (Scaling Attack 10x).*

| Scenario | Round 10 Accuracy | Analysis |
|---|:---:|---|
| **Clean Baseline (FedAvg, No Attack)** | 98.89% | Optimal convergence when all clients are benign. |
| **FedAvg + Scaling Attack** | 09.82% | Total collapse. The attacker dominates the mean. |
| **Krum + Scaling Attack** | **82.91%** | **Partial success.** Krum correctly identifies and filters out the attackers (preventing the 9% crash). However, because the data is Non-IID, selecting only a *single* benign client per round discards too much valuable data, preventing the model from reaching the optimal 98% accuracy. |

**Conclusion:** State-of-the-art baselines like Krum can prevent total catastrophic failure, but their performance heavily degrades under Non-IID settings because they discard too much benign diversity.

---

## Key Design Decisions (BTP Viva Answers)

| Question | Answer |
|---|---|
| Why does Krum require an `f` parameter? | Krum requires a prior assumption of how many Byzantine (malicious) attackers exist. It uses this to calculate distances to the $N - f - 2$ nearest neighbors. If $f$ is guessed incorrectly, Krum's performance degrades. Our proposed method (Phase 6) aims to be unsupervised and not require a strict prior on $f$. |
| Why is Multi-Krum better than Krum? | Krum selects a single update as the new global model, which throws away all the useful data learned by the other $N - f - 1$ benign clients. Multi-Krum mitigates this by averaging the top $m$ safest updates. |
| Why do Median and Trimmed Mean operate "coordinate-wise"? | They don't look at the update as a single high-dimensional vector. Instead, for every single weight in the neural network (e.g., layer 1, node 3 bias), they calculate the median across all clients independently. This makes them robust against extreme outliers on specific weights, but they can destroy the internal structural relationships between layers. |

---

## What Phase 5 Adds (Next)

[Brief preview of the next phase: Re-implementing the FedCVG and FedCC state-of-the-art baselines so we have clustering-based defenses to compare against.]

---

*Phase 4 completed: 2026-10-10*
