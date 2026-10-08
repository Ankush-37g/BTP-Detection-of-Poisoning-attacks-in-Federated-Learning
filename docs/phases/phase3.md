# Phase 3 — Model Poisoning Attacks

**Status:** ✅ Complete  
**Date:** 2026-10-04  

---

## What This Phase Does

This phase implements **Model Poisoning Attacks** against the federated learning simulation. Instead of just simulating benign clients training on Non-IID data (Phase 2), we now introduce malicious clients that deliberately try to destroy the global model's performance.

We implemented three untargeted attacks:
1. **Sign-Flip Attack:** Inverts the direction of the client's update. Instead of moving the global model towards lower loss on its local data, the client moves it towards higher loss.
2. **Scaling Attack:** Scales up the client's legitimate update by a huge factor (e.g., 10x). This allows the malicious client to overwhelm the updates from benign clients during aggregation.
3. **Noise Attack:** Adds Gaussian noise to the client's local weights, corrupting them.

By demonstrating that these attacks successfully crash the accuracy of standard FedAvg, we establish the **Research Gap**: Naive aggregation methods are highly vulnerable to poisoning, necessitating robust defense mechanisms (Phase 5).

---

## Why This Phase Is Needed

To evaluate any poisoning *defense*, we first need effective poisoning *attacks*.
- We must prove that our FL framework is actually vulnerable.
- We need a way to generate poisoned updates so that in Phase 6, we can analyze the representations of both benign and poisoned updates to build our detector.
- Simulating attacks under the **Non-IID setting** (α=0.5) is critical because separating poisoned updates from naturally heterogeneous benign updates is the core difficulty in FL anomaly detection.

---

## Files Created / Modified

### `proposed/attacks/poisoning.py`
**What:** Core attack functions (`sign_flip_attack`, `scaling_attack`, `noise_attack`).  
**Why:** Separates attack logic from client logic. The `apply_attack()` router function takes clean local weights and global weights, returning poisoned weights.

```python
# Sign-flip attack formulation
delta_w = w_local - w_global
w_poisoned = w_global - delta_w = 2 * w_global - w_local

# Scaling attack formulation
w_poisoned = w_global + scale_factor * (w_local - w_global)
```

### `proposed/federated/client.py`
**What:** Modified to accept `is_malicious`, `attack_type`, and `attack_kwargs`.  
**Why:** If a client is flagged as malicious by the server, it applies the specified attack to its weights *after* local training and *before* computing the final `delta_w` sent to the server.

### `experiments/run_fl.py`
**What:** Updated to parse `--attack`, `--malicious_fraction`, and `--scale_factor`.  
**Why:** Allows dynamically turning on attacks from the CLI and assigning a subset of clients (e.g., 30%) as malicious.

### `experiments/configs/attack_sign_flip.yaml` & `attack_scaling.yaml`
**What:** Baseline experiment configurations for testing the impact of attacks under Non-IID (α=0.5) conditions with a 30% malicious fraction.

### `tests/test_phase3.py`
**What:** Unit tests for the attack mathematical formulations and FLClient integration.  

---

## How to Run Phase 3

### Activate the environment
```bash
cd BTP-FL-Poisoning
source .venv/bin/activate
```

### Run Sign-Flip Attack (30% malicious, Non-IID α=0.5)
```bash
python experiments/run_fl.py --config experiments/configs/attack_sign_flip.yaml
```

### Run Scaling Attack (30% malicious, factor=10, Non-IID α=0.5)
```bash
python experiments/run_fl.py --config experiments/configs/attack_scaling.yaml
```

### Run tests
```bash
python -m pytest tests/test_phase3.py -v
```

---

## Verified Results (Impact of Attacks)

*All experiments run on MNIST, 10 clients, Non-IID (α=0.5), 10 rounds, Seed=42, 30% malicious clients.*

| Scenario | Round 10 Accuracy | Impact |
|---|:---:|---|
| **Clean Baseline (No Attack)** | 98.89% | — |
| **Sign-Flip Attack** | 90.33% | Drops ~8.5%. The model fights the poison but converges to a degraded state. |
| **Scaling Attack (10x)** | **09.82%** | **Catastrophic failure.** The model performs worse than random guessing. |

**Conclusion:** FedAvg is highly vulnerable. Even a simple scaling attack by 30% of clients can completely destroy the global model. This establishes the clear need for robust aggregation and anomaly detection.

---

## Key Design Decisions (BTP Viva Answers)

| Question | Answer |
|---|---|
| Why are attacks applied *after* local training? | In model poisoning, the attacker typically has full control over the weights sent to the server. By computing the legitimate update first, the attacker can precisely craft a payload (like sign-flip) that maximizes damage based on the actual local gradient. |
| Why is `delta_w` recomputed after the attack? | The server receives the client's final weights. Our framework uses `delta_w` as the feature for update-norm calculations in Phase 6. Therefore, `delta_w` must represent the *poisoned* update (w_poisoned - w_global), not the clean one. |
| Why test attacks under Non-IID data? | Under IID, all benign updates are very similar (small variance). Poisoned updates stand out easily. Under Non-IID, benign updates already have high variance due to label skew, making it much harder to distinguish malicious updates from naturally skewed benign ones. This is the realistic setting. |

---

## What Phase 4 Adds (Next)

[Brief preview of the next phase: Visualizing the impact of attacks and preparing for the proposed defense module]

---

*Phase 3 completed: 2026-10-04*
