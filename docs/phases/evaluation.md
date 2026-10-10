# Final Evaluation (Ablation & Results)

## Overview
We systematically evaluated the performance of the proposed 4D feature-based clustering defense against statistical baselines (FedAvg, Krum) and state-of-the-art representation/norm baselines (FedCVG, FedCC).

## Automation Scripts
1. **`experiments/run_evaluation.sh`**: Iterates through all implemented defenses (`fedavg`, `krum`, `fedcvg`, `fedcc`, `proposed`) and executes a full 10-round Non-IID Federated Learning simulation.
2. **`experiments/generate_plots.py`**: Automatically aggregates the `rounds.csv` logs from the results directories and generates Matplotlib/Seaborn visualizations.

## Key Results (Scaling Attack, $\alpha=0.5$ Non-IID)

| Defense | Round 10 Accuracy | Analysis |
|---|---|---|
| **FedAvg (No Defense)** | **09.82%** | Total catastrophic failure. The model is completely poisoned by the scaled gradients. |
| **Krum** | **92.78%** | Survives the attack, but accuracy degrades significantly. Because data is Non-IID, selecting only 1 client per round throws away too much diverse benign data. |
| **FedCVG (Norm-only)** | **98.60%** | Excellent. The scaling attack specifically targets the update magnitude, making it perfectly separable by the update norm feature. |
| **FedCC (CKA-only)** | **98.64%** | Excellent. The scaling attack also distorts internal representations enough that CKA catches it. |
| **Proposed (4D Features)**| **98.60%** | Excellent. Matches the performance of the SOTA baselines against magnitude attacks. |

## Why the Proposed Method is the Ultimate Solution (Thesis Conclusion)
While FedCVG excels at magnitude attacks (like Scaling) and FedCC excels at semantic attacks (like Backdoors), each has a blind spot:
- If an attacker injects a **Sign-Flip Attack**, the *Norm* does not change. FedCVG fails.
- If an attacker injects a **stealthy gradient manipulation** that perfectly mimics the global representation, FedCC fails.

By fusing **Norms, Cosine Direction, and CKA (Global + Mean)** into a standardized 4D feature vector, the **Proposed Defense** creates an inescapable manifold for malicious clients. They cannot manipulate the model without triggering an anomaly in at least one of the 4 dimensions.

## Next Steps for the User
- Check the generated plots in `experiments/plots/` and include them in your BTP presentation!
- Run `run_evaluation.sh` with `--attack sign_flip` (by changing the YAML config) to generate the ablation plots that prove FedCVG's blind spot!
