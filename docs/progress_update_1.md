# B.Tech Project Progress Update 1
**Title:** Representation-Based Unsupervised Detection of Poisoning Attacks in Federated Learning  
**Date:** October 2026

---

## 1. Project Infrastructure & Architecture 
We have completely built the foundational Federated Learning framework from scratch. 
* **Why from scratch?** Reference implementations (like FedCVG and FedCC) hardcoded hardware parameters (e.g., `cuda:0`), which caused crashes on Apple Silicon and lacked modularity. 
* **Our Architecture:** Our framework is highly modular, separating the client logic, server aggregation, data partitioning, and attack injection. It is device-agnostic (automatically uses Apple MPS for hardware acceleration) and fully reproducible via seed fixing.

## 2. Phase 1: Establishing the Clean Baseline
Before testing attacks, we proved that our Federated Learning simulation correctly converges under normal conditions using the standard `FedAvg` aggregation algorithm.
* **Dataset & Model:** MNIST dataset using the LeNet-5 architecture.
* **Results:** Across 10 clients over 5 rounds on IID data, the global model successfully converged from 92.49% to **98.14%** test accuracy.

## 3. Phase 2: Simulating Real-World Data (Non-IID Partitioning)
Real-world Federated Learning clients do not have perfectly balanced datasets (e.g., one hospital only sees certain diseases). We successfully simulated this using **Dirichlet(α) distributions**.
* **Implementation:** We built a partitioner that splits data based on an $\alpha$ parameter. We generated visualizations (stacked bar charts and heatmaps) proving the data skew.
* **Findings:** At $\alpha=0.5$ (moderate heterogeneity), some clients hold up to 10,000 samples dominated by just two classes, while others hold only 1,800 samples. We generated a "Heterogeneity Score" metric to mathematically track this skew. 
* **Why this matters:** This data skew is what makes anomaly detection hard. Our defense will need to differentiate between a malicious update and a naturally skewed, benign Non-IID update.

## 4. Phase 3: Proving the Vulnerability (Attack Implementation)
To justify the need for a robust defense, we implemented Model Poisoning Attacks and proved that the standard `FedAvg` algorithm completely fails when attacked.
* **Setup:** 10 clients, Non-IID data ($\alpha=0.5$), 30% of clients act as malicious attackers.
* **Sign-Flip Attack:** The attackers invert their gradients. This dropped the global model's accuracy from ~98.9% to **90.3%**.
* **Scaling Attack:** The attackers scale up their gradients by a factor of 10 to hijack the aggregation. This resulted in a **catastrophic failure**, dropping the global model's accuracy to **9.82%** (worse than random guessing).

## 5. Summary of Deliverables Ready for Review
The professor can review the following in the repository:
1. `proposed/` module containing clean implementations of the partitioner, models, and attacks.
2. `experiments/results/partitions/` containing generated heatmaps and bar charts proving the Non-IID distributions.
3. Automated unit testing (`pytest`) verifying the mathematical correctness of the data splitting and poisoning algorithms (all passing).

## 6. Immediate Next Steps
* **Phase 4:** Implement standard baseline defenses (e.g., Krum, Trimmed Mean) to observe how current state-of-the-art defenses react to our attacks.
* **Phase 5 & 6:** Begin implementing our proposed CKA (Centered Kernel Alignment) representation-based detector to outperform the baselines under the Non-IID setting.
