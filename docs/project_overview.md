# Project Overview: Dataset & Methodology

This document serves as a high-level overview of the project's dataset, its core objective, and instructions on how to run it. This summary is designed to be useful for BTP presentations or viva preparations.

## 1. What Dataset Are We Using?
Currently, we are using the **MNIST** dataset (handwritten digits). 

**Why?** Because it is standard for Federated Learning (FL) baselines, trains quickly on Apple Silicon (MPS) chips, and allows us to rapidly build and verify the pipeline (partitions, attacks, etc.). 

*Note:* The codebase is already set up to handle **FashionMNIST** and **CIFAR-10** (in `proposed/data/datasets.py`). In the final phase of the project, experiments will be run on CIFAR-10 to prove the method works on more complex, real-world data.

---

## 2. What Are We Doing? (The Project Summary)
**Project Title:** *Representation-Based Unsupervised Detection of Poisoning Attacks in Federated Learning.*

**The Setup (Federated Learning):** Multiple clients (like hospitals or phones) collaboratively train a shared AI model. They don't share their private data; they only send "model updates" (gradients) to a central server, which averages them (FedAvg) to improve the global model.

**The Problem:** 
1. **Poisoning Attacks:** A malicious client can send a mathematically crafted, destructive update (like a "Scaling Attack") that completely destroys the global model. 
2. **The Non-IID Challenge:** Real-world clients have naturally diverse, skewed data (e.g., one hospital only sees X-rays for disease A, another only sees disease B). This is called **Non-IID data**. Because their data is different, their normal updates look very different from each other.
3. **The Trap:** Existing defense mechanisms often fail here. Because they just look at the raw numbers in the update, they accidentally flag *benign clients with unique data* as malicious, or they miss smart attackers hiding in the noise.

**Our Proposed Solution:**
Instead of just looking at the raw update numbers (Update-level features), we will also look at how the model *thinks* (Representation-level features). 
We extract the activations from the second-to-last layer (penultimate layer) of the model and use a metric called **CKA (Centered Kernel Alignment)** to compare how different clients' models represent data. By combining these two views into a single clustering algorithm (K-Means), our unsupervised detector will accurately isolate malicious clients, even when the data is highly Non-IID.

---

## 3. How the Dataset is Used in This Specific Project
In standard Machine Learning, the entire dataset is passed into one central model. **In Federated Learning, data is decentralized.** 

Here is exactly what we do with the dataset (`proposed/data/partitioner.py`):

**Step 1: Simulating Real-World Clients**
We take the 60,000 training images and split them among the 10 clients. The central server *never* sees this training data—it only sees the 10,000 testing images to evaluate how good the global model is.

**Step 2: IID vs. Non-IID Partitioning (The Core Research Problem)**
* **IID (Independent and Identically Distributed):** We shuffle the dataset and give exactly 6,000 random images to each of the 10 clients. Every client gets roughly equal amounts of 0s, 1s, 2s, etc. *(Phase 1)*.
* **Non-IID (The Real World):** We use a mathematical function called a **Dirichlet Distribution ($\alpha$)** to deliberately skew the data. We might give Client 1 mostly images of 0s and 1s, and Client 2 mostly images of 8s and 9s. *(Phase 2)*.

**Step 3: Why This Matters for Poisoning Attacks**
If Client 1 only has images of 0s and 1s (Non-IID data), the model update it sends to the server will look mathematically "weird" compared to Client 2. 

If Client 3 is a **malicious attacker** executing a poisoning attack (Phase 3), its update is *also* designed to look "weird" to trick the server. 

**The core challenge:** How can the server look at the updates and accurately tell the difference between Client 1 (a benign client with skewed data) and Client 3 (a malicious attacker trying to destroy the model)? Our proposed detector solves this using representation analysis.

---

## 4. How to Run This Project
The architecture is modular and research-grade. You don't have to change the code to run different experiments; you just use configuration files and the central experiment runner.

**Activate the virtual environment:**
```bash
cd BTP-FL-Poisoning
source .venv/bin/activate
```

**Run a clean baseline Federated Learning experiment (No attacks):**
```bash
python experiments/run_fl.py --config experiments/configs/base.yaml
```

**Run a Non-IID Data Analysis (Shows how skewed data affects clients):**
*(Generates heatmaps and bar charts in `experiments/results/partitions/`)*
```bash
python experiments/run_partition_analysis.py --dataset MNIST --num_clients 10
```

**Run a Poisoning Attack (e.g., Scaling Attack with 30% malicious clients):**
```bash
python experiments/run_fl.py --config experiments/configs/attack_scaling.yaml
```

**Run custom experiments using CLI flags (overrides the YAML config):**
```bash
python experiments/run_fl.py --dataset MNIST --partition noniid --alpha 0.1 --attack sign_flip --malicious_fraction 0.2
```

All results (accuracy logs, configuration backups, and metrics) are automatically saved in self-describing folders under `experiments/results/`.
