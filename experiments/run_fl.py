"""
Main FL experiment runner.

Usage:
    # From BTP-FL-Poisoning directory:
    python experiments/run_fl.py --config experiments/configs/base.yaml
    python experiments/run_fl.py --dataset MNIST --num_clients 10 --num_rounds 5 --partition iid
    python experiments/run_fl.py --dataset MNIST --partition noniid --alpha 0.5 --attack none --defense fedavg

Supported arguments:
    All keys in experiments/configs/base.yaml can be set via CLI flags.
    CLI flags override YAML config values.

Phase 1 scope:
    - Clean FL simulation (no attacks)
    - IID and Non-IID partitioning
    - FedAvg aggregation
    - MNIST dataset, LeNet-5 model

This file is intentionally kept as the ONLY entry point for running experiments.
Do not scatter experiment logic across multiple scripts.
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path regardless of working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

from proposed.utils.reproducibility import set_seed
from proposed.utils.device import get_device, device_name
from proposed.data.datasets import load_dataset, get_num_classes
from proposed.data.partitioner import (
    iid_partition,
    dirichlet_partition,
    make_client_datasets,
    partition_summary,
)
from proposed.models.lenet import build_model
from proposed.federated.client import FLClient
from proposed.federated.server import FLServer
from proposed.aggregation.fedavg import fedavg
from proposed.aggregation.baselines import median, trimmed_mean, krum, multi_krum, fedcvg, fedcc
from proposed.evaluation.metrics import ResultLogger


# ── Config handling ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. CLI flags override YAML config."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (base defaults)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "experiments", "configs", "base.yaml"),
        help="Path to YAML configuration file.",
    )

    # Data
    parser.add_argument("--dataset", type=str, help="MNIST | FashionMNIST | CIFAR10")
    parser.add_argument("--data_root", type=str, help="Dataset storage directory")
    parser.add_argument("--model", type=str, help="LeNet5 | SimpleCNN")

    # Federated learning
    parser.add_argument("--num_clients", type=int, help="Total number of FL clients")
    parser.add_argument("--clients_per_round", type=int, help="Clients selected per round")
    parser.add_argument("--num_rounds", type=int, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, help="Local epochs per client")
    parser.add_argument("--batch_size", type=int, help="Local mini-batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, help="SGD weight decay")

    # Partition
    parser.add_argument("--partition", type=str, choices=["iid", "noniid"], help="Data partition type")
    parser.add_argument("--alpha", type=float, help="Dirichlet alpha for Non-IID")

    # Attack & defense
    parser.add_argument(
        "--attack", type=str,
        choices=["none", "sign_flip", "scaling", "noise", "backdoor"],
        help="Poisoning attack type",
    )
    parser.add_argument("--malicious_fraction", type=float, help="Fraction of malicious clients")
    parser.add_argument("--scale_factor", type=float, default=10.0, help="Scaling factor for scaling attack")
    parser.add_argument(
        "--defense", type=str,
        choices=["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "fedcvg", "fedcc", "proposed"],
        help="Defense / aggregation method",
    )
    parser.add_argument("--krum_f", type=int, help="Number of Byzantine attackers for Krum/Multi-Krum")
    parser.add_argument("--krum_m", type=int, help="Number of selected clients for Multi-Krum")
    parser.add_argument("--trim_ratio", type=float, help="Fraction to trim for Trimmed Mean")

    # Misc
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--results_dir", type=str, help="Root directory for results")

    return parser.parse_args()


def merge_config(yaml_cfg: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config with CLI args.
    CLI values take precedence over YAML values.
    """
    cfg = dict(yaml_cfg)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            cfg[key] = value
    return cfg


# ── Main experiment function ────────────────────────────────────────────────────

def run_experiment(cfg: dict) -> None:
    """
    Execute one complete FL experiment.

    Args:
        cfg: Merged configuration dict.
    """
    # ── 0. Setup ──────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device = get_device()

    print("=" * 60)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"  Dataset:          {cfg['dataset']}")
    print(f"  Model:            {cfg['model']}")
    print(f"  Clients:          {cfg['num_clients']} (per round: {cfg['clients_per_round']})")
    print(f"  Rounds:           {cfg['num_rounds']}")
    print(f"  Local epochs:     {cfg['local_epochs']}")
    print(f"  Partition:        {cfg['partition']}" +
          (f"  (alpha={cfg['alpha']})" if cfg['partition'] == 'noniid' else ""))
    print(f"  Attack:           {cfg['attack']}  (fraction: {cfg['malicious_fraction']})")
    print(f"  Defense:          {cfg['defense']}")
    print(f"  Seed:             {cfg['seed']}")
    print(f"  Device:           {device_name()}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset ...")
    data_root = os.path.join(_PROJECT_ROOT, cfg.get("data_root", "data"))
    train_dataset, test_dataset = load_dataset(cfg["dataset"], data_root=data_root)
    num_classes = get_num_classes(cfg["dataset"])
    print(f"  Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # ── 2. Partition data ─────────────────────────────────────────────────────
    print("\n[2/5] Partitioning data ...")
    if cfg["partition"] == "iid":
        partition = iid_partition(train_dataset, cfg["num_clients"])
    else:
        partition = dirichlet_partition(
            train_dataset, cfg["num_clients"], alpha=cfg["alpha"]
        )

    client_datasets = make_client_datasets(train_dataset, partition)
    summary = partition_summary(train_dataset, partition)

    # Print partition statistics
    sizes = [v["num_samples"] for v in summary.values()]
    print(f"  Client sample counts — min: {min(sizes)}, max: {max(sizes)}, "
          f"mean: {sum(sizes)/len(sizes):.0f}")

    # ── 3. Build model and clients ────────────────────────────────────────────
    print("\n[3/5] Initialising model and clients ...")
    global_model = build_model(cfg["model"], num_classes=num_classes)
    global_model = global_model.to(device)

    # Malicious client IDs (for evaluation only — Phase 1 has none)
    num_malicious = int(cfg["num_clients"] * cfg["malicious_fraction"])
    malicious_ids = set(range(num_malicious))  # clients 0..num_malicious-1
    if malicious_ids:
        print(f"  Malicious clients: {sorted(malicious_ids)}")

    clients = [
        FLClient(
            client_id=cid,
            dataset=client_datasets[cid],
            model=global_model,
            local_epochs=cfg["local_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            device=device,
            is_malicious=(cid in malicious_ids),
            attack_type=cfg["attack"],
            attack_kwargs={"scale_factor": cfg.get("scale_factor", 10.0)},
        )
        for cid in range(cfg["num_clients"])
    ]
    print(f"  Created {len(clients)} clients.")

    # ── 4. Select aggregator ──────────────────────────────────────────────────
    aggregator = _get_aggregator(cfg)

    # ── 5. Run FL ─────────────────────────────────────────────────────────────
    results_dir = os.path.join(_PROJECT_ROOT, cfg.get("results_dir", "experiments/results"))
    logger = ResultLogger(cfg, results_dir=results_dir)

    server = FLServer(
        global_model=global_model,
        clients=clients,
        test_dataset=test_dataset,
        clients_per_round=cfg["clients_per_round"],
        aggregator=aggregator,
        device=device,
        malicious_ids=malicious_ids,
    )

    print(f"\n[4/5] Running {cfg['num_rounds']} FL rounds ...")
    for rnd in range(cfg["num_rounds"]):
        record = server.run_round(round_idx=rnd)
        logger.log_round(record)

    # ── 6. Save results ───────────────────────────────────────────────────────
    print("\n[5/5] Saving results ...")
    logger.save()
    print("\nDone.")


import functools

def _get_aggregator(cfg: dict):
    """Return the aggregation function for the given defense."""
    defense = cfg.get("defense", "fedavg")
    
    if defense == "fedavg":
        return fedavg
    elif defense == "median":
        return median
    elif defense == "trimmed_mean":
        trim_ratio = cfg.get("trim_ratio", 0.1)
        return functools.partial(trimmed_mean, trim_ratio=trim_ratio)
    elif defense == "krum":
        f = cfg.get("krum_f", 1) # Default to 1 attacker assumption
        return functools.partial(krum, f=f)
    elif defense == "multi_krum":
        f = cfg.get("krum_f", 1)
        m = cfg.get("krum_m", 1)
        return functools.partial(multi_krum, f=f, m=m)
    elif defense == "fedcvg":
        return fedcvg
    elif defense == "fedcc":
        return fedcc
    elif defense == "proposed":
        # Phase 6/7 will register their aggregator here.
        print(f"  [WARN] Defense '{defense}' not yet implemented. Using FedAvg.")
        return fedavg
    else:
        raise ValueError(f"Unknown defense: {defense}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Load base YAML config, then override with CLI args
    if os.path.exists(args.config):
        yaml_cfg = load_config(args.config)
    else:
        yaml_cfg = {}

    cfg = merge_config(yaml_cfg, args)
    run_experiment(cfg)
