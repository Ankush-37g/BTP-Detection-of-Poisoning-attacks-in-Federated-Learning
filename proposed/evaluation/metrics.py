"""
Result logging and metrics for FL experiments.

Saves:
    - results/metrics/<experiment_id>/rounds.csv    — per-round metrics
    - results/metrics/<experiment_id>/config.json   — full experiment config
    - results/metrics/<experiment_id>/summary.json  — final summary

The experiment_id is derived from the config parameters, making it
self-describing and collision-resistant.
"""

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class ResultLogger:
    """
    Structured result logger for FL experiments.

    Usage:
        logger = ResultLogger(config, results_dir="experiments/results")
        for round_record in round_results:
            logger.log_round(round_record)
        logger.save()

    Args:
        config:      Experiment configuration dict.
        results_dir: Root directory for saving results.
    """

    def __init__(self, config: Dict[str, Any], results_dir: str = "experiments/results"):
        self.config = config
        self.results_dir = results_dir
        self.round_records: List[Dict] = []
        self.start_time = time.time()
        self.start_datetime = datetime.now().isoformat()

        # Build experiment ID from key config parameters
        self.experiment_id = self._build_experiment_id(config)
        self.exp_dir = os.path.join(results_dir, self.experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save config immediately so it's available even if experiment crashes
        self._save_config()

        print(f"[Logger] Results will be saved to: {self.exp_dir}")

    def log_round(self, record: Dict) -> None:
        """
        Log one round's results.

        Args:
            record: Dict with at minimum:
                    round, test_accuracy, test_loss, avg_client_loss
        """
        self.round_records.append(record)
        r = record['round']
        acc = record.get('test_accuracy', 0.0)
        loss = record.get('test_loss', 0.0)
        client_loss = record.get('avg_client_loss', 0.0)
        print(
            f"  Round {r+1:3d} | "
            f"Test Acc: {acc:.4f} | "
            f"Test Loss: {loss:.4f} | "
            f"Client Loss: {client_loss:.4f}"
        )

    def save(self) -> None:
        """Save all results to disk (CSV + JSON)."""
        self._save_rounds_csv()
        self._save_summary()
        print(f"[Logger] Results saved to {self.exp_dir}/")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_experiment_id(self, config: Dict) -> str:
        parts = [
            config.get("dataset", "mnist").lower(),
            config.get("partition", "iid"),
            f"a{config.get('alpha', 'na')}",
            f"c{config.get('num_clients', 10)}",
            f"r{config.get('num_rounds', 5)}",
            f"atk_{config.get('attack', 'none')}",
            f"def_{config.get('defense', 'fedavg')}",
            f"seed{config.get('seed', 42)}",
        ]
        return "_".join(parts)

    def _save_config(self) -> None:
        path = os.path.join(self.exp_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

    def _save_rounds_csv(self) -> None:
        if not self.round_records:
            return
        path = os.path.join(self.exp_dir, "rounds.csv")
        fieldnames = list(self.round_records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.round_records:
                # Flatten lists to strings for CSV
                flat = {
                    k: str(v) if isinstance(v, list) else v
                    for k, v in record.items()
                }
                writer.writerow(flat)

    def _save_summary(self) -> None:
        elapsed = time.time() - self.start_time
        if self.round_records:
            final = self.round_records[-1]
            best_acc = max(r["test_accuracy"] for r in self.round_records)
        else:
            final = {}
            best_acc = 0.0

        summary = {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "start_datetime": self.start_datetime,
            "total_time_seconds": round(elapsed, 2),
            "total_rounds": len(self.round_records),
            "final_test_accuracy": final.get("test_accuracy", None),
            "final_test_loss": final.get("test_loss", None),
            "best_test_accuracy": best_acc,
        }
        path = os.path.join(self.exp_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
