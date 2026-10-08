"""
FL Server.

The server orchestrates federated learning:
    1. Maintains the global model.
    2. Selects a subset of clients each round.
    3. Broadcasts global weights to selected clients.
    4. Collects client updates.
    5. Aggregates via a pluggable aggregator (default: FedAvg).
    6. Evaluates the global model on the test set.
    7. Logs round results.

Design decisions:
    - The server accepts any aggregation function via the `aggregator` argument.
      This is the extension point for Phase 5 (FedCVG, FedCC baselines) and
      Phase 7 (proposed robust aggregation).
    - Client selection is random without replacement each round.
    - The server does NOT access any client's private data except through the
      client's train() interface.
    - Ground-truth malicious-client labels are stored separately (for evaluation
      only) and never passed to the aggregator or detector.
"""

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from proposed.utils.device import get_device
from proposed.aggregation.fedavg import fedavg
from proposed.federated.client import FLClient


StateDict = Dict[str, torch.Tensor]


class FLServer:
    """
    Federated Learning Server.

    Args:
        global_model:       Initial global model (nn.Module).
        clients:            List of FLClient instances.
        test_dataset:       Server-side test dataset for global evaluation.
        clients_per_round:  Number of clients selected each round.
                            If None, all clients participate.
        aggregator:         Aggregation function with signature:
                            aggregator([(state_dict, num_samples), ...]) → state_dict
                            Defaults to fedavg.
        device:             Torch device for the global model and evaluation.
        malicious_ids:      Set of client IDs that are malicious.
                            ONLY used for evaluation metrics, never for detection.
        eval_batch_size:    Batch size for test-set evaluation.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FLClient],
        test_dataset,
        clients_per_round: Optional[int] = None,
        aggregator: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        malicious_ids: Optional[set] = None,
        eval_batch_size: int = 256,
    ):
        self.device = device or get_device()
        self.global_model = copy.deepcopy(global_model).to(self.device)
        self.clients = clients
        self.test_loader = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=False
        )
        self.clients_per_round = clients_per_round or len(clients)
        self.aggregator = aggregator or fedavg
        self.malicious_ids = malicious_ids or set()

        # Round history (populated by run_round)
        self.history: List[Dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_round(self, round_idx: int) -> Dict:
        """
        Execute one FL communication round.

        Returns:
            Dict with round metrics:
                round, selected_clients, avg_client_loss,
                test_accuracy, test_loss
        """
        # 1. Select clients
        selected = self._select_clients()

        # 2. Get current global weights (CPU copy for broadcasting)
        global_weights = {
            k: v.cpu() for k, v in self.global_model.state_dict().items()
        }

        # 3. Client local training
        client_results = []
        client_losses = []
        for client in selected:
            local_weights, num_samples, loss, delta_w = client.train(global_weights)
            client_results.append((local_weights, num_samples))
            client_losses.append(loss)

        # 4. Aggregation
        aggregated_weights = self.aggregator(client_results)

        # 5. Update global model
        self.global_model.load_state_dict(
            {k: v.to(self.device) for k, v in aggregated_weights.items()}
        )

        # 6. Evaluate
        test_acc, test_loss = self.evaluate()

        # 7. Build round record
        record = {
            "round": round_idx,
            "selected_clients": [c.client_id for c in selected],
            "num_selected": len(selected),
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "test_accuracy": test_acc,
            "test_loss": test_loss,
        }
        self.history.append(record)
        return record

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the global model on the test set.

        Returns:
            (accuracy, avg_loss) — both floats.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        avg_loss = total_loss / max(num_batches, 1)
        return accuracy, avg_loss

    def get_global_weights(self) -> StateDict:
        """Return a CPU copy of the current global model state_dict."""
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _select_clients(self) -> List[FLClient]:
        """Randomly select clients_per_round clients without replacement."""
        k = min(self.clients_per_round, len(self.clients))
        return random.sample(self.clients, k)
