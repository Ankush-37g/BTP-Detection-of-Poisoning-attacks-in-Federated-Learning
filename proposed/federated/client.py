"""
FL Client.

Each client:
    1. Receives the current global model weights from the server.
    2. Trains locally for `local_epochs` on its private dataset.
    3. Returns the updated local weights, number of samples, training loss,
       and the raw update (delta_w = w_local - w_global).

Design decisions:
    - The client does NOT store state between rounds. It is stateless.
      Each round, it loads the global weights, trains, and returns results.
    - The raw update delta_w is computed and returned alongside the weights.
      This is needed by the feature extraction module (Phase 6) for
      update norm and cosine similarity computation.
    - Attacks are applied AFTER local training by wrapping the update.
      In Phase 1 (clean), no attack is applied.
    - The optimizer is created fresh each round (SGD with momentum).
      This matches the standard FL setup and avoids optimizer state leakage
      across rounds for clients not selected in every round.
"""

import copy
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from proposed.utils.device import get_device
from proposed.attacks.poisoning import apply_attack


StateDict = Dict[str, torch.Tensor]


class FLClient:
    """
    Federated Learning Client.

    Args:
        client_id:    Unique integer identifier.
        dataset:      Local training dataset (Subset or Dataset).
        model:        The model architecture (nn.Module). A fresh copy is made
                      for each client to avoid shared state.
        local_epochs: Number of local training epochs per round.
        batch_size:   Local mini-batch size.
        lr:           SGD learning rate.
        momentum:     SGD momentum.
        weight_decay: SGD weight decay (L2 regularization).
        device:       Torch device. If None, auto-selected via get_device().
        is_malicious: Whether this client is a malicious attacker.
        attack_type:  Type of attack (e.g., 'none', 'sign_flip', 'scaling').
        attack_kwargs: Additional parameters for the attack.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        model: nn.Module,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        is_malicious: bool = False,
        attack_type: str = "none",
        attack_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device or get_device()
        
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}

        # Each client owns its own model instance
        self.model = copy.deepcopy(model).to(self.device)

    def train(
        self,
        global_weights: StateDict,
    ) -> Tuple[StateDict, int, float, StateDict]:
        """
        Perform one round of local training.

        Steps:
            1. Load global weights into local model.
            2. Train for local_epochs on local data.
            3. Compute delta_w = w_local - w_global.
            4. Return results.

        Args:
            global_weights: state_dict from the server's global model (on CPU).

        Returns:
            Tuple of:
                local_weights  (StateDict): Updated model weights (on CPU).
                num_samples    (int):       Number of local training samples.
                avg_train_loss (float):     Average training loss this round.
                delta_w        (StateDict): Update = local_weights - global_weights.
        """
        # 1. Load global weights
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_weights.items()}
        )
        self.model.train()

        # 2. Set up data loader and optimizer
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # 3. Local training
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        # 4. Extract local weights
        local_weights: StateDict = {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
        }
        
        # 5. Apply poisoning attack if malicious
        if self.is_malicious and self.attack_type != "none":
            local_weights = apply_attack(
                local_weights, 
                global_weights, 
                self.attack_type, 
                **self.attack_kwargs
            )

        # 6. Compute update delta_w = w_local - w_global (on CPU)
        delta_w: StateDict = {
            k: local_weights[k] - global_weights[k].cpu()
            for k in local_weights
        }

        return local_weights, len(self.dataset), avg_loss, delta_w
