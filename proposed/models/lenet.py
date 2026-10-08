"""
LeNet-5 model for MNIST / FashionMNIST.

Architecture (input: 1×28×28):
    conv1 (1→6, 5×5) → ReLU → MaxPool(2×2)
    conv2 (6→16, 5×5) → ReLU → MaxPool(2×2)
    flatten → fc1 (256→120) → ReLU
    fc2 (120→84) → ReLU          ← PENULTIMATE LAYER
    fc3 (84→num_classes)         ← CLASSIFIER HEAD

Penultimate layer: fc2
    - Output dimension: 84
    - Activation: ReLU applied
    - This is the layer used for representation extraction in Phase 6 (CKA).

Design decision: The penultimate layer is explicitly named 'fc2' and its
dimension is stored as a class constant (PENULTIMATE_DIM = 84). This makes
the code self-documenting and avoids the fragile key-index approach used
in FedCC (keys[-4]).

The method get_penultimate_representation() provides a clean API for
extracting activations without rewriting the forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 for grayscale image classification (MNIST, FashionMNIST).

    Constants:
        PENULTIMATE_DIM: Dimension of the penultimate-layer representation (84).
        PENULTIMATE_LAYER_NAME: Name of the penultimate layer ('fc2').

    Args:
        num_classes: Number of output classes (default: 10).
    """

    PENULTIMATE_DIM: int = 84
    PENULTIMATE_LAYER_NAME: str = "fc2"

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Feature extraction layers ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier layers ---
        # After two conv+pool stages: 16 × 4 × 4 = 256 features
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)          # Penultimate layer
        self.fc3 = nn.Linear(84, num_classes)  # Classification head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass. Returns class logits."""
        x = self.pool(F.relu(self.conv1(x)))   # → [B, 6, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))   # → [B, 16, 4, 4]
        x = x.view(x.size(0), -1)             # → [B, 256]
        x = F.relu(self.fc1(x))               # → [B, 120]
        x = F.relu(self.fc2(x))               # → [B, 84]  ← penultimate
        x = self.fc3(x)                        # → [B, num_classes]
        return x

    def get_penultimate_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate-layer activations (fc2 output after ReLU).

        This is the representation used for CKA similarity computation in the
        proposed detection method (Phase 6). The model must be in eval() mode
        and the input must be a batch of data samples.

        Args:
            x: Input tensor of shape [N, 1, 28, 28].

        Returns:
            Representation tensor of shape [N, 84].
        """
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))  # penultimate representation
        return x


class SimpleCNN(nn.Module):
    """
    A lightweight CNN for MNIST as an alternative to LeNet-5.
    Smaller model, faster training — useful for quick smoke tests.

    Architecture (input: 1×28×28):
        conv1 (1→32, 3×3) → ReLU → MaxPool(2×2)
        conv2 (32→64, 3×3) → ReLU → MaxPool(2×2)
        flatten → fc1 (64×6×6→128) → ReLU    ← PENULTIMATE LAYER
        fc2 (128→num_classes)                  ← CLASSIFIER HEAD

    Penultimate layer: fc1 (dim=128)
    """

    PENULTIMATE_DIM: int = 128
    PENULTIMATE_LAYER_NAME: str = "fc1"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pool layers: 64 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # Penultimate
        self.fc2 = nn.Linear(128, num_classes)    # Classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))   # penultimate
        x = self.fc2(x)
        return x

    def get_penultimate_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer activations (fc1 output after ReLU)."""
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
        return x


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to build a model by name.

    Args:
        model_name:  One of 'LeNet5', 'SimpleCNN'.
        num_classes: Number of output classes.

    Returns:
        Initialized nn.Module.

    Raises:
        ValueError: If model_name is not recognised.
    """
    registry = {
        "lenet5": LeNet5,
        "simplecnn": SimpleCNN,
    }
    key = model_name.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[key](num_classes=num_classes)
