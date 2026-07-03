"""
dqn/networks/feedforward.py
Configurable Multi-Layer Perceptron for Q-value estimation.
"""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Standard MLP architecture. 
    All structural parameters are injected during initialization.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_sizes: Sequence[int],
            dropout: float = 0.0,
    ) -> None:
        """
        Builds the network layers based on the provided configuration.
        """
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = state_dim

        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden

        # Final output layer
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Q-value logits for a given state batch.
        """
        x = x.float()
        # Flatten input if necessary
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        
        return self.model(x)
