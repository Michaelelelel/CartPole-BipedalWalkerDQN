"""
policies/greedy.py
Greedy policy for evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .base import BasePolicy


class GreedyPolicy(BasePolicy):
    """
    Greedy policy for exploitation/evaluation.
    """

    def get_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        # Fast non-blocking inference
        q_values = self._get_q_values(state)
        
        return q_values.argmax(dim=1).cpu().numpy()
