"""
dqn/learner.py
Core learning logic for Deep Q-Networks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import copy


class DQNLearner:
    """
    Handles the value-function optimization process.
    Implements Double-DQN to stabilize learning and reduce overestimation.
    """

    def __init__(
        self,
        policy_network: nn.Module,
        learning_rate: float,
        discount: float,
        device: torch.device,
        tau: float = 1.0,
    ):
        self.brain = policy_network
        self.target_brain = copy.deepcopy(policy_network).to(device)
        self.target_brain.eval()

        self.updater = torch.optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.gamma = discount
        self.soft_update_rate = tau
        self.criterion = nn.SmoothL1Loss()
        self.device = device

    def update(
        self, 
        s_batch: torch.Tensor, 
        a_batch: torch.Tensor, 
        r_batch: torch.Tensor, 
        s_next_batch: torch.Tensor, 
        done_batch: torch.Tensor
    ) -> float:
        """
        Calculates TD-error and performs a gradient descent step.
        """
        # Predicted Q-values for current state
        q_predictions = self.brain(s_batch)
        # Select values for the specific actions taken
        q_selected = q_predictions.gather(dim=1, index=a_batch.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN implementation: 
            # 1. Use primary network to find the best actions for the next state
            # 2. Use target network to evaluate the value of those actions
            actions_prime = self.brain(s_next_batch).argmax(dim=1, keepdim=True)
            q_next_target = self.target_brain(s_next_batch).gather(dim=1, index=actions_prime).squeeze(-1)
            
            # Double-DQN Bellman target: choose next action with the online network,
            # then evaluate that action with the target network.
            y_expected = r_batch + (1.0 - done_batch.float()) * self.gamma * q_next_target

        td_loss = self.criterion(q_selected, y_expected)

        self.updater.zero_grad(set_to_none=True)
        td_loss.backward()
        self.updater.step()

        return td_loss.item()

    def sync_target_network(self):
        """
        Synchronizes weights from the online network to the target network.
        """
        if self.soft_update_rate >= 1.0:
            self.target_brain.load_state_dict(self.brain.state_dict())
        else:
            # Polyak averaging for soft updates
            for t_par, p_par in zip(self.target_brain.parameters(), self.brain.parameters()):
                t_par.data.mul_(1.0 - self.soft_update_rate)
                t_par.data.add_(self.soft_update_rate * p_par.data)
