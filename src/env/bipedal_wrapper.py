"""
bipedal_wrapper.py
Provides a Gymnasium ActionWrapper to discretize the continuous joint torque space
of the BipedalWalker environment and applies optimized reward shaping.
Encourages alternating bilateral leg usage (prevents skating/one-legged wiggling).
"""

from __future__ import annotations

import itertools
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete


class DiscreteBipedalWrapper(gym.ActionWrapper):
    """
    Gymnasium wrapper that discretizes the continuous 4D action space of BipedalWalker-v3
    into a discrete action space of 2^4 = 16 actions (using bins=2 for Bang-Bang control),
    and reshapes rewards to prevent one-legged sliding local optima.
    """

    # Constants for reward shaping
    PENALTY_FALL = -5.0
    PENALTY_POSTURE = -2.0
    PENALTY_STABILIZATION = -0.5
    PENALTY_STUCK = -0.3
    REWARD_VELOCITY = 5.0
    EMA_ALPHA = 0.08
    MAX_CONTACT_STEPS = 15

    def __init__(self, env: gym.Env, bins: int = 2) -> None:
        """
        Initialize the discretization wrapper and compute the discrete action torque mappings.

        Args:
            env (gym.Env): The continuous BipedalWalker environment.
            bins (int): Discretization steps per joint. Defaults to 2 (Bang-Bang: -1.0, 1.0).
        """
        super().__init__(env)
        
        self.num_joints = 4
        self.bins = bins
        joint_values = np.linspace(-1.0, 1.0, bins)
        self.action_mapping = list(itertools.product(joint_values, repeat=self.num_joints))
        self.action_space = Discrete(len(self.action_mapping))
        
        self.leg1_contact_steps = 0
        self.leg2_contact_steps = 0
        self.leg1_contact_ema = 0.5
        self.leg2_contact_ema = 0.5

    def reset(self, **kwargs):
        """
        Reset the environment and the ground contact durations.
        """
        self.leg1_contact_steps = 0
        self.leg2_contact_steps = 0
        self.leg1_contact_ema = 0.5
        self.leg2_contact_ema = 0.5
        return self.env.reset(**kwargs)

    def action(self, action: int) -> np.ndarray:
        """
        Map the discrete action integer ID to the 4D continuous torque force command vector.

        Args:
            action (int): The selected action index.

        Returns:
            np.ndarray: The 4D continuous joint forces vector.
        """
        return np.array(self.action_mapping[action], dtype=np.float32)

    def reverse_action(self, action: np.ndarray) -> int:
        """
        Find the closest discrete action integer index corresponding to a continuous force vector.

        Args:
            action (np.ndarray): The 4D continuous joint forces vector.

        Returns:
            int: The closest matched discrete action index.
        """
        distances = [np.linalg.norm(action - np.array(a)) for a in self.action_mapping]
        return int(np.argmin(distances))

    def step(self, action: int):
        """
        Execute an environment action step with custom reward shaping metrics.

        Args:
            action (int): The selected discrete action index.

        Returns:
            tuple: (obs, shaped_reward, terminated, truncated, info)
        """
        continuous_action = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        hull_angle = float(obs[0])
        hull_angular_vel = float(obs[1])
        hull_horizontal_vel = float(obs[2])
        hull_vertical_vel = float(obs[3])
        
        hip1_speed = abs(float(obs[5]))
        hip2_speed = abs(float(obs[10]))

        leg1_contact = bool(obs[8])
        leg2_contact = bool(obs[13])

        if reward == -100.0:
            shaped_reward = self.PENALTY_FALL
        else:
            shaped_reward = reward
            
        posture_penalty = self.PENALTY_POSTURE * abs(hull_angle)
        shaped_reward += posture_penalty

        stabilization_penalty = self.PENALTY_STABILIZATION * abs(hull_angular_vel) + self.PENALTY_STABILIZATION * abs(hull_vertical_vel)
        shaped_reward += stabilization_penalty

        if hull_horizontal_vel <= 0.0:
            shaped_reward += self.PENALTY_STUCK
        else:
            velocity_reward = hull_horizontal_vel * self.REWARD_VELOCITY
            shaped_reward += velocity_reward

        if leg1_contact:
            self.leg1_contact_steps += 1
        else:
            self.leg1_contact_steps = 0
            
        if leg2_contact:
            self.leg2_contact_steps += 1
        else:
            self.leg2_contact_steps = 0

        self.leg1_contact_ema = (1.0 - self.EMA_ALPHA) * self.leg1_contact_ema + self.EMA_ALPHA * float(leg1_contact)
        self.leg2_contact_ema = (1.0 - self.EMA_ALPHA) * self.leg2_contact_ema + self.EMA_ALPHA * float(leg2_contact)

        if self.leg1_contact_steps > self.MAX_CONTACT_STEPS:
            shaped_reward -= 0.1 * (self.leg1_contact_steps - self.MAX_CONTACT_STEPS)
        if self.leg2_contact_steps > self.MAX_CONTACT_STEPS:
            shaped_reward -= 0.1 * (self.leg2_contact_steps - self.MAX_CONTACT_STEPS)

        if self.leg1_contact_ema > 0.75:
            shaped_reward -= 1.5 * (self.leg1_contact_ema - 0.75)
        if self.leg2_contact_ema > 0.75:
            shaped_reward -= 1.5 * (self.leg2_contact_ema - 0.75)

        contact_diff = abs(self.leg1_contact_ema - self.leg2_contact_ema)
        if contact_diff > 0.3:
            shaped_reward -= 1.0 * (contact_diff - 0.3)

        if hip1_speed < 0.05 and hip2_speed > 0.2:
            shaped_reward -= 0.3
        if hip2_speed < 0.05 and hip1_speed > 0.2:
            shaped_reward -= 0.3

        return obs, float(shaped_reward), terminated, truncated, info


def make_discrete_bipedal(env_id: str = "BipedalWalker-v3", hardcore: bool = False, **kwargs) -> gym.Env:
    """
    Factory helper function to create and return a wrapped discrete BipedalWalker environment.

    Args:
        env_id (str): Gym environment ID. Defaults to "BipedalWalker-v3".
        hardcore (bool): Enables complex environments with obstacles if True.
        **kwargs: Additional gymnasium environment arguments.

    Returns:
        gym.Env: Discretized and wrapped environment.
    """
    env = gym.make(env_id, hardcore=hardcore, **kwargs)
    return DiscreteBipedalWrapper(env)
