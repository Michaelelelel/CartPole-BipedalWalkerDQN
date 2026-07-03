"""
training.py
Command-line runner for training a DQN agent on CartPole-v1 or BipedalWalker-v3.
"""
from __future__ import annotations

import argparse

from dqn.dqn_agent import DQNAgent
from dqn.config import DQNConfig


def parse_args() -> DQNConfig:
    """
    Parse CLI arguments and return a fully resolved DQNConfig.

    All arguments are optional overrides; missing ones are filled from the
    auto-detected JSON config file for the chosen environment.
    The explicit add_argument list is intentional — it documents every tunable
    parameter and keeps each flag's type, dest, and help string in one place.
    """
    parser = argparse.ArgumentParser(description="DQN Training Runner")

    # Required
    parser.add_argument("--env", type=str, required=True, dest="env_id",
                        help="Gym environment ID (CartPole-v1 or BipedalWalker-v3)")
    parser.add_argument("--config", default=None, dest="config_path",
                        help="Path to a JSON config file (overrides the auto-detected default)")

    # System
    parser.add_argument("--device", default=None,
                        help="Torch compute device: 'cpu' or 'cuda'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--envs-count", type=int, default=None, dest="num_envs",
                        help="Number of parallel vectorised environments")

    # Model & Policy
    parser.add_argument("--network", default=None, dest="network_type",
                        help="Network architecture key: mlp_small | mlp_medium | mlp_large")
    parser.add_argument("--policy", default=None,
                        help="Exploration strategy: epsilon_greedy | boltzmann | greedy")
    parser.add_argument("--epsilon", type=float, default=None, dest="explore_start",
                        help="Start exploration value (epsilon or Boltzmann temperature)")
    parser.add_argument("--epsilon-end", type=float, default=None, dest="explore_end",
                        help="Final exploration value (epsilon or Boltzmann temperature)")

    # Training loop
    parser.add_argument("--epochs", type=int, default=None, dest="total_epochs",
                        help="Total training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=None, dest="steps_per_epoch",
                        help="Environment steps collected per epoch")
    parser.add_argument("--prewarm", type=int, default=None, dest="warmup_steps",
                        help="Random steps before training starts (pre-fills replay buffer)")
    parser.add_argument("--optim-per-epoch", type=int, default=None, dest="updates_per_epoch",
                        help="Gradient update steps per epoch")
    parser.add_argument("--target-update-every", type=int, default=None, dest="target_sync_interval",
                        help="Epochs between target-network syncs")

    # Optimisation & Memory
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate",
                        help="Adam optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor γ")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                        help="Mini-batch size for gradient updates")
    parser.add_argument("--replay-buffer-size", type=int, default=None, dest="buffer_size",
                        help="Maximum replay buffer capacity")

    # Metadata
    parser.add_argument("--run-name", default=None, dest="run_name",
                        help="Human-readable name prefix for MLflow run and saved weights")

    args = parser.parse_args()
    arg_dict = vars(args)
    env_id = arg_dict.pop("env_id")
    config_path = arg_dict.pop("config_path")

    return DQNConfig.load(env_id=env_id, path=config_path, **arg_dict)


def main() -> None:
    """Main training CLI entry point."""
    config = parse_args()
    agent = DQNAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
