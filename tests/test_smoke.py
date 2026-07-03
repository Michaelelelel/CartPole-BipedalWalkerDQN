"""
tests/test_smoke.py
Lightweight smoke tests that verify the core components load and function correctly.
These are not full integration tests — they run in seconds and are designed to catch
import errors, config loading failures, and obvious API breakage.
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Make src importable from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestConfigLoading:
    """Verify that DQNConfig loads correctly for both environments."""

    def test_cartpole_config_loads(self):
        from dqn.config import DQNConfig
        config = DQNConfig.load("CartPole-v1")
        assert config.env_id == "CartPole-v1"
        assert config.explore_start > config.explore_end
        assert config.buffer_size > config.batch_size
        assert config.warmup_steps >= config.batch_size

    def test_bipedalwalker_config_loads(self):
        from dqn.config import DQNConfig
        config = DQNConfig.load("BipedalWalker-v3")
        assert config.env_id == "BipedalWalker-v3"
        assert config.explore_start == 1.0, "BipedalWalker must start with full exploration"

    def test_cli_override_applied(self):
        from dqn.config import DQNConfig
        config = DQNConfig.load("CartPole-v1", total_epochs=42)
        assert config.total_epochs == 42

    def test_invalid_batch_size_raises(self):
        from dqn.config import DQNConfig
        from dqn.dqn_agent import DQNAgent
        config = DQNConfig.load("CartPole-v1", batch_size=99999)
        with pytest.raises(ValueError, match="batch_size"):
            DQNAgent._validate(config)

    def test_invalid_warmup_raises(self):
        from dqn.config import DQNConfig
        from dqn.dqn_agent import DQNAgent
        config = DQNConfig.load("CartPole-v1", warmup_steps=1)
        with pytest.raises(ValueError, match="warmup_steps"):
            DQNAgent._validate(config)


class TestPoliciesImport:
    """Verify that all registered policies can be imported without errors."""

    def test_epsilon_greedy_importable(self):
        from dqn.policies.epsilon_greedy import EpsilonGreedyPolicy
        assert EpsilonGreedyPolicy is not None

    def test_policy_factory_creates_epsilon_greedy(self):
        import torch
        import torch.nn as nn
        from dqn.policies import PolicyFactory
        net = nn.Linear(4, 2)
        policy = PolicyFactory.create(
            "epsilon_greedy", network=net, device=torch.device("cpu"),
            seed=None, epsilon_start=0.1, epsilon_end=0.05,
        )
        assert policy is not None


class TestExperienceReplay:
    """Verify the replay buffer add/sample cycle works correctly."""

    def test_add_and_sample(self):
        import torch
        from dqn.memory import ExperienceReplay
        buf = ExperienceReplay(capacity=1000, state_shape=(4,), device=torch.device("cpu"))

        # Add a vectorised batch of 2 transitions
        obs = np.ones((2, 4), dtype=np.float32)
        act = np.array([0, 1], dtype=np.int64)
        rew = np.array([1.0, -1.0], dtype=np.float32)
        term = np.array([False, True])
        buf.add(obs, act, rew, obs, term)

        assert len(buf) == 2

        obs_b, act_b, rew_b, next_b, term_b = buf.sample(2)
        assert obs_b.shape == (2, 4)
        assert act_b.shape == (2,)

    def test_ring_buffer_wraps(self):
        import torch
        from dqn.memory import ExperienceReplay
        buf = ExperienceReplay(capacity=4, state_shape=(2,), device=torch.device("cpu"))

        for _ in range(3):  # Add 6 transitions total — wraps around capacity=4
            obs = np.ones((2, 2), dtype=np.float32)
            buf.add(obs, np.zeros(2, dtype=np.int64), np.zeros(2), obs, np.zeros(2, dtype=bool))

        assert len(buf) == 4  # Buffer is full, not 6


class TestMetricsTracker:
    """Verify that MetricsTracker works correctly without MLflow."""

    def test_log_epoch_no_mlflow(self):
        from dqn.reporting import MetricsTracker
        tracker = MetricsTracker("test", enable_mlflow=False)
        tracker.log_epoch(0, 10, {"loss": 0.5, "reward": 1.0, "length": 100.0})

    def test_start_and_end_training_no_mlflow(self):
        from dqn.reporting import MetricsTracker
        tracker = MetricsTracker("test", enable_mlflow=False)
        tracker.start_training({"env_id": "CartPole-v1"})
        tracker.end_training()

    def test_log_final_no_mlflow(self):
        from dqn.reporting import MetricsTracker
        tracker = MetricsTracker("test", enable_mlflow=False)
        tracker.log_final(epoch=5, metrics={"eval_avg_length": 200.0, "model_path": "/some/path"})
