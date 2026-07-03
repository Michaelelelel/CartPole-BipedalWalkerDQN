"""Verify replay-memory behavior independently from the training loop."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dqn.memory import ExperienceReplay


def transition_batch(values: list[float]) -> tuple[np.ndarray, ...]:
    """Build a deterministic transition batch from scalar identifiers."""
    observations = np.asarray([[value, -value] for value in values], dtype=np.float32)
    actions = np.asarray([int(value) % 2 for value in values], dtype=np.int64)
    rewards = np.asarray(values, dtype=np.float32)
    next_observations = observations + 0.5
    terminals = np.asarray([value < 0 for value in values], dtype=np.bool_)
    return observations, actions, rewards, next_observations, terminals


def test_sample_preserves_shapes_dtypes_and_device() -> None:
    """Return batches in the format consumed by DQNLearner."""
    replay = ExperienceReplay(capacity=8, state_shape=(2,), device=torch.device("cpu"))
    replay.add(*transition_batch([1.0, 2.0, 3.0, 4.0]))

    observations, actions, rewards, next_observations, terminals = replay.sample(3)

    assert observations.shape == (3, 2)
    assert next_observations.shape == (3, 2)
    assert observations.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert terminals.dtype == torch.bool
    assert all(tensor.device.type == "cpu" for tensor in (observations, actions, rewards, next_observations, terminals))


def test_repeated_samples_reuse_batch_storage() -> None:
    """Avoid repeated tensor allocation for the training batch size."""
    replay = ExperienceReplay(capacity=8, state_shape=(2,), device=torch.device("cpu"))
    replay.add(*transition_batch([1.0, 2.0, 3.0, 4.0]))

    first_sample = replay.sample(2)
    second_sample = replay.sample(2)

    assert all(first.data_ptr() == second.data_ptr() for first, second in zip(first_sample, second_sample))


def test_wraparound_retains_the_latest_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overwrite the oldest entries when a batch crosses the capacity boundary."""
    replay = ExperienceReplay(capacity=4, state_shape=(2,), device=torch.device("cpu"))
    replay.add(*transition_batch([0.0, 1.0, 2.0]))
    replay.add(*transition_batch([3.0, 4.0]))

    monkeypatch.setattr(torch, "randint", lambda high, size: torch.arange(size[0]))
    observations, _, _, _, _ = replay.sample(4)

    assert len(replay) == 4
    assert set(observations[:, 0].tolist()) == {1.0, 2.0, 3.0, 4.0}


def test_invalid_batches_fail_with_clear_errors() -> None:
    """Reject batches that cannot be represented safely by the buffer."""
    replay = ExperienceReplay(capacity=2, state_shape=(2,), device=torch.device("cpu"))

    with pytest.raises(ValueError, match="capacity"):
        replay.add(*transition_batch([1.0, 2.0, 3.0]))

    replay.add(*transition_batch([1.0]))
    with pytest.raises(ValueError, match="available"):
        replay.sample(2)
