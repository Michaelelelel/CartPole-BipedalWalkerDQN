"""Verify the standalone model runner without opening a real renderer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import test_model


class FixedActionModel(nn.Module):
    """Return deterministic Q-values that always select action one."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Produce one Q-value pair for each observation."""
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32).repeat(observations.shape[0], 1)


class TwoStepEnvironment:
    """Provide a minimal Gym-compatible environment for runner tests."""

    def __init__(self) -> None:
        """Initialize the episode step counter."""
        self.steps = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        """Start a fresh deterministic episode."""
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def render(self) -> np.ndarray:
        """Return a small RGB frame."""
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Terminate after two valid actions."""
        assert action == 1
        self.steps += 1
        return np.full(4, self.steps, dtype=np.float32), 1.5, self.steps == 2, False, {}


def test_output_paths_support_direct_files_and_directories(tmp_path: Path) -> None:
    """Resolve stable recording paths for both supported output forms."""
    directory, base = test_model.resolve_output_paths(tmp_path / "videos", "CartPole-v1", "mlp_small")
    assert directory == tmp_path / "videos"
    assert base == directory / "CartPole-v1_mlp_small"

    directory, base = test_model.resolve_output_paths(tmp_path / "episode.mp4", "CartPole-v1", "mlp_small")
    assert directory == tmp_path
    assert base == tmp_path / "episode"


def test_run_episodes_collects_returns_and_frames() -> None:
    """Run deterministic episodes and retain every rendered frame."""
    returns, frames = test_model.run_episodes(
        environment=TwoStepEnvironment(),
        model=FixedActionModel(),
        episodes=2,
        interactive=False,
    )

    assert returns == [3.0, 3.0]
    assert len(frames) == 4


def test_argument_parser_preserves_the_public_cli() -> None:
    """Keep the documented required arguments and defaults stable."""
    arguments = test_model.parse_arguments([
        "--env", "CartPole-v1",
        "--model-path", "model.pt",
        "--model-string", "mlp_small",
    ])

    assert arguments.env == "CartPole-v1"
    assert arguments.model_path == "model.pt"
    assert arguments.model_string == "mlp_small"
    assert arguments.episodes == 5
    assert arguments.output is None
    assert arguments.interactive is False
