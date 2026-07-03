"""Load a trained DQN model, evaluate it, and optionally record rendered episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn

from dqn.networks.factory import NetworkFactory


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public command-line interface for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to PyTorch model weights")
    parser.add_argument("--model-string", type=str, required=True, help="Registered network architecture")
    parser.add_argument("--env", type=str, required=True, help="Gymnasium environment identifier")
    parser.add_argument("--output", type=str, default=None, help="Recording directory or MP4/GIF path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--interactive", action="store_true", help="Render interactively without recording")
    return parser.parse_args(arguments)


def create_environment(environment_id: str, interactive: bool) -> Any:
    """Create the requested evaluation environment with the appropriate renderer and wrapper."""
    render_mode = "human" if interactive else "rgb_array"
    environment = gym.make(environment_id, render_mode=render_mode)
    if "BipedalWalker" in environment_id:
        from env.bipedal_wrapper import DiscreteBipedalWrapper
        environment = DiscreteBipedalWrapper(environment)
    return environment


def load_model(model_path: str | Path, model_name: str, environment: Any) -> nn.Module:
    """Construct the configured network and restore its serialized parameters."""
    state_dimension = int(np.prod(environment.observation_space.shape))
    action_dimension = int(environment.action_space.n)
    model = NetworkFactory.create(model_name, input_dim=state_dimension, output_dim=action_dimension)
    parameters = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(parameters)
    model.eval()
    return model


def choose_action(model: nn.Module, observation: np.ndarray) -> int:
    """Select the greedy action predicted by a trained Q-network."""
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return int(model(observation_tensor).argmax(dim=1).item())


def run_episodes(
    environment: Any,
    model: nn.Module,
    episodes: int,
    interactive: bool,
) -> tuple[list[float], list[np.ndarray]]:
    """Evaluate a model and return episode rewards together with captured RGB frames."""
    returns: list[float] = []
    frames: list[np.ndarray] = []

    for episode_index in range(episodes):
        observation, _ = environment.reset()
        episode_return = 0.0
        finished = False

        while not finished:
            frame = environment.render()
            if not interactive and frame is not None:
                frames.append(np.asarray(frame))

            action = choose_action(model, observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += float(reward)
            finished = bool(terminated or truncated)

        returns.append(episode_return)
        print(f"Episode {episode_index + 1}/{episodes}: return={episode_return:.2f}")

    return returns, frames


def resolve_output_paths(
    output: str | Path | None,
    environment_id: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Resolve the recording directory and suffix-free filename base."""
    requested_path = Path(output) if output is not None else Path("results") / environment_id / "videos"
    if requested_path.suffix.lower() in {".mp4", ".gif"}:
        return requested_path.parent, requested_path.with_suffix("")
    return requested_path, requested_path / f"{environment_id}_{model_name}"


def save_recordings(frames: list[np.ndarray], directory: Path, filename_base: Path) -> None:
    """Write captured frames as MP4 and GIF recordings."""
    directory.mkdir(parents=True, exist_ok=True)
    mp4_path = filename_base.with_suffix(".mp4")
    gif_path = filename_base.with_suffix(".gif")
    imageio.mimsave(mp4_path, frames, fps=30)
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved {len(frames)} frames to {mp4_path} and {gif_path}")


def print_summary(returns: list[float]) -> None:
    """Print the mean return when at least one episode was evaluated."""
    if returns:
        print(f"Average return over {len(returns)} episodes: {float(np.mean(returns)):.2f}")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run model loading, evaluation, reporting, and optional recording."""
    options = parse_arguments(arguments)
    environment = create_environment(options.env, options.interactive)
    try:
        model = load_model(options.model_path, options.model_string, environment)
        returns, frames = run_episodes(environment, model, options.episodes, options.interactive)
    finally:
        environment.close()

    print_summary(returns)
    if not options.interactive and frames:
        directory, filename_base = resolve_output_paths(options.output, options.env, options.model_string)
        save_recordings(frames, directory, filename_base)


if __name__ == "__main__":
    main()
