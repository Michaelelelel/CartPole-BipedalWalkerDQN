"""
dqn/config.py
Strict configuration schema for the DQN agent.
All JSON config keys map 1:1 to field names — no aliasing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DQNConfig:
    """
    Schema for DQN hyperparameters.
    Every field name matches its JSON config key exactly.
    """
    # System & Env
    env_id: str
    num_envs: int
    device: str
    seed: Optional[int]

    # Model & Policy
    network_type: str
    policy: str

    # Optimization
    learning_rate: float
    gamma: float
    buffer_size: int
    batch_size: int
    target_sync_interval: int

    # Training Loop
    total_epochs: int
    steps_per_epoch: int
    warmup_steps: int
    updates_per_epoch: int

    # Exploration — shared across policies:
    # epsilon_greedy uses these as start/end epsilon values.
    # boltzmann uses these as start/end temperature values.
    explore_start: float
    explore_end: float

    # Metadata
    run_name: Optional[str]

    @classmethod
    def load(cls, env_id: str, path: Optional[str | Path] = None, **overrides) -> DQNConfig:
        """
        Loads configuration from a JSON file, then applies any CLI overrides.
        JSON keys must match DQNConfig field names exactly.
        """
        if path is None:
            root = Path(__file__).resolve().parent.parent.parent
            name = "bipedalwalker.json" if "BipedalWalker" in env_id else "cartpole.json"
            path = root / "configs" / name
        else:
            path = Path(path)

        with open(path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        data["env_id"] = env_id

        for key, value in overrides.items():
            if value is not None:
                data[key] = value

        if not data.get("run_name"):
            data["run_name"] = f"dqn_{env_id}_{path.stem}"

        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
