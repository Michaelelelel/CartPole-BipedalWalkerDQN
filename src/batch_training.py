"""
batch_training.py
Parallel batch training runner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dqn.dqn_agent import DQNAgent
from dqn.config import DQNConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch JSON configuration training runner.")
    parser.add_argument("--env", type=str, required=True, help="The target Gym environment ID.")
    parser.add_argument("--configs-dir", default="test_configs", dest="configs_dir", help="Directory containing JSON configuration files.")
    return parser.parse_args()


def run_one(env_name: str, config_path: str) -> str:
    settings = DQNConfig.load(env_id=env_name, path=config_path)
    
    print(f"Starting batch run: {settings.run_name} for environment {settings.env_id}")
    agent = DQNAgent(settings)
    agent.train()
    return f"Completed batch run: {settings.run_name}"


def main() -> None:
    args = parse_args()
    configs_dir = Path(args.configs_dir)

    if not configs_dir.exists() or not configs_dir.is_dir():
        print(f"Error: Configurations directory '{configs_dir}' does not exist.")
        return

    json_files = sorted(configs_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON configuration files found in '{configs_dir}'.")
        return

    print(f"Found {len(json_files)} configuration files. Starting execution...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_one, args.env, str(j_file)): j_file.stem
            for j_file in json_files
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"Error in config '{name}': {exc}")

    print("Batch training complete!")


if __name__ == "__main__":
    main()
