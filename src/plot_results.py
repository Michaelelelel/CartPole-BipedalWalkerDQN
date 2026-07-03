"""
plot_results.py
Reads MLflow metrics from the local database and generates training performance plots.
Saves .png files to results/<env_name>/plots/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


def plot_metric(
    metric_history: list,
    title: str,
    ylabel: str,
    save_path: Path,
    color: str = "blue",
) -> None:
    """
    Plot a single MLflow metric history and save it as a PNG.

    Args:
        metric_history: List of MLflow Metric objects.
        title: Plot title string.
        ylabel: Y-axis label string.
        save_path: Destination file path.
        color: Line color.
    """
    if not metric_history:
        print(f"No data found for '{title}'. Skipping.")
        return

    steps = [m.step for m in metric_history]
    values = [m.value for m in metric_history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, color=color, linewidth=2, label=ylabel)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def find_run_id(client: MlflowClient, env_name: str) -> str | None:
    """
    Find the most recent MLflow run ID for the given environment.

    Args:
        client: Active MLflow client.
        env_name: Gym environment ID to search for.

    Returns:
        Run ID string, or None if not found.
    """
    experiments = client.search_experiments()
    if not experiments:
        print("No MLflow experiments found.")
        return None

    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        order_by=["start_time DESC"],
    )
    for run in runs:
        if run.data.params.get("env_id") == env_name:
            print(f"Found run '{run.info.run_name}' (ID: {run.info.run_id}) for {env_name}.")
            return run.info.run_id

    print(f"No MLflow runs found matching environment: {env_name}")
    return None


def main() -> None:
    """Generate training plots from MLflow logs for a given environment."""
    parser = argparse.ArgumentParser(description="Generate training plots from MLflow logs.")
    parser.add_argument("--env", type=str, required=True,
                        help="Gym environment ID (e.g. CartPole-v1, BipedalWalker-v3)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific MLflow Run ID. If omitted, uses the most recent run.")
    args = parser.parse_args()

    client = MlflowClient()
    run_id = args.run_id or find_run_id(client, args.env)
    if run_id is None:
        return

    output_dir = Path("results") / args.env / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_metric(
            client.get_metric_history(run_id, "loss"),
            f"Training Loss ({args.env})", "Loss",
            output_dir / "loss.png", color="red",
        )
        plot_metric(
            client.get_metric_history(run_id, "length"),
            f"Episode Length ({args.env})", "Steps",
            output_dir / "length.png", color="green",
        )
        plot_metric(
            client.get_metric_history(run_id, "reward"),
            f"Episode Reward ({args.env})", "Total Reward",
            output_dir / "reward.png", color="blue",
        )
    except Exception as exc:
        print(f"Error fetching metrics from MLflow: {exc}")


if __name__ == "__main__":
    main()
