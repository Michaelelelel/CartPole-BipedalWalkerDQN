"""
dqn/reporting.py
Clean metrics reporting and progress tracking.
"""
from __future__ import annotations

import logging
from typing import Dict, Any
try:
    import mlflow
except ModuleNotFoundError:  # Allows tests with enable_mlflow=False to run without MLflow installed.
    mlflow = None

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DQN")


class MetricsTracker:
    """
    Standard metrics tracker that integrates with MLflow and local logging.
    """

    def __init__(
        self,
        run_name: str,
        experiment: str = "DQN_Experiments",
        enable_mlflow: bool = True
    ):
        self.run_name = run_name
        self.enable_mlflow = enable_mlflow

        if enable_mlflow and mlflow is not None:
            mlflow.set_experiment(experiment)
        elif enable_mlflow:
            logger.warning("MLflow is not installed; continuing with console logging only.")
            self.enable_mlflow = False

    def start_training(self, config: Dict[str, Any]):
        """Start an MLflow run and log all hyperparameters."""
        if self.enable_mlflow and mlflow is not None:
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(config)
        logger.info(f"Started training run: {self.run_name}")

    def end_training(self):
        """Close the active MLflow run."""
        if self.enable_mlflow and mlflow is not None:
            mlflow.end_run()
        logger.info("Training complete.")

    def log_epoch(self, epoch: int, total_epochs: int, stats: Dict[str, Any]):
        """
        Log per-epoch summary metrics (mean loss, reward, episode length).
        """
        stat_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in stats.items()
        )
        logger.info(f"[{epoch + 1}/{total_epochs}] {stat_str}")

        if self.enable_mlflow and mlflow is not None:
            mlflow.log_metrics(stats, step=epoch)



    def log_final(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log end-of-training summary. Numeric values go to metrics, strings to params.
        """
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        params = {k: v for k, v in metrics.items() if not isinstance(v, (int, float))}

        if self.enable_mlflow and mlflow is not None:
            if numeric:
                mlflow.log_metrics(numeric, step=epoch)
            if params:
                mlflow.log_params(params)

        for key, value in metrics.items():
            logger.info(f"Final — {key}: {value:.4f}" if isinstance(value, float) else f"Final — {key}: {value}")
