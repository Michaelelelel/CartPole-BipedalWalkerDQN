"""
dqn/dqn_agent.py
DQN Agent with modern, modular architecture.
"""
from __future__ import annotations

import uuid
import torch
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # Allows config-only tests to import DQNAgent without Gymnasium installed.
    gym = None
    spaces = None
from pathlib import Path

from .config import DQNConfig
from .memory import ExperienceReplay
from .learner import DQNLearner
from .networks.factory import NetworkFactory
from .policies import PolicyFactory, GreedyPolicy
from .reporting import MetricsTracker, logger


def _is_bipedal(env_id: str) -> bool:
    """Return True when the environment requires the discrete BipedalWalker wrapper."""
    return "BipedalWalker" in env_id


class DQNAgent:
    """
    Modular DQN Agent that manages the interaction between the policy,
    the learner, and the replay memory.
    """

    def __init__(self, config: DQNConfig):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to create and train a DQNAgent")

        self.config = config
        self.device = torch.device(config.device)
        self.epoch_history: list[float] = []

        # 1. Environment dimensions — use a throwaway env to inspect spaces
        temp_env = gym.make(config.env_id)
        if _is_bipedal(config.env_id):
            from env.bipedal_wrapper import DiscreteBipedalWrapper
            temp_env = DiscreteBipedalWrapper(temp_env)

        if not isinstance(temp_env.action_space, spaces.Discrete):
            raise ValueError(
                f"DQN requires a discrete action space, got {type(temp_env.action_space).__name__}"
            )

        state_dim = int(np.prod(temp_env.observation_space.shape))
        action_dim = int(temp_env.action_space.n)
        state_shape = temp_env.observation_space.shape
        temp_env.close()

        self._validate(config)

        # 2. Neural Network
        q_net = NetworkFactory.create(
            config.network_type,
            input_dim=state_dim,
            output_dim=action_dim
        )

        # 3. Policy — explore_start/explore_end are shared across policies:
        #    epsilon_greedy interprets them as epsilon, boltzmann as temperature.
        policy_kwargs: dict = {}
        if config.policy == "epsilon_greedy":
            policy_kwargs = {
                "epsilon_start": config.explore_start,
                "epsilon_end": config.explore_end,
            }
        elif config.policy == "boltzmann":
            policy_kwargs = {
                "temperature_start": config.explore_start,
                "temperature_end": config.explore_end,
            }

        self.policy = PolicyFactory.create(
            config.policy,
            network=q_net,
            device=self.device,
            seed=config.seed,
            **policy_kwargs
        )

        # 4. Replay Memory and Learner
        self.memory = ExperienceReplay(
            capacity=config.buffer_size,
            state_shape=state_shape,
            device=self.device
        )

        self.learner = DQNLearner(
            policy_network=self.policy.network,
            learning_rate=config.learning_rate,
            discount=config.gamma,
            device=self.device
        )

        # 5. Vectorised Training Environment (seeded for reproducibility)
        wrappers = []
        if _is_bipedal(config.env_id):
            from env.bipedal_wrapper import DiscreteBipedalWrapper
            wrappers = [lambda env: DiscreteBipedalWrapper(env)]

        wrappers.append(lambda env: gym.wrappers.RecordEpisodeStatistics(env))

        self.env = gym.make_vec(
            config.env_id,
            num_envs=config.num_envs,
            vectorization_mode="sync",
            wrappers=wrappers
        )

        self.tracker = MetricsTracker(
            run_name=config.run_name or f"dqn_{config.env_id}",
            enable_mlflow=True
        )

    @property
    def _total_steps(self) -> int:
        """
        Total environment steps across the full training run (warmup + all epochs).
        Used to drive the exploration annealing schedule.
        """
        return (
            self.config.warmup_steps + self.config.steps_per_epoch * self.config.total_epochs
        ) * self.config.num_envs

    @staticmethod
    def _validate(config: DQNConfig) -> None:
        """
        Sanity-check the configuration before committing to heavy allocations.
        Fails fast with a clear message rather than a cryptic error deep in training.
        """
        if config.batch_size >= config.buffer_size:
            raise ValueError(
                f"batch_size ({config.batch_size}) must be smaller than "
                f"buffer_size ({config.buffer_size})"
            )
        if config.warmup_steps < config.batch_size:
            raise ValueError(
                f"warmup_steps ({config.warmup_steps}) must be >= batch_size "
                f"({config.batch_size}) to ensure the buffer has enough data before training"
            )

    def train(self) -> None:
        """
        Full training loop: warmup → epoch collection → optimization → finalization.
        """
        self.tracker.start_training(self.config.to_dict())

        total_steps = self._total_steps
        global_step = self.config.warmup_steps * self.config.num_envs
        opt_step = 0

        # --- Warmup: fill replay buffer with random transitions ---
        if self.config.seed is None:
            obs, _ = self.env.reset()
        else:
            obs, _ = self.env.reset(
                seed=[self.config.seed + i for i in range(self.config.num_envs)]
            )

        logger.info(f"Starting warmup for {self.config.warmup_steps} steps...")
        for _ in range(self.config.warmup_steps):
            actions = np.random.randint(0, self.env.single_action_space.n, size=self.config.num_envs)
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            self.memory.add(obs, actions, rewards, next_obs, np.logical_or(terms, truncs))
            obs = next_obs

        logger.info("Warmup complete. Starting main training loop.")

        # --- Epoch Loop ---
        for epoch in range(self.config.total_epochs):
            epoch_rewards: list[float] = []
            epoch_lengths: list[int] = []

            # Collection Phase
            for _ in range(self.config.steps_per_epoch):
                actions = self.policy.get_action(
                    obs,
                    current_step=global_step,
                    max_steps=total_steps
                )
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                self.memory.add(obs, actions, rewards, next_obs, np.logical_or(terms, truncs))
                obs = next_obs
                global_step += self.config.num_envs

                if "episode" in infos:
                    episode_info = infos["episode"]
                    mask = infos.get("_episode", np.ones(self.config.num_envs, dtype=bool))

                    for i, finished in enumerate(mask):
                        if finished:
                            epoch_rewards.append(float(episode_info["r"][i]))
                            epoch_lengths.append(int(episode_info["l"][i]))

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            epoch_rewards.append(float(info["episode"]["r"]))
                            epoch_lengths.append(int(info["episode"]["l"]))

            # Optimization Phase
            epoch_losses: list[float] = []
            for _ in range(self.config.updates_per_epoch):
                batch = self.memory.sample(self.config.batch_size)
                loss = self.learner.update(*batch)
                epoch_losses.append(loss)
                opt_step += 1

            # Target Network Sync
            if (epoch + 1) % self.config.target_sync_interval == 0:
                self.learner.sync_target_network()

            # Per-Epoch Reporting
            stats = {
                "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                "length": float(np.mean(epoch_lengths)) if epoch_lengths else 0.0,
            }
            self.tracker.log_epoch(epoch, self.config.total_epochs, stats)
            if epoch_lengths:
                self.epoch_history.append(float(np.mean(epoch_lengths)))

        self.tracker.end_training()
        self._finalize()

    def evaluate(self, episodes: int = 20, max_episode_steps: int = 10_000) -> float:
        """
        Evaluate the trained Q-network in fresh environment instances with a greedy policy.

        Returns:
            float: Mean episode length over all evaluation episodes.
        """
        if gym is None:
            raise ImportError("gymnasium is required to evaluate a DQNAgent")

        eval_env = gym.make(self.config.env_id, max_episode_steps=max_episode_steps)
        if _is_bipedal(self.config.env_id):
            from env.bipedal_wrapper import DiscreteBipedalWrapper
            eval_env = DiscreteBipedalWrapper(eval_env)

        eval_policy = GreedyPolicy(
            network=self.policy.network,
            device=self.device
        )

        self.policy.network.eval()

        lengths: list[int] = []
        try:
            for ep in range(episodes):
                seed = None if self.config.seed is None else self.config.seed + ep
                s, _ = eval_env.reset(seed=seed)

                done = False
                ep_len = 0

                while not done:
                    action = eval_policy.get_action(s)[0]
                    s, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    ep_len += 1
                lengths.append(ep_len)
        finally:
            eval_env.close()
            self.policy.network.train()

        return float(np.mean(lengths))

    def _finalize(self) -> None:
        """
        End-of-training tasks: log final stats, run evaluation, save weights.
        """
        final_metrics: dict = {}

        if self.epoch_history:
            final_metrics["avg_length_last_100"] = float(np.mean(self.epoch_history[-100:]))

        try:
            final_metrics["eval_avg_length"] = self.evaluate(episodes=20, max_episode_steps=10_000)
        except Exception as exc:
            logger.warning(f"Evaluation failed: {exc}")

        self.tracker.log_final(self.config.total_epochs, final_metrics)
        self._save_model()

    def _save_model(self) -> None:
        """
        Persist the trained policy network weights to disk and log the path to MLflow.
        """
        try:
            save_dir = Path("results") / self.config.env_id / "artifacts"
            save_dir.mkdir(parents=True, exist_ok=True)

            name_parts = [self.config.run_name or "dqn-run"]
            if self.config.seed is not None:
                name_parts.append(f"seed{self.config.seed}")
            name_parts.append(uuid.uuid4().hex[:6])
            save_path = save_dir / ("_".join(name_parts) + ".pt")

            torch.save(self.learner.brain.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")
            self.tracker.log_final(self.config.total_epochs, {"model_path": str(save_path.resolve())})
        except Exception as exc:
            logger.warning(f"Failed to save model: {exc}")
