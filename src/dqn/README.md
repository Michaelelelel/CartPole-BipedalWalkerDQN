# DQN Core Module (`src/dqn/`)

This directory contains the core logic for the Deep Q-Network (DQN) reinforcement learning algorithm. The package is split into small modules so that configuration, training orchestration, replay memory, optimization, networks, policies, and reporting remain easy to inspect.

## Why is there an `__init__.py` file in these folders?

In Python, an `__init__.py` file marks a directory as a **package**. This allows imports such as `from dqn.dqn_agent import DQNAgent` and lets subpackages such as `dqn.networks` and `dqn.policies` expose clean factory interfaces.

## Directory Structure

* **`config.py`**: Defines the `DQNConfig` dataclass and loads JSON hyperparameter files.
* **`dqn_agent.py`**: Main coordinator for environment setup, replay collection, optimization, target-network sync, evaluation, and model saving.
* **`learner.py`**: Implements the Double-DQN loss, Adam optimization step, and target-network synchronization.
* **`memory.py`**: Experience replay ring buffer for storing and sampling `(state, action, reward, next_state, done)` transitions.
* **`logic.py`**: Shared utility functions, currently the linear exploration annealing schedule.
* **`reporting.py`**: Console logging and optional MLflow metric tracking.
* **`networks/`**: Contains the configurable MLP Q-network architectures (`mlp_small`, `mlp_medium`, `mlp_large`).
* **`policies/`**: Contains action-selection policies: epsilon-greedy, Boltzmann, and greedy evaluation.
