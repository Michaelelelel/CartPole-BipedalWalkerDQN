# Parameters Explained

This document explains the architecture of how parameters are managed in this project, and provides a detailed breakdown of what every hyperparameter does.

Configuration lives in two places:
1. **JSON config files** (`configs/cartpole.json`, `configs/bipedalwalker.json`) — set the defaults for each environment.
2. **CLI overrides** — any parameter can be overridden at runtime via `src/training.py` flags.

All JSON keys map 1:1 to `DQNConfig` field names (no translation layer).

---

## Training CLI Parameters

### Core Environment & System

| CLI Flag | JSON Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `--env` | *(required)* | — | **Gym environment ID**. `CartPole-v1` or `BipedalWalker-v3`. |
| `--config` | `config_path` | auto | **Optional path to JSON config**. Auto-detected from `--env` if not set. |
| `--device` | `device` | `cpu` | **Hardware device**. Use `cuda` for GPU acceleration. |
| `--seed` | `seed` | `null` | **Random seed** for full reproducibility (policy RNG and env reset). |
| `--envs-count` | `num_envs` | CartPole: `4`<br>Bipedal: `8` | **Parallel environments** for vectorised data collection. |

### Neural Network Architecture

| CLI Flag | JSON Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `--network` | `network_type` | CartPole: `mlp_small`<br>Bipedal: `mlp_large` | **Architecture key**. `mlp_small` = (128, 128), `mlp_medium` = (256, 128, 64), `mlp_large` = (512, 256, 128, 64). |
| `--target-update-every` | `target_sync_interval` | CartPole: `1`<br>Bipedal: `20` | **Target network sync rate** (in epochs). Lower = more stable, higher = less staleness. |

### Exploration Strategy

| CLI Flag | JSON Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `--policy` | `policy` | `epsilon_greedy` | **Exploration strategy**. `epsilon_greedy` takes random actions with probability ε. `boltzmann` samples from a softmax over Q-values. |
| `--epsilon` | `explore_start` | CartPole: `0.1`<br>Bipedal: `1.0` | **Start exploration value**. For `epsilon_greedy`: starting ε. For `boltzmann`: starting temperature. BipedalWalker starts at `1.0` to discover walking from scratch. |
| `--epsilon-end` | `explore_end` | `0.05` | **Final exploration value**. The floor the value decays to over the full training budget. |

### Gradient Descent & Replay Buffer

| CLI Flag | JSON Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lr` | `learning_rate` | CartPole: `1e-3`<br>Bipedal: `3e-4` | **Adam learning rate**. How aggressively weights are updated per step. |
| `--gamma` | `gamma` | `0.99` | **Discount factor γ**. How much the agent values future rewards. |
| `--epochs` | `total_epochs` | `300` | **Training epochs**. One epoch = one collection + one optimization phase. |
| `--steps-per-epoch` | `steps_per_epoch` | CartPole: `100`<br>Bipedal: `500` | **Environment steps per epoch**. Frames collected before each optimization phase. |
| `--prewarm` | `warmup_steps` | CartPole: `1000`<br>Bipedal: `10000` | **Random pre-fill steps**. Populates the replay buffer before learning begins. |
| `--optim-per-epoch` | `updates_per_epoch` | CartPole: `200`<br>Bipedal: `1000` | **Gradient steps per epoch**. How many mini-batch updates happen after each collection phase. |
| `--batch-size` | `batch_size` | CartPole: `64`<br>Bipedal: `128` | **Mini-batch size**. Experiences sampled per gradient step. |
| `--replay-buffer-size` | `buffer_size` | CartPole: `10000`<br>Bipedal: `100000` | **Replay buffer capacity**. Maximum stored `(s, a, r, s', done)` transitions. |

### Metadata

| CLI Flag | JSON Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run-name` | `run_name` | auto | **Run label** used as MLflow run name and saved `.pt` filename prefix. |
