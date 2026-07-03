# CartPoleAndBipedalWalkerDQN

A unified, modular **Double Deep Q-Network (DQN)** framework that trains reinforcement learning agents on two environments of very different complexity: the simple **CartPole-v1** balance task and an extended **BipedalWalker-v3** locomotion challenge.

---

## 📂 Project Structure

```
src/
├── training.py          # Single-run training CLI
├── batch_training.py    # Parallel multi-config training runner
├── plot_results.py      # MLflow metrics → PNG plots
├── test_model.py        # Load weights, evaluate, record video
├── dqn/
│   ├── config.py        # DQNConfig schema and JSON loading
│   ├── dqn_agent.py     # Main agent orchestrator
│   ├── learner.py       # Double DQN optimization (TD-error, target network)
│   ├── memory.py        # Ring-buffer experience replay
│   ├── logic.py         # Linear annealing schedule
│   ├── reporting.py     # MLflow + console metrics tracker
│   ├── networks/        # MLP architectures (small / medium / large)
│   └── policies/        # Exploration strategies (epsilon-greedy, Boltzmann, greedy)
└── env/
    └── bipedal_wrapper.py  # Discretization + reward shaping for BipedalWalker
configs/
├── cartpole.json        # Default CartPole-v1 hyperparameters
└── bipedalwalker.json   # Default BipedalWalker-v3 hyperparameters
test_configs/
├── cartpole/            # Experiment configs for CartPole (architecture / policy / param sweeps)
└── bipedal/             # Experiment configs for BipedalWalker
tests/
├── test_memory.py       # Replay-buffer behavior and allocation reuse
├── test_model_runner.py # Model evaluation and recording utilities
└── test_smoke.py        # Configuration and integration smoke tests
```

---

## 🛠️ Installation

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv). The lockfile supports
Intel and Apple Silicon macOS, x86-64 Linux, and 64-bit Windows without forcing a
CUDA 12.4 build on every system:

```bash
uv sync
```

The default configurations use the CPU. CUDA remains available when a compatible
PyTorch build is installed for the target NVIDIA driver and `--device cuda` is used.

---

## 🚀 Training (`src/training.py`)

Trains a Double DQN agent. Hyperparameters are auto-loaded from the matching `configs/` file and can be overridden via CLI flags.

```bash
# Use default config for the environment
uv run src/training.py --env CartPole-v1
uv run src/training.py --env BipedalWalker-v3

# Use a custom config file
uv run src/training.py --env CartPole-v1 --config test_configs/cartpole/cp_policy_boltzmann.json

# Override individual parameters
uv run src/training.py --env CartPole-v1 --lr 0.0005 --epochs 200 --network mlp_medium
```

**All available flags:**

| Flag | JSON key | Description |
|:---|:---|:---|
| `--env` | *(required)* | Gym environment ID (`CartPole-v1` or `BipedalWalker-v3`) |
| `--config` | *(auto-detected)* | Path to a JSON config file |
| `--device` | `device` | `cpu` or `cuda` |
| `--seed` | `seed` | Random seed for full reproducibility |
| `--envs-count` | `num_envs` | Number of parallel environments |
| `--network` | `network_type` | `mlp_small` / `mlp_medium` / `mlp_large` |
| `--policy` | `policy` | `epsilon_greedy` / `boltzmann` / `greedy` |
| `--epsilon` | `explore_start` | Start exploration value (ε or temperature) |
| `--epsilon-end` | `explore_end` | Final exploration value |
| `--epochs` | `total_epochs` | Total training epochs |
| `--steps-per-epoch` | `steps_per_epoch` | Env steps collected per epoch |
| `--prewarm` | `warmup_steps` | Random steps before training starts |
| `--optim-per-epoch` | `updates_per_epoch` | Gradient updates per epoch |
| `--target-update-every` | `target_sync_interval` | Epochs between target-network syncs |
| `--lr` | `learning_rate` | Adam optimizer learning rate |
| `--gamma` | `gamma` | Discount factor γ |
| `--batch-size` | `batch_size` | Mini-batch size |
| `--replay-buffer-size` | `buffer_size` | Replay buffer capacity |
| `--run-name` | `run_name` | Name prefix for MLflow run and saved model |

---

## 🧠 Batch Training (`src/batch_training.py`)

Run multiple configs in parallel (uses `ProcessPoolExecutor`). All `.json` files in the target directory are executed concurrently.

```bash
# Run all CartPole experiment configs in parallel
uv run src/batch_training.py --env CartPole-v1 --configs-dir test_configs/cartpole/

# Run all BipedalWalker experiment configs
uv run src/batch_training.py --env BipedalWalker-v3 --configs-dir test_configs/bipedal/
```

---

## 📈 Plot Results (`src/plot_results.py`)

Extracts MLflow training metrics and saves `loss.png`, `length.png`, and `reward.png` to `results/<env>/plots/`.

```bash
# Plot the most recent run for CartPole
uv run src/plot_results.py --env CartPole-v1

# Plot a specific run by ID
uv run src/plot_results.py --env CartPole-v1 --run-id <MLFLOW_RUN_ID>
```

---

## 📺 Evaluate a Trained Model (`src/test_model.py`)

Loads saved weights, runs evaluation episodes, and optionally records an MP4/GIF.

```bash
# Record video of a trained CartPole agent
uv run src/test_model.py \
  --env CartPole-v1 \
  --model-path results/CartPole-v1/artifacts/<model>.pt \
  --model-string mlp_small

# Watch live (no video saved)
uv run src/test_model.py --env CartPole-v1 --model-path <path>.pt --model-string mlp_small --interactive
```

---

## 🧪 Tests

```bash
uv run pytest tests/ -v
```

19 tests covering configuration loading, policies, replay memory, metrics tracking,
model evaluation, and recording utilities.

---

## 📖 See also

- [`PARAMETER.md`](PARAMETER.md) — detailed explanation of every hyperparameter
- [`src/env/README.md`](src/env/README.md) — BipedalWalker discretization and reward shaping design
