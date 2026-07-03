# Environment Wrappers (`src/env/`)

This directory contains wrappers that modify the default behavior of standard Gymnasium environments to make them suitable for our specific DQN agent.

## Why is there an `__init__.py` file here?
Just like the core `dqn/` folder, this file tells Python that the `env/` directory is a package, allowing us to cleanly import the wrappers in our training scripts using `from env.bipedal_wrapper import DiscreteBipedalWrapper`.

## Wrappers
* **`bipedal_wrapper.py`**: 
  1. **Discretization**: Standard DQN only works with discrete action spaces (like pushing left or right). However, `BipedalWalker-v3` has continuous motor joints. This wrapper discretizes the joints into `bins=2` so our DQN agent can control it.
  2. **Reward Shaping**: BipedalWalker has a known "loophole" where agents learn to glide/skate on one leg rather than walking. This wrapper uses an Exponential Moving Average (EMA) to track ground contact for both legs, severely penalizing the agent if it tries to skate asymmetrically.
