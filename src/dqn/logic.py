"""
dqn/logic.py
Shared mathematical and logic utilities.
"""
from __future__ import annotations


def linear_anneal(start: float, end: float, current_step: int | None, max_steps: int | None) -> float:
    """
    Linearly anneals a value from start to end based on training progress.

    Args:
        start: Initial value (at step 0).
        end: Final value (at max_steps).
        current_step: Current training step.
        max_steps: Total training steps at which end is reached.

    Returns:
        float: Interpolated value between start and end.
    """
    if current_step is None or max_steps is None or max_steps <= 0:
        return start

    progress = min(max(current_step, 0), max_steps) / float(max_steps)
    return start + progress * (end - start)
