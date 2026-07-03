"""Store and sample DQN transitions in a bounded replay memory."""
from __future__ import annotations

import numpy as np
import torch


class ExperienceReplay:
    """Maintain a tensor-backed circular buffer on CPU for DQN experience batches."""

    def __init__(self, capacity: int, state_shape: tuple[int, ...], device: torch.device) -> None:
        """Allocate transition storage and configure the sampling device."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not state_shape:
            raise ValueError("state_shape must contain at least one dimension")

        self.capacity = int(capacity)
        self.state_shape = tuple(int(dimension) for dimension in state_shape)
        self.device = device
        self._write_index = 0
        self._size = 0
        self._pinned = device.type == "cuda" and torch.cuda.is_available()

        tensor_options = {"device": "cpu", "pin_memory": self._pinned}
        self._observations = torch.empty((self.capacity, *self.state_shape), dtype=torch.float32, **tensor_options)
        self._next_observations = torch.empty_like(self._observations)
        self._actions = torch.empty(self.capacity, dtype=torch.int64, **tensor_options)
        self._rewards = torch.empty(self.capacity, dtype=torch.float32, **tensor_options)
        self._terminals = torch.empty(self.capacity, dtype=torch.bool, **tensor_options)
        self._host_sample: tuple[torch.Tensor, ...] | None = None
        self._device_sample: tuple[torch.Tensor, ...] | None = None

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> None:
        """Insert a vectorized transition batch and overwrite the oldest entries as needed."""
        batch = self._as_cpu_tensors(observations, actions, rewards, next_observations, terminals)
        batch_size = batch[0].shape[0]
        self._validate_batch(batch, batch_size)

        first_length = min(batch_size, self.capacity - self._write_index)
        self._copy_range(batch, source_start=0, destination_start=self._write_index, length=first_length)

        remaining = batch_size - first_length
        if remaining:
            self._copy_range(batch, source_start=first_length, destination_start=0, length=remaining)

        self._write_index = (self._write_index + batch_size) % self.capacity
        self._size = min(self.capacity, self._size + batch_size)

    def sample(self, count: int) -> tuple[torch.Tensor, ...]:
        """Sample transitions with replacement and move the resulting batch to the configured device."""
        if count <= 0:
            raise ValueError("sample count must be positive")
        if count > self._size:
            raise ValueError(f"sample count {count} exceeds {self._size} available transitions")

        indices = torch.randint(self._size, (count,))
        host_sample, device_sample = self._sample_buffers(count)
        stores = (
            self._observations,
            self._actions,
            self._rewards,
            self._next_observations,
            self._terminals,
        )
        for store, destination in zip(stores, host_sample):
            torch.index_select(store, 0, indices, out=destination)

        if device_sample is host_sample:
            return host_sample

        for source, destination in zip(host_sample, device_sample):
            destination.copy_(source, non_blocking=self._pinned)
        return device_sample

    def __len__(self) -> int:
        """Return the number of transitions currently available for sampling."""
        return self._size

    @staticmethod
    def _as_cpu_tensors(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        """Convert one input batch to the canonical replay-memory tensor types."""
        return (
            torch.as_tensor(observations, dtype=torch.float32, device="cpu"),
            torch.as_tensor(actions, dtype=torch.int64, device="cpu"),
            torch.as_tensor(rewards, dtype=torch.float32, device="cpu"),
            torch.as_tensor(next_observations, dtype=torch.float32, device="cpu"),
            torch.as_tensor(terminals, dtype=torch.bool, device="cpu"),
        )

    def _validate_batch(self, batch: tuple[torch.Tensor, ...], batch_size: int) -> None:
        """Reject malformed transition batches before mutating replay storage."""
        observations, actions, rewards, next_observations, terminals = batch
        if batch_size == 0:
            raise ValueError("transition batch must not be empty")
        if batch_size > self.capacity:
            raise ValueError(f"transition batch size {batch_size} exceeds capacity {self.capacity}")
        if tuple(observations.shape[1:]) != self.state_shape:
            raise ValueError(f"observation shape must be {self.state_shape}")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations")
        if any(tensor.shape != (batch_size,) for tensor in (actions, rewards, terminals)):
            raise ValueError("actions, rewards, and terminals must be one-dimensional transition batches")

    def _copy_range(
        self,
        batch: tuple[torch.Tensor, ...],
        source_start: int,
        destination_start: int,
        length: int,
    ) -> None:
        """Copy a contiguous portion of a transition batch into replay storage."""
        if length == 0:
            return

        source = slice(source_start, source_start + length)
        destination = slice(destination_start, destination_start + length)
        stores = (
            self._observations,
            self._actions,
            self._rewards,
            self._next_observations,
            self._terminals,
        )
        for store, values in zip(stores, batch):
            store[destination].copy_(values[source])

    def _sample_buffers(self, count: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Allocate reusable host and device batches for one sampling size."""
        if self._host_sample is None or self._host_sample[0].shape[0] != count:
            host_options = {"device": "cpu", "pin_memory": self._pinned}
            self._host_sample = (
                torch.empty((count, *self.state_shape), dtype=torch.float32, **host_options),
                torch.empty(count, dtype=torch.int64, **host_options),
                torch.empty(count, dtype=torch.float32, **host_options),
                torch.empty((count, *self.state_shape), dtype=torch.float32, **host_options),
                torch.empty(count, dtype=torch.bool, **host_options),
            )
            if self.device.type == "cpu":
                self._device_sample = self._host_sample
            else:
                self._device_sample = tuple(
                    torch.empty(tensor.shape, dtype=tensor.dtype, device=self.device)
                    for tensor in self._host_sample
                )

        return self._host_sample, self._device_sample
