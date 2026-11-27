"""Cross-process GPU slot allocator.

When several training tasks run in parallel on a multi-GPU box
(Luigi local scheduler or a CI matrix) they need to coordinate so
each grabs a different ``cuda:N`` device. :class:`CudaRepository`
implements the simplest possible solution: a disk-backed Deque
(``diskcache``) treated as a pool of free device ids — workers
``pop`` to acquire and ``append`` to release.
"""

import os

import torch

from diskcache.persistent import Deque


class CudaRepository(object):
    """Disk-backed pool of available CUDA device ids."""

    _avaliable_devices: Deque = Deque(directory=os.path.join("output", "cuda_devices"))

    @classmethod
    def fill(cls):
        """Reset the pool to ``[0, 1, …, torch.cuda.device_count() - 1]``."""
        CudaRepository._avaliable_devices.clear()
        CudaRepository._avaliable_devices.extend([i for i in range(torch.cuda.device_count())])

    @classmethod
    def get_avaliable_device(self) -> int:
        """Acquire a free device id, or fall back to ``0`` when the pool is empty."""
        try:
            return CudaRepository._avaliable_devices.pop()
        except IndexError:
            return 0

    @classmethod
    def put_available_device(self, device: int):
        """Release a previously acquired device id back into the pool."""
        CudaRepository._avaliable_devices.append(device)
