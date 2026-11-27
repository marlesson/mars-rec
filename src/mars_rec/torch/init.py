"""Weight-initialisation helpers.

Registered in
:data:`mars_rec.simulation.training.TORCH_WEIGHT_INIT` as the
``lecun_normal`` and ``he`` choices; ``xavier_normal`` is the
standard PyTorch initialiser.
"""

import torch
from math import sqrt
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def lecun_normal_init(tensor: torch.Tensor):
    """LeCun-normal init: ``N(0, sqrt(1 / fan_in))``."""
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    nn.init.normal_(tensor, 0, sqrt(1.0 / fan_in))


def he_init(tensor: torch.Tensor):
    """He / Kaiming-normal init with ``fan_in`` (default for ReLU networks)."""
    nn.init.kaiming_normal_(tensor, mode="fan_in")
