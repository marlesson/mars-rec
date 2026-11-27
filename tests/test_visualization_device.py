"""``trace_model_shapes`` must build its synthetic batch on the model's device.

Training calls the tracer with ``device=self.device`` *before* Lightning
moves the model to the accelerator. With ``device="cuda"`` the model was
still on the CPU, every trace failed with "Expected all tensors to be on the
same device" and GPU runs wrote an ``architecture.svg`` without shapes.
CI has no GPU, so the mismatch is reproduced with the ``meta`` device.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mars_rec.model.base_model import LogisticRegression
from mars_rec.visualization import _module_device, trace_model_shapes


class _DeviceRecorder(nn.Module):
    """One parameter on the CPU; records the device its inputs arrive on."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_device = None

    def forward(self, user_ids, item_ids):
        self.seen_device = user_ids.device
        return user_ids.float()


def test_synthetic_batch_is_built_on_the_model_device(minimal_project_config, index_mapping_small):
    # Requested "meta" while the model lives on the CPU: the same mismatch as
    # device="cuda" with a not-yet-moved model during training.
    model = _DeviceRecorder()
    trace_model_shapes(model, minimal_project_config, index_mapping_small, device="meta")
    assert model.seen_device == torch.device("cpu")


def test_real_model_traces_without_error(minimal_project_config, index_mapping_small):
    model = LogisticRegression(minimal_project_config, index_mapping_small, n_factors=4)
    trace = trace_model_shapes(model, minimal_project_config, index_mapping_small, device="meta")
    assert trace["error"] is None
    assert trace["order"], "no module was traced"
    assert all(p.device.type == "cpu" for p in model.parameters())  # model untouched


def test_requested_device_is_used_for_a_model_without_parameters(minimal_project_config, index_mapping_small):
    class _NoParams(nn.Module):
        def forward(self, user_ids, item_ids):
            self.seen_device = user_ids.device
            return user_ids.float()

    model = _NoParams()
    assert _module_device(model) is None
    trace = trace_model_shapes(model, minimal_project_config, index_mapping_small, device="cpu")
    assert trace["error"] is None
    assert model.seen_device == torch.device("cpu")


def test_module_device_falls_back_to_buffers():
    model = nn.Module()
    model.register_buffer("scale", torch.ones(1))
    assert _module_device(model) == torch.device("cpu")
