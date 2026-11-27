"""Keras-style model summary printer.

Called by :class:`SupervisedModelTraining` after model construction
to write the per-layer parameter / shape table into ``summary.txt``
under the run's output directory.
"""

import numpy as np
import torch
from collections import OrderedDict


def summary(model, x, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)

    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = OrderedDict()
            info["id"] = id(module)
            try:
                if isinstance(outputs, (list, tuple)):
                    info["out"] = list(outputs[0].size())
                else:
                    info["out"] = list(outputs.size())
            except RuntimeError:
                info["out"] = "nested"

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement()

                if "weight" == name:
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]
        model(*input_params) if not (kwargs or args) else model(*input_params, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    if was_training:
        model.train()

    header_fmt = "{:<24} {:>20} {:>20} {:>14} {:>14}"
    row_fmt = "{:<24} {:>20} {:>20} {:>14} {:>14}"
    width = 24 + 1 + 20 + 1 + 20 + 1 + 14 + 1 + 14  # 96

    print("=" * width)
    print(
        header_fmt.format(
            "Layer",
            "Kernel Shape",
            "Output Shape",
            "# Params (K)",
            "# Mult-Adds (M)",
        )
    )
    print("-" * width)

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_ksize = str(info["ksize"])
        repr_out = str(info["out"])
        repr_params = info["params"]
        repr_macs = info["macs"]

        if isinstance(repr_params, (int, float)):
            total_params += repr_params
            repr_params = "{0:,.2f}".format(repr_params / 1000)
        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs
            repr_macs = "{0:,.2f}".format(repr_macs / 1000000)

        print(row_fmt.format(layer, repr_ksize, repr_out, repr_params, repr_macs))

        for inner_name, inner_shape in info["inner"].items():
            print("  {:<22} {:>20}".format(inner_name, str(inner_shape)))

    print("-" * width)
    print("Total params:    {0:,.2f}K".format(total_params / 1000))
    print("Total Mult-Adds: {0:,.2f}M".format(total_macs / 1000000))
    print("=" * width)
