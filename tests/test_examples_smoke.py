"""Smoke tests for the reference projects under ``examples/``.

The examples are documentation that runs — but only against public datasets
that are too big to download in CI, so nothing used to catch them drifting
out of sync with the framework (``forward()`` argument order, output shape
vs. loss, ``ProjectConfig`` changes, renamed params).

Each test:

1. executes the example's own ``run_training.py`` as ``__main__`` with
   ``SupervisedModelTraining`` swapped for a stub that records the kwargs —
   so the test uses **exactly** the params the script ships with;
2. resolves the ``project`` and ``recommender_module_class`` dotted paths;
3. builds a synthetic index mapping + batch from ``project.input_columns``;
4. runs forward + the configured loss + backward, checking the output
   shape contract from AGENTS.md (``[batch]`` for BCE-family losses).

No data is downloaded or read.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.meta_config import IOType, ProjectConfig
from mars_rec.model.abstract import RecommenderModule
from mars_rec.utils.index_mapping import create_index_mapping
from mars_rec.utils.reflection import load_attr

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(p.name for p in EXAMPLES_DIR.iterdir() if (p / "run_training.py").is_file())

BATCH = 6
VOCAB = 20  # synthetic distinct values per INDEXABLE column
POINTWISE_LOSSES = {"bce", "bce_logits", "mse", "focal_loss", "implicit_feedback_bce", "dummy"}


class _Captured(Exception):
    pass


def _captured_training_kwargs(example: str) -> dict:
    """Run ``examples/<example>/run_training.py`` and return the kwargs it
    passes to ``SupervisedModelTraining`` (training itself never starts)."""
    import mars_rec.simulation.training as training

    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        raise _Captured

    original = training.SupervisedModelTraining
    training.SupervisedModelTraining = _stub
    try:
        with pytest.raises(_Captured):
            runpy.run_path(str(EXAMPLES_DIR / example / "run_training.py"), run_name="__main__")
    finally:
        training.SupervisedModelTraining = original
    return captured


def _array_len(column, extra_params: dict) -> int:
    """Length of a per-row array input, taken from the model's own params
    when it declares one (history length / embedding dim)."""
    if column.type == IOType.INDEXABLE_ARRAY:
        return int(extra_params.get("history_len", 5))
    for key in ("text_emb_dim", "genre_dim", "emb_dim"):
        if key in extra_params:
            return int(extra_params[key])
    return 5


def _synthetic_inputs(config: ProjectConfig, extra_params: dict):
    index_mapping = {}
    for column in [config.user_column, config.item_column, *config.other_input_columns, config.output_column]:
        if column.type in (IOType.INDEXABLE, IOType.INDEXABLE_ARRAY):
            key = column.same_index_as or column.name
            index_mapping.setdefault(key, create_index_mapping(f"{key}_{i}" for i in range(VOCAB)))
            index_mapping[column.name] = index_mapping[key]

    gen = torch.Generator().manual_seed(0)
    low, high = 3, VOCAB + 3  # real ids start after the 3 reserved slots
    inputs = []
    for column in config.input_columns:
        if column.type == IOType.INDEXABLE:
            inputs.append(torch.randint(low, high, (BATCH,), generator=gen))
        elif column.type == IOType.INDEXABLE_ARRAY:
            inputs.append(torch.randint(low, high, (BATCH, _array_len(column, extra_params)), generator=gen))
        elif column.type == IOType.FLOAT_ARRAY:
            inputs.append(torch.rand(BATCH, _array_len(column, extra_params), generator=gen))
        elif column.type == IOType.NUMBER:
            inputs.append(torch.rand(BATCH, generator=gen))
        else:  # pragma: no cover - surfaces new IOTypes in examples
            pytest.fail(f"smoke test does not know how to synthesise {column.type} ({column.name})")
    return index_mapping, inputs


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_trains_one_step(example):
    from mars_rec.simulation.training import TORCH_LOSS_FUNCTIONS

    kwargs = _captured_training_kwargs(example)
    assert kwargs["project"].startswith(f"examples.{example}."), kwargs["project"]

    config = load_attr(kwargs["project"], ProjectConfig)
    assert issubclass(config.prepare_data_frames_task, BasePrepareDataFrames)

    module_cls = load_attr(kwargs["recommender_module_class"], nn.Module)
    assert issubclass(module_cls, RecommenderModule)

    extra = kwargs.get("recommender_extra_params", {})
    index_mapping, inputs = _synthetic_inputs(config, extra)

    torch.manual_seed(0)
    model = module_cls(project_config=config, index_mapping=index_mapping, **extra)
    model.train()
    output = model(*inputs)

    loss_name = kwargs.get("loss_function", "mse")
    if loss_name in POINTWISE_LOSSES:
        assert output.shape == (BATCH,), f"{loss_name} expects [batch] output, got {tuple(output.shape)}"
        target = torch.randint(0, 2, (BATCH,)).float()
    else:  # ce → [batch, n_items]
        assert output.dim() == 2 and output.shape[0] == BATCH
        target = torch.randint(0, output.shape[1], (BATCH,))

    loss = TORCH_LOSS_FUNCTIONS[loss_name]()(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reached the parameters"


def test_all_examples_are_covered():
    """Guard against a new example silently skipping the smoke test."""
    expected = {"movielens", "lastfm", "online_retail", "amazon_beauty", "movielens100k_slate"}
    assert expected <= set(EXAMPLES)
