"""Tests for ``mars_rec.torch.callbacks``.

The existing ``test_negative_samplers_new.py::TestDNSCallbackAutoAttach``
covers ``_inject``, ``_resolve_model``, and dataloader list-walking in
isolation. This suite covers the Lightning hook integration: at
``on_train_start`` and ``on_validation_start`` the callback must reach
through the trainer to inject the live model into each dataset that
exposes ``set_model``. Without this, DNS-style samplers stay on their
uniform-fallback path and a "DNS run" silently degenerates to random.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from mars_rec.torch.callbacks import DynamicNegativeMiningCallback


class _StubDS:
    """A dataset that records ``set_model`` calls."""

    def __init__(self):
        self.calls = []

    def set_model(self, model, device):
        self.calls.append((model, device))


class _StubDL:
    def __init__(self, ds):
        self.dataset = ds


def _make_trainer(train_dl, val_dl, device="cpu"):
    """Build a MagicMock trainer with the attributes the callback reads."""
    trainer = MagicMock()
    trainer.train_dataloader = train_dl
    trainer.val_dataloaders = val_dl
    trainer.strategy.root_device = device
    return trainer


def _make_lit_module(model_obj):
    """A bare object exposing ``recommender_module`` so ``_resolve_model`` picks it."""
    class _LM:
        pass

    lm = _LM()
    lm.recommender_module = model_obj
    return lm


class TestDNSCallbackHooks:
    def test_on_train_start_injects_into_train_dataloader_dataset(self):
        ds = _StubDS()
        dl = _StubDL(ds)
        trainer = _make_trainer(train_dl=dl, val_dl=None, device="cpu")
        lm = _make_lit_module(model_obj="RECMODEL")

        cb = DynamicNegativeMiningCallback()
        cb.on_train_start(trainer, lm)

        assert ds.calls == [("RECMODEL", "cpu")]

    def test_on_validation_start_injects_into_val_dataloader_dataset(self):
        ds = _StubDS()
        dl = _StubDL(ds)
        trainer = _make_trainer(train_dl=None, val_dl=dl, device="cpu")
        lm = _make_lit_module(model_obj="RECMODEL")

        cb = DynamicNegativeMiningCallback()
        cb.on_validation_start(trainer, lm)

        assert ds.calls == [("RECMODEL", "cpu")]

    def test_on_validation_start_walks_list_of_dataloaders(self):
        """Lightning sometimes passes ``val_dataloaders`` as a list; both must
        receive the injection."""
        ds_a, ds_b = _StubDS(), _StubDS()
        trainer = _make_trainer(
            train_dl=None,
            val_dl=[_StubDL(ds_a), _StubDL(ds_b)],
            device="cuda:0",
        )
        lm = _make_lit_module(model_obj="M")

        cb = DynamicNegativeMiningCallback()
        cb.on_validation_start(trainer, lm)

        assert ds_a.calls == [("M", "cuda:0")]
        assert ds_b.calls == [("M", "cuda:0")]

    def test_on_train_start_is_noop_when_dataset_lacks_set_model(self):
        """Datasets that don't expose ``set_model`` must be silently skipped —
        not raise — so non-DNS projects can keep the callback installed."""
        class _PlainDS:
            pass

        dl = _StubDL(_PlainDS())
        trainer = _make_trainer(train_dl=dl, val_dl=None, device="cpu")
        lm = _make_lit_module(model_obj="ANY")

        cb = DynamicNegativeMiningCallback()
        # The assertion is: no exception is raised.
        cb.on_train_start(trainer, lm)
