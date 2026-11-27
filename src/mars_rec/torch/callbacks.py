"""Reusable PyTorch Lightning callbacks for the framework's training loop."""
from __future__ import annotations

import pytorch_lightning as pl


class DynamicNegativeMiningCallback(pl.Callback):
    """Inject the live model into a DNS-style sampler at train start.

    Several negative samplers in :mod:`mars_rec.data.dataset` (notably
    :class:`InteractionsWithDynamicNegativeMiningDataset`) need access to
    the *current* model to score candidate negatives and pick the highest-
    scoring miss. The sampler exposes a ``set_model(model, device)`` method
    for this purpose.

    Without this callback the sampler runs the uniform-fallback path
    silently (the model is never injected, ``self._model`` stays ``None``,
    and every negative is drawn uniformly). The training run looks normal
    but the DNS strategy degenerates to plain random sampling.

    Behaviour: at ``on_train_start`` (and again at ``on_validation_start``
    for safety), traverse the train and val dataloaders, find any wrapped
    Dataset that exposes ``set_model``, and inject the LightningModule's
    underlying recommender plus the trainer's root device.

    Idempotent — re-injecting the same model is a no-op for the sampler,
    but lets us cover both fresh runs and ``trainer.fit(ckpt_path=...)``
    resume scenarios where the dataloaders are rebuilt.
    """

    def _inject(self, dataloader, model, device) -> None:
        if dataloader is None:
            return
        candidates = (
            dataloader if isinstance(dataloader, (list, tuple)) else [dataloader]
        )
        for dl in candidates:
            ds = getattr(dl, "dataset", None)
            if ds is None:
                continue
            setter = getattr(ds, "set_model", None)
            if callable(setter):
                setter(model, device)

    def _resolve_model(self, pl_module: pl.LightningModule):
        # The recommender module is wrapped inside our LightningModule;
        # fall back to the LightningModule itself when the attribute name
        # differs (subclasses).
        for attr in ("recommender_module", "model", "module"):
            m = getattr(pl_module, attr, None)
            if m is not None:
                return m
        return pl_module

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = str(trainer.strategy.root_device) if trainer.strategy.root_device else "cpu"
        model = self._resolve_model(pl_module)
        self._inject(trainer.train_dataloader, model, device)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = str(trainer.strategy.root_device) if trainer.strategy.root_device else "cpu"
        model = self._resolve_model(pl_module)
        self._inject(trainer.val_dataloaders, model, device)


__all__ = ["DynamicNegativeMiningCallback"]
