"""PyTorch Lightning wrappers around :class:`RecommenderModule`.

Two flavours:

- :class:`RecommenderLightningModule` — the default. Wraps any
  ``RecommenderModule`` whose ``forward`` returns per-row scores and
  whose loss compares those scores against the target.
- :class:`EmbeddingRecommenderLightningModule` — for two-tower /
  retrieval models that emit a query embedding and compare it to a
  precomputed target embedding (used with ``InfoNCELoss`` /
  ``CosineSimilarityLoss`` / ``HardNegInfoNCELoss``).

The training task picks the right wrapper automatically based on the
configured ``loss_function`` (see ``_EMBEDDING_LOSSES`` in
:mod:`mars_rec.simulation.training`).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from mars_rec.torch.metrics import METRIC_REGISTRY


_LOGIT_LOSSES = {
    "bce_logits", "bce_logits_smooth", "bpr", "focal_loss",
    "ranknet", "lambdarank",
}


class RecommenderLightningModule(pl.LightningModule):
    """Wraps a RecommenderModule (nn.Module) for use with PyTorch Lightning."""

    def __init__(
        self,
        module: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        metric_names: list,
        loss_name: str = "",
        lr_scheduler_config: Optional[dict] = None,
    ):
        super().__init__()
        self.module = module
        self._optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_names = [m for m in metric_names if m != "loss"]
        # When loss is logit-based (BCEWithLogits/Focal), metrics that expect
        # probabilities ([0,1]) need sigmoid applied to preds first.
        self._sigmoid_for_metrics = loss_name in _LOGIT_LOSSES
        # Lightning-ready scheduler dict (``{"scheduler": ..., "monitor": ...,
        # "interval": ...}``) or ``None`` for legacy "no scheduler".
        self._lr_scheduler_config = lr_scheduler_config

    def forward(self, *args, **kwargs):
        """Delegate to the wrapped :class:`RecommenderModule`."""
        return self.module(*args, **kwargs)

    def _shared_step(self, batch, stage: str):
        inputs, targets = batch

        if isinstance(inputs, (list, tuple)):
            preds = self.module(*inputs)
        else:
            preds = self.module(inputs)

        if isinstance(targets, (list, tuple)):
            loss_target = targets[0]
        else:
            loss_target = targets

        loss = self.loss_fn(preds, loss_target)

        # ---- Optional auxiliary loss hook (e.g. InfoNCE alignment) ----
        # The wrapped module can opt-in by exposing both:
        #   - ``aux_loss_lambda`` (float > 0)
        #   - ``compute_aux_loss(inputs, targets)`` returning a scalar tensor
        # The aux is added to the TRAIN loss only — val_loss stays comparable
        # to the lambda=0 baseline so early stopping and cross-model
        # comparisons remain meaningful.
        aux_lambda = float(getattr(self.module, "aux_loss_lambda", 0.0) or 0.0)
        aux_loss_fn = getattr(self.module, "compute_aux_loss", None)
        if aux_lambda > 0.0 and aux_loss_fn is not None:
            aux_in = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
            aux = aux_loss_fn(aux_in, loss_target)
            if aux is not None and torch.is_tensor(aux) and torch.isfinite(aux).all():
                if stage == "train":
                    loss = loss + aux_lambda * aux
                self.log(
                    f"val_aux_loss" if stage == "val" else "aux_loss",
                    aux.detach(),
                    prog_bar=False, on_step=False, on_epoch=True,
                )

        # Log loss
        loss_key = "val_loss" if stage == "val" else "loss"
        self.log(loss_key, loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute and log custom metrics
        with torch.no_grad():
            metric_preds = preds.detach()
            if self._sigmoid_for_metrics:
                metric_preds = torch.sigmoid(metric_preds)
            for name in self.metric_names:
                metric_fn = METRIC_REGISTRY.get(name)
                if metric_fn is not None:
                    val = metric_fn(metric_preds, loss_target.detach())
                    log_name = f"val_{name}" if stage == "val" else name
                    self.log(log_name, val, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    _step_logged = False

    def training_step(self, batch, batch_idx):
        """Lightning hook — forwards to :meth:`_shared_step` with ``stage="train"``."""
        if not RecommenderLightningModule._step_logged:
            RecommenderLightningModule._step_logged = True
            inputs, targets = batch
            shapes = [getattr(x, 'shape', '?') for x in (inputs if isinstance(inputs, (list, tuple)) else [inputs])]
            # Targets may be a tensor (labels) or a list/tuple (e.g. DNS
            # returns ``(labels, hard_negatives)``). Be defensive — the diag
            # is informational and must never crash the step.
            if isinstance(targets, (list, tuple)):
                target_shape = [getattr(t, 'shape', '?') for t in targets]
            else:
                target_shape = getattr(targets, 'shape', '?')
            print(
                f"[DIAG] training_step batch_idx=0 input_shapes={shapes} "
                f"target_shape={target_shape}",
                flush=True,
            )
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        metrics = {k: f"{v:.4f}" for k, v in self.trainer.callback_metrics.items() if isinstance(v, (int, float)) or hasattr(v, 'item')}
        print(f"[EPOCH] {self.current_epoch} metrics={metrics}", flush=True)

    def validation_step(self, batch, batch_idx):
        """Lightning hook — forwards to :meth:`_shared_step` with ``stage="val"``."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Return the optimiser (and scheduler dict, when configured)."""
        if self._lr_scheduler_config is None:
            return self._optimizer
        return {"optimizer": self._optimizer, "lr_scheduler": self._lr_scheduler_config}


class EmbeddingRecommenderLightningModule(RecommenderLightningModule):
    """Lightning module for embedding-output models (``pred_emb`` vs ``target_emb``).

    Expects:
    - ``module.forward(*inputs)`` → ``pred_emb [B, emb_dim]``, L2-normalised.
    - ``module.get_target_emb(track_ids)`` → ``target_emb [B, emb_dim]``, L2-normalised.
    - ``loss_fn(pred_emb, target_emb)`` → scalar (e.g. ``InfoNCELoss``).
    - ``output_column=Column("track_id", IOType.INDEXABLE)`` so ``targets`` is ``[B]`` int.
    """

    def _shared_step(self, batch, stage: str):
        inputs, track_ids = batch

        if isinstance(inputs, (list, tuple)):
            pred_emb = self.module(*inputs)
        else:
            pred_emb = self.module(inputs)

        if isinstance(track_ids, (list, tuple)):
            track_ids = track_ids[0]

        target_emb = self.module.get_target_emb(track_ids)

        # V6 — Hard negatives: pass per-query CF-BPR top-K neighbors when
        # (a) the model exposes a precomputed ``hard_neg_indices`` buffer and
        # (b) the loss accepts a third arg.  Falls back to standard
        # ``loss_fn(pred, target)`` otherwise.
        hard_neg_embs = None
        hn = getattr(self.module, "hard_neg_indices", None)
        if hn is not None:
            B = track_ids.size(0)
            hard_ids = hn[track_ids]  # [B, K]
            K = hard_ids.size(1)
            hard_emb_flat = self.module.get_target_emb(hard_ids.reshape(-1))
            hard_neg_embs = hard_emb_flat.view(B, K, -1)

        try:
            loss = self.loss_fn(pred_emb, target_emb, hard_neg_embs)
        except TypeError:
            loss = self.loss_fn(pred_emb, target_emb)

        loss_key = "val_loss" if stage == "val" else "loss"
        self.log(loss_key, loss, prog_bar=True, on_step=False, on_epoch=True)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(pred_emb.float(), target_emb.float(), dim=-1).mean()
            log_name = "val_avg_cos_sim" if stage == "val" else "avg_cos_sim"
            self.log(log_name, cos_sim, prog_bar=True, on_step=False, on_epoch=True)

        return loss
