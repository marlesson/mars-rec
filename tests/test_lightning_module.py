"""Tests for ``mars_rec.torch.lightning_module``.

Covers the Lightning-side contract that wraps a ``nn.Module`` and routes:
- batch unpacking (single tensor vs tuple of inputs),
- loss application,
- logit→sigmoid conversion for metrics when the loss is logit-based,
- optimizer + scheduler config wiring,
- EmbeddingRecommenderLightningModule's hard-negative pass-through.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mars_rec.torch.lightning_module import (
    EmbeddingRecommenderLightningModule,
    RecommenderLightningModule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityModule(nn.Module):
    """Returns the first input verbatim — used to inspect shared_step inputs."""

    def __init__(self):
        super().__init__()
        # A tiny param so optimizer.step() has something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, *inputs):
        self.last_inputs = inputs
        # Return logits shaped [B] so BCEWithLogitsLoss is happy with [B] target.
        return inputs[0].float() + self.dummy


class _EmbModule(nn.Module):
    """Returns pred_emb directly and exposes get_target_emb."""

    def __init__(self, n_items: int = 8, emb_dim: int = 4, with_hardneg: bool = False):
        super().__init__()
        self.target_emb_table = nn.Embedding(n_items, emb_dim)
        self.emb_dim = emb_dim
        if with_hardneg:
            # K=2 hard negatives per query, indexed by track_id.
            self.register_buffer("hard_neg_indices", torch.randint(0, n_items, (n_items, 2)))

    def forward(self, x):
        # ``x`` is [B] int tensor → return [B, emb_dim] normalised embedding.
        return nn.functional.normalize(self.target_emb_table(x), dim=-1)

    def get_target_emb(self, track_ids):
        return nn.functional.normalize(self.target_emb_table(track_ids), dim=-1)


# ---------------------------------------------------------------------------
# Tests — RecommenderLightningModule
# ---------------------------------------------------------------------------


class TestSharedStep:
    def test_shared_step_returns_scalar_loss(self):
        module = _IdentityModule()
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        lit = RecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_names=["mse"],
            loss_name="mse",
        )
        # batch = (inputs, targets) — both [B] float for MSE
        inputs = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        loss = lit._shared_step((inputs, targets), stage="train")
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad

    def test_shared_step_unpacks_tuple_inputs(self):
        module = _IdentityModule()
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        lit = RecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_names=[],
            loss_name="mse",
        )
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([5.0, 6.0])
        targets = torch.tensor([1.0, 2.0])
        lit._shared_step(((a, b), targets), stage="train")
        # _IdentityModule stored ``inputs`` from forward — assert both passed.
        assert len(module.last_inputs) == 2
        assert torch.equal(module.last_inputs[0], a)
        assert torch.equal(module.last_inputs[1], b)

    def test_logit_loss_triggers_sigmoid_for_metrics(self):
        """When loss_name is logit-based, metrics receive sigmoid'd preds.

        Set ``loss_name="bce_logits"`` and assert the metric callable saw
        values in [0, 1] even though preds were raw [-inf, inf]-ish logits.
        """
        module = _IdentityModule()
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)

        # Spy: capture preds the metric receives.
        captured = {}

        def fake_metric(preds, targets):
            captured["preds"] = preds.clone()
            return torch.tensor(0.5)

        # Patch METRIC_REGISTRY to expose our spy under a real-looking name.
        from mars_rec.torch import lightning_module as lm

        lm.METRIC_REGISTRY["__spy__"] = fake_metric
        try:
            lit = RecommenderLightningModule(
                module=module,
                optimizer=opt,
                loss_fn=loss_fn,
                metric_names=["__spy__"],
                loss_name="bce_logits",
            )
            # Manual logits well outside [0, 1]:
            inputs = torch.tensor([10.0, -10.0, 0.0])
            targets = torch.tensor([1.0, 0.0, 1.0])
            # Patch self.log so Lightning doesn't complain about no trainer.
            lit.log = lambda *a, **kw: None
            lit._shared_step((inputs, targets), stage="train")
            assert "preds" in captured
            preds = captured["preds"]
            # sigmoid(10) ≈ 1.0, sigmoid(-10) ≈ 0.0, sigmoid(0) = 0.5
            assert (preds >= 0).all() and (preds <= 1).all()
        finally:
            lm.METRIC_REGISTRY.pop("__spy__", None)

    def test_non_logit_loss_passes_preds_as_is(self):
        """For non-logit losses (e.g. ``mse``), preds reach the metric unchanged."""
        module = _IdentityModule()
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        captured = {}

        def fake_metric(preds, targets):
            captured["preds"] = preds.clone()
            return torch.tensor(0.0)

        from mars_rec.torch import lightning_module as lm

        lm.METRIC_REGISTRY["__spy2__"] = fake_metric
        try:
            lit = RecommenderLightningModule(
                module=module,
                optimizer=opt,
                loss_fn=loss_fn,
                metric_names=["__spy2__"],
                loss_name="mse",  # not in _LOGIT_LOSSES
            )
            lit.log = lambda *a, **kw: None
            inputs = torch.tensor([7.0, -3.0])
            targets = torch.tensor([7.0, -3.0])
            lit._shared_step((inputs, targets), stage="val")
            # preds = inputs (Identity module) → identical to inputs, no sigmoid.
            assert torch.equal(captured["preds"], inputs)
        finally:
            lm.METRIC_REGISTRY.pop("__spy2__", None)


class TestConfigureOptimizers:
    def test_returns_optimizer_only_when_no_scheduler(self):
        module = _IdentityModule()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        lit = RecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=nn.MSELoss(),
            metric_names=[],
            loss_name="mse",
        )
        out = lit.configure_optimizers()
        assert out is opt

    def test_returns_optimizer_and_scheduler_dict_when_configured(self):
        module = _IdentityModule()
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        scheduler_cfg = {"scheduler": sched, "interval": "epoch"}
        lit = RecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=nn.MSELoss(),
            metric_names=[],
            loss_name="mse",
            lr_scheduler_config=scheduler_cfg,
        )
        out = lit.configure_optimizers()
        assert isinstance(out, dict)
        assert out["optimizer"] is opt
        assert out["lr_scheduler"] is scheduler_cfg


# ---------------------------------------------------------------------------
# Tests — EmbeddingRecommenderLightningModule (hard negatives)
# ---------------------------------------------------------------------------


class TestEmbeddingLightningModule:
    def test_passes_hard_negs_when_loss_accepts_three_args(self):
        """If the module exposes ``hard_neg_indices`` and the loss accepts a
        third positional arg, hard_neg_embs must be forwarded (not None)."""
        n_items, emb_dim = 8, 4
        module = _EmbModule(n_items=n_items, emb_dim=emb_dim, with_hardneg=True)

        received = {}

        def loss_fn(pred, target, hard_neg=None):
            received["hard_neg_shape"] = None if hard_neg is None else tuple(hard_neg.shape)
            # any scalar loss for backward; cosine sim is fine
            return (pred * target).sum()

        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        lit = EmbeddingRecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_names=[],
            loss_name="infonce",
        )
        lit.log = lambda *a, **kw: None
        # batch: (inputs, track_ids)
        track_ids = torch.tensor([2, 3, 5])
        inputs = track_ids  # module.forward(x) takes idx → emb
        lit._shared_step((inputs, track_ids), stage="train")
        # K=2, B=3, emb_dim=4 → expected [3, 2, 4]
        assert received["hard_neg_shape"] == (3, 2, 4)

    def test_falls_back_when_loss_rejects_third_arg(self):
        """If loss_fn signature only accepts (pred, target), the try/except
        must swallow the TypeError and call loss_fn with 2 args instead."""
        module = _EmbModule(with_hardneg=True)

        call_arities = []

        def loss_fn(pred, target):
            call_arities.append(2)
            return (pred * target).sum()

        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        lit = EmbeddingRecommenderLightningModule(
            module=module,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_names=[],
            loss_name="infonce",
        )
        lit.log = lambda *a, **kw: None
        track_ids = torch.tensor([1, 2])
        lit._shared_step((track_ids, track_ids), stage="train")
        assert call_arities == [2]


class TestFirstBatchLog:
    def test_first_training_batch_shapes_go_to_the_logger_once(self, caplog, capsys, monkeypatch):
        module = _IdentityModule()
        lit = RecommenderLightningModule(
            module=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=0.01),
            loss_fn=nn.MSELoss(),
            metric_names=[],
            loss_name="mse",
        )
        monkeypatch.setattr(RecommenderLightningModule, "_step_logged", False)
        monkeypatch.setattr(lit, "_shared_step", lambda batch, stage: torch.tensor(0.0))
        batch = ((torch.zeros(4, 3),), torch.zeros(4))

        with caplog.at_level("INFO", logger="mars_rec.torch.lightning_module"):
            lit.training_step(batch, 0)
            lit.training_step(batch, 1)

        messages = [r.getMessage() for r in caplog.records if r.name == "mars_rec.torch.lightning_module"]
        assert len(messages) == 1
        assert "input_shapes=[torch.Size([4, 3])]" in messages[0]
        assert "[DIAG]" not in capsys.readouterr().out  # no stray print to stdout
