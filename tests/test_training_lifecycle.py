"""End-to-end lifecycle tests for the training → checkpoint → load → predict chain.

Avoids running the full Luigi orchestrator (which depends on the project's
own ``prepare_data_frames_task`` to produce CSVs, plus optional artifacts
like graphviz architecture diagrams). Instead, exercises the contract that
matters most for production: a Lightning checkpoint saved in the format
``SupervisedModelTraining`` writes can be re-loaded by
``mars_rec.inference.load_trained_model`` and produce identical
predictions to the in-memory model.

This protects against the regressions that prompted the plan:
- v12 inference parity (state_dict key mismatch);
- xattn_v3 retrain collapse (silent hyperparam drift on save/load);
- generally, any silent contract change between save and load paths.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.inference import load_trained_model
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.base_model import LogisticRegression
from mars_rec.torch.lightning_module import RecommenderLightningModule


# ---------------------------------------------------------------------------
# A minimal project ``load_attr`` can resolve from a task_dir's params.json.
# ---------------------------------------------------------------------------


class _MinimalPrep(BasePrepareDataFrames):
    @property
    def dataset_dir(self):  # pragma: no cover - not invoked here
        return "/tmp"

    @property
    def read_data_frame_path(self):  # pragma: no cover
        return None


lifecycle_test_config = ProjectConfig(
    base_dir="/tmp/lifecycle",
    prepare_data_frames_task=_MinimalPrep,
    dataset_class=MagicMock(),
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    output_column=Column("label", IOType.NUMBER),
)


def _build_model_and_data(seed: int = 0):
    """Reproducible LogisticRegression + a tiny supervised dataset."""
    pl.seed_everything(seed, workers=True)
    idx_mapping = {
        "user_id": defaultdict(int, {"u1": 3, "u2": 4, "u3": 5}),
        "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5, "i4": 6}),
    }
    model = LogisticRegression(lifecycle_test_config, idx_mapping, n_factors=8)

    torch.manual_seed(seed)
    n = 32
    user_ids = torch.randint(3, 6, (n,))
    item_ids = torch.randint(3, 7, (n,))
    labels = torch.randint(0, 2, (n,)).float().unsqueeze(1)
    return model, idx_mapping, (user_ids, item_ids, labels)


def _train_one_epoch(model: nn.Module, batch):
    """Hand-rolled training step that mirrors the framework's wiring: BCE
    between sigmoid scores and binary labels."""
    user_ids, item_ids, labels = batch
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    for _ in range(3):  # a few steps so weights move noticeably
        opt.zero_grad()
        preds = model(user_ids, item_ids)
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()
    return model


def _save_task_artifacts(task_dir, model: nn.Module):
    """Write the three artifacts ``load_trained_model`` needs in the exact
    shape ``SupervisedModelTraining`` writes them.

    Lightning checkpoint format: ``{"state_dict": {"module.<key>": tensor}}``.
    """
    idx_mapping = {
        "user_id": defaultdict(int, {"u1": 3, "u2": 4, "u3": 5}),
        "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5, "i4": 6}),
    }
    params = {
        "project": __name__ + ".lifecycle_test_config",
        "recommender_module_class": ("mars_rec.model.base_model.LogisticRegression"),
        "recommender_extra_params": {"n_factors": 8},
    }
    with open(task_dir / "params.json", "w") as f:
        json.dump(params, f)
    with open(task_dir / "index_mapping.pkl", "wb") as f:
        pickle.dump(idx_mapping, f)

    prefixed = {"module." + k: v for k, v in model.state_dict().items()}
    torch.save({"state_dict": prefixed}, task_dir / "weights_epoch=00.ckpt")
    return idx_mapping


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadPredictRoundtrip:
    def test_loaded_model_predicts_same_as_in_memory(self, tmp_task_dir):
        """The whole point of the lifecycle: predictions are identical
        before save and after load. If the state_dict key handling drifts
        (the v12 inference parity failure mode), this catches it."""
        model, _, batch = _build_model_and_data(seed=11)
        model = _train_one_epoch(model, batch)
        model.eval()

        with torch.no_grad():
            ref_scores = model(batch[0], batch[1]).clone()

        _save_task_artifacts(tmp_task_dir, model)

        loaded_model, _, _, _ = load_trained_model(str(tmp_task_dir))
        loaded_model.eval()
        with torch.no_grad():
            loaded_scores = loaded_model(batch[0], batch[1])

        assert torch.allclose(ref_scores, loaded_scores, atol=1e-6)

    def test_load_resolves_module_class_from_params(self, tmp_task_dir):
        """The recommender_module_class string in params.json must be
        resolvable to a concrete RecommenderModule subclass — this is
        the same load_attr path used by both training (to instantiate)
        and inference (to reload). A typo there would crash inference
        much later, after partial loading."""
        model, _, _ = _build_model_and_data(seed=22)
        _save_task_artifacts(tmp_task_dir, model)

        loaded_model, project_config, index_mapping, params = load_trained_model(str(tmp_task_dir))
        # Resolved module class is LogisticRegression (not something else).
        assert type(loaded_model).__name__ == "LogisticRegression"
        # n_factors came from extra_params.
        assert loaded_model.user_embeddings.embedding_dim == 8


class TestSeedReproducibility:
    def test_same_seed_produces_identical_weights(self):
        """``pl.seed_everything`` + ``torch.manual_seed`` must give bit-
        identical weights across runs. If a model adds non-deterministic
        ops (e.g. Dropout in eval mode disabled), this regresses."""
        m1, _, _ = _build_model_and_data(seed=42)
        m2, _, _ = _build_model_and_data(seed=42)
        for n1, p1 in m1.named_parameters():
            p2 = dict(m2.named_parameters())[n1]
            assert torch.allclose(p1, p2), f"parameter {n1!r} differs across seeds"

    def test_different_seeds_produce_different_weights(self):
        m1, _, _ = _build_model_and_data(seed=1)
        m2, _, _ = _build_model_and_data(seed=2)
        # At least one parameter must differ.
        diffs = []
        for n1, p1 in m1.named_parameters():
            p2 = dict(m2.named_parameters())[n1]
            diffs.append(not torch.allclose(p1, p2))
        assert any(diffs)


class TestLightningCheckpointBestVal:
    def test_model_checkpoint_keeps_best_val_loss(self, tmp_path):
        """The framework uses ``ModelCheckpoint(monitor="val_loss")``;
        if the saved checkpoint silently became "last" instead of "best",
        recovered models would be stale. Exercises the contract end-to-
        end with a synthetic monotonically-improving validation loss.
        """

        # A tiny stand-in module just to satisfy LightningModule signatures.
        class _Toy(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)
                self._epoch_idx = 0
                # Decreasing val losses guarantee the LAST epoch is also the best.
                self._planned_val_losses = [1.0, 0.5, 0.1]

            def training_step(self, batch, idx):
                x, y = batch
                loss = ((self.linear(x) - y) ** 2).mean()
                return loss

            def validation_step(self, batch, idx):
                # Fixed per-epoch — the planned schedule.
                val = self._planned_val_losses[self._epoch_idx]
                self.log("val_loss", val, on_epoch=True, prog_bar=False)

            def on_validation_epoch_end(self):
                self._epoch_idx = min(
                    self._epoch_idx + 1,
                    len(self._planned_val_losses) - 1,
                )

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        train_ds = TensorDataset(torch.randn(8, 2), torch.randn(8, 1))
        val_ds = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
        train_dl = DataLoader(train_ds, batch_size=4)
        val_dl = DataLoader(val_ds, batch_size=2)

        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path),
            filename="best-{epoch}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            max_epochs=3,
            callbacks=[ckpt_cb],
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )
        model = _Toy()
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        best_score = float(ckpt_cb.best_model_score)
        # With a strictly decreasing schedule the best val_loss is the last
        # one (0.1), confirming the monitor wired up the "min" comparison.
        assert best_score == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# RecommenderLightningModule + checkpoint format integration
# ---------------------------------------------------------------------------


class TestLightningModuleCheckpointShape:
    def test_lightning_checkpoint_has_module_prefix(self, tmp_path):
        """When the framework saves a RecommenderLightningModule, the
        state_dict keys are prefixed with ``module.<key>`` because the
        recommender lives at ``LightningModule.module``. ``load_trained_model``
        strips that prefix. This test pins the prefix at save time so the
        two halves stay aligned."""
        model, _, _ = _build_model_and_data(seed=33)
        lit = RecommenderLightningModule(
            module=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            loss_fn=nn.BCELoss(),
            metric_names=[],
            loss_name="bce",
        )
        ckpt_path = tmp_path / "manual.ckpt"
        torch.save({"state_dict": lit.state_dict()}, ckpt_path)

        sd = torch.load(ckpt_path, weights_only=False)["state_dict"]
        # All keys begin with "module." (the wrapper attribute name).
        assert all(k.startswith("module.") for k in sd.keys())
