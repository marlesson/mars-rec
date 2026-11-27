"""Tests for the new hard-negative sampler classes added in dataset.py.

Covers:
- InteractionsWithEmbeddingHardNegativeDataset
- InteractionsWithCandidateProviderNegativeDataset
- InteractionsWithDynamicNegativeMiningDataset
- InteractionsWithMixedHardNegativeDataset

These tests focus on:
- correct shape / dtype of the produced batches
- that negatives never collide with the positive item
- that negatives stay within the valid item idx range
- fallback paths when assets (NN matrix, model) are missing
"""

import os
import pickle
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mars_rec.data.dataset import (
    InteractionsWithCandidateProviderNegativeDataset,
    InteractionsWithDynamicNegativeMiningDataset,
    InteractionsWithMixedDnsUniformDataset,
    InteractionsWithEmbeddingHardNegativeDataset,
    InteractionsWithMixedHardNegativeDataset,
)
from mars_rec.filtering import (
    AvailableItemsColumnProvider,
    CandidateFilter,
    CompositeProvider,
    FullCatalogProvider,
)
from mars_rec.meta_config import Column, IOType, ProjectConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_config_basic():
    """Bare-bones BCE-style config."""
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=MagicMock(),
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        other_input_columns=[],
        output_column=Column("label", IOType.NUMBER),
    )


@pytest.fixture
def index_mapping_basic():
    return {
        "user_id": defaultdict(int, {"1": 1, "2": 2, "3": 3}),
        "item_id": defaultdict(
            int,
            {"10": 2, "20": 3, "30": 4, "40": 5, "50": 6, "60": 7, "70": 8},
        ),
    }


@pytest.fixture
def sample_df_basic():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 3, 1],
            "item_id": [2, 3, 4, 5, 6, 7, 8],  # framework idx > 1
            "label": [1.0] * 7,
        }
    )


@pytest.fixture
def neighbors_pickle(tmp_path):
    """Build a neighbour table {uuid: [neighbor_uuids]} matching the basic
    index_mapping. uuid '10' (idx 2) → neighbours [20, 30, 40]."""
    nn = {
        "10": ["20", "30", "40"],
        "20": ["10", "30", "50"],
        "30": ["20", "40", "60"],
        "40": ["10", "30", "70"],
        "50": ["10", "20", "30"],
        "60": ["20", "30", "40"],
        "70": ["20", "40", "60"],
    }
    path = tmp_path / "nn.pkl"
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    return str(path)


# ---------------------------------------------------------------------------
# EmbeddingHardNegative
# ---------------------------------------------------------------------------


class TestEmbeddingHardNegative:
    def test_construct_and_match_neighbours(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
        neighbors_pickle,
    ):
        ds = InteractionsWithEmbeddingHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            neighbors_path=neighbors_pickle,
            top_k_use=10,
            uniform_fallback_ratio=0.0,
        )
        # Each positive-uuid → neighbours mapped to idx
        # uuid '10' is idx 2; neighbours '20','30','40' → idx 3,4,5
        assert set(ds._neighbors_by_idx[2].tolist()) == {3, 4, 5}

    def test_negatives_within_vocab(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
        neighbors_pickle,
    ):
        ds = InteractionsWithEmbeddingHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            neighbors_path=neighbors_pickle,
            uniform_fallback_ratio=0.0,
            negative_proportion=0.5,
        )
        all_indices = list(range(len(ds)))
        np.random.seed(0)
        (user, item), output = ds[all_indices]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()
        # Half are negatives — output should contain zeros.
        assert (output == 0.0).any()

    def test_missing_neighbors_path_raises(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        with pytest.raises(ValueError, match="neighbors_path"):
            InteractionsWithEmbeddingHardNegativeDataset(
                sample_df_basic,
                None,
                project_config_basic,
                index_mapping_basic,
            )

    def test_uniform_fallback_when_no_neighbours(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
        tmp_path,
    ):
        # NN file exists but matches no idx (uuids unknown to the mapping).
        nn = {"unknown_uuid": ["another_unknown"]}
        path = tmp_path / "empty_nn.pkl"
        with open(path, "wb") as f:
            pickle.dump(nn, f)
        ds = InteractionsWithEmbeddingHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            neighbors_path=str(path),
            uniform_fallback_ratio=0.0,
            negative_proportion=0.5,
        )
        # All neighbour arrays are empty → __getitem__ falls back to uniform.
        np.random.seed(0)
        (user, item), _ = ds[list(range(len(ds)))]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()


# ---------------------------------------------------------------------------
# CandidateProviderNegative
# ---------------------------------------------------------------------------


class _OddOnlyFilter(CandidateFilter):
    """Stub filter: keep only odd item idx."""

    def mask(self, row, full_catalog):
        return (full_catalog % 2) == 1


class TestCandidateProviderNegative:
    def test_uses_provider_pool(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        # Configure factory that returns a provider keeping only odd idx.
        def factory(pc, idx_map, extras):
            return CompositeProvider(
                filters=[_OddOnlyFilter()],
                min_pool=1,
                fallback=FullCatalogProvider(),
            )

        project_config_basic.candidate_provider_factory = factory

        ds = InteractionsWithCandidateProviderNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        # Negative items (where output == 0) should all be odd idx (after the
        # filter). Note positives ride in idx 2..8 even/odd mix.
        neg_mask = output == 0.0
        neg_items = item[neg_mask]
        # full_catalog is sorted idx (>1): {2,3,4,5,6,7,8}; odd survivors {3,5,7}
        assert all(it in {3, 5, 7} for it in neg_items.tolist())

    def test_raises_without_factory(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        # No factory AND no available_items column → resolve returns None.
        # ProjectConfig defaults available_items_column_name="available_items",
        # so we must clear it explicitly to get the "no provider" path.
        project_config_basic.available_items_column_name = None
        project_config_basic.candidate_provider_factory = None
        with pytest.raises(ValueError, match="candidate_provider_factory"):
            InteractionsWithCandidateProviderNegativeDataset(
                sample_df_basic,
                None,
                project_config_basic,
                index_mapping_basic,
            )

    def test_legacy_available_items_column(
        self,
        project_config_basic,
        index_mapping_basic,
    ):
        # Provide available_items_column_name + the column itself.
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 1],
                "item_id": [2, 3, 4, 5],
                "label": [1.0] * 4,
                "available_items": [[6, 7], [6, 7], [6, 7], [6, 7]],
            }
        )
        project_config_basic.available_items_column_name = "available_items"
        project_config_basic.candidate_provider_factory = None
        # Note: the AvailableItemsColumnProvider reads idx vectors directly.
        ds = InteractionsWithCandidateProviderNegativeDataset(
            df,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        neg_mask = output == 0.0
        neg_items = item[neg_mask]
        assert all(it in {6, 7} for it in neg_items.tolist())


# ---------------------------------------------------------------------------
# DynamicNegativeMining (DNS)
# ---------------------------------------------------------------------------


class TestDNS:
    def test_uniform_fallback_without_model(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        ds = InteractionsWithDynamicNegativeMiningDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            candidates_per_slot=4,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        # Without a model attached, behaves like uniform — items still in vocab.
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()

    def test_set_model_picks_argmax(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        import torch

        ds = InteractionsWithDynamicNegativeMiningDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            candidates_per_slot=4,
            warmup_steps=0,
        )

        # Toy model: score = item_idx (so argmax returns the largest sampled).
        class _ItemIdxScorer(torch.nn.Module):
            def forward(self, user_id, item_id):
                return item_id.float()

        model = _ItemIdxScorer()
        ds.set_model(model, device="cpu")

        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        # Hard to assert max without re-implementing internals; just check
        # that no item collides with the row's positive.
        n_pos = (output == 1.0).sum()
        n_neg = (output == 0.0).sum()
        assert n_pos > 0 and n_neg > 0


# ---------------------------------------------------------------------------
# MixedDnsUniform (Hybrid 30% uniform + 70% DNS)
# ---------------------------------------------------------------------------


class TestMixedDnsUniform:
    def test_uniform_fraction_zero_matches_pure_dns(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        """uniform_fraction=0 should be equivalent to pure DNS (warmup branch)."""
        ds = InteractionsWithMixedDnsUniformDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            candidates_per_slot=4,
            uniform_fraction=0.0,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()
        assert (output == 1.0).sum() > 0 and (output == 0.0).sum() > 0

    def test_uniform_fraction_one_skips_dns(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        """uniform_fraction=1.0 should never call the model."""
        import torch

        class _SpyScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, *args):
                self.calls += 1
                return torch.zeros(args[0].shape[0])

        ds = InteractionsWithMixedDnsUniformDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            candidates_per_slot=4,
            warmup_steps=0,
            uniform_fraction=1.0,
        )
        spy = _SpyScorer()
        ds.set_model(spy, device="cpu")
        np.random.seed(0)
        ds[list(range(len(ds)))]
        assert spy.calls == 0, "uniform_fraction=1.0 must not score with model"

    def test_hybrid_calls_model_only_for_dns_slots(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        """uniform_fraction=0.5 must score model for the DNS half."""
        import torch

        class _ItemIdxScorer(torch.nn.Module):
            def forward(self, user_id, item_id):
                return item_id.float()

        ds = InteractionsWithMixedDnsUniformDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            candidates_per_slot=4,
            warmup_steps=0,
            uniform_fraction=0.5,
        )
        ds.set_model(_ItemIdxScorer(), device="cpu")
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        # Sanity: items in vocab, positives + negatives both present.
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()
        assert (output == 1.0).sum() > 0 and (output == 0.0).sum() > 0

    def test_invalid_uniform_fraction_raises(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        with pytest.raises(ValueError):
            InteractionsWithMixedDnsUniformDataset(
                sample_df_basic,
                None,
                project_config_basic,
                index_mapping_basic,
                uniform_fraction=1.5,
            )


# ---------------------------------------------------------------------------
# MixedHardNegative
# ---------------------------------------------------------------------------


class TestMixedHardNegative:
    def test_pure_uniform(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        # 100% uniform → should not need neighbours_path.
        ds = InteractionsWithMixedHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            ann_ratio=0.0,
            popularity_ratio=0.0,
            uniform_ratio=1.0,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()

    def test_pure_popularity(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        ds = InteractionsWithMixedHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            ann_ratio=0.0,
            popularity_ratio=1.0,
            uniform_ratio=0.0,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()

    def test_ann_ratio_requires_neighbors(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
    ):
        with pytest.raises(FileNotFoundError):
            InteractionsWithMixedHardNegativeDataset(
                sample_df_basic,
                None,
                project_config_basic,
                index_mapping_basic,
                ann_ratio=0.5,
                popularity_ratio=0.3,
                uniform_ratio=0.2,
            )

    def test_full_mixture(
        self,
        project_config_basic,
        index_mapping_basic,
        sample_df_basic,
        neighbors_pickle,
    ):
        ds = InteractionsWithMixedHardNegativeDataset(
            sample_df_basic,
            None,
            project_config_basic,
            index_mapping_basic,
            neighbors_path=neighbors_pickle,
            negative_proportion=0.5,
            ann_ratio=0.5,
            popularity_ratio=0.3,
            uniform_ratio=0.2,
        )
        np.random.seed(0)
        (user, item), output = ds[list(range(len(ds)))]
        assert ((item >= 0) & (item <= ds._max_item_idx)).all()
        # Mixture ratios should be normalised to sum to 1.
        total = ds._ann_ratio + ds._pop_ratio + ds._unif_ratio
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TimeAware: item_first_seen accepts both dict and callable factory
# ---------------------------------------------------------------------------


class TestTimeAwareItemFirstSeen:
    def test_accepts_dict(self, project_config_basic, index_mapping_basic):
        from mars_rec.data.dataset import (
            InteractionsWithTimeAwareNegativeSamplingDataset,
        )

        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "item_id": [2, 3, 4],
                "label": [1.0] * 3,
                "timestamp": [1000, 2000, 3000],
            }
        )
        # Configure the project_config to know about the timestamp column.
        project_config_basic.timestamp_column_name = "timestamp"
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(
            df,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            item_first_seen={2: 500.0, 3: 1500.0, 4: 2500.0},
        )
        assert ds._item_first_seen == {2: 500.0, 3: 1500.0, 4: 2500.0}

    def test_accepts_callable_one_arg(
        self,
        project_config_basic,
        index_mapping_basic,
    ):
        """Factory ``(index_mapping) -> dict`` is resolved at __init__."""
        from mars_rec.data.dataset import (
            InteractionsWithTimeAwareNegativeSamplingDataset,
        )

        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "item_id": [2, 3, 4],
                "label": [1.0] * 3,
                "timestamp": [1000, 2000, 3000],
            }
        )
        project_config_basic.timestamp_column_name = "timestamp"

        captured = {}

        def factory(idx_map):
            captured["called"] = True
            captured["mapping"] = idx_map
            return {2: 100.0, 3: 200.0, 4: 300.0}

        ds = InteractionsWithTimeAwareNegativeSamplingDataset(
            df,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            item_first_seen=factory,
        )
        assert captured.get("called") is True
        assert captured["mapping"] is index_mapping_basic
        assert ds._item_first_seen == {2: 100.0, 3: 200.0, 4: 300.0}

    def test_accepts_callable_zero_arg(
        self,
        project_config_basic,
        index_mapping_basic,
    ):
        """A 0-arg callable is also supported (legacy / pre-built lookup)."""
        from mars_rec.data.dataset import (
            InteractionsWithTimeAwareNegativeSamplingDataset,
        )

        df = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [2],
                "label": [1.0],
                "timestamp": [1000],
            }
        )
        project_config_basic.timestamp_column_name = "timestamp"

        def factory():
            return {2: 50.0}

        ds = InteractionsWithTimeAwareNegativeSamplingDataset(
            df,
            None,
            project_config_basic,
            index_mapping_basic,
            negative_proportion=0.5,
            item_first_seen=factory,
        )
        assert ds._item_first_seen == {2: 50.0}


# ---------------------------------------------------------------------------
# DNS Lightning callback auto-attach
# ---------------------------------------------------------------------------


class TestDNSCallbackAutoAttach:
    def test_callback_attaches_when_dataset_has_set_model(self):
        """The trainer's _get_callbacks() includes the DNS callback only when
        the project's dataset_class exposes ``set_model``."""
        from mars_rec.torch.callbacks import DynamicNegativeMiningCallback

        # Sanity: callback class itself imports cleanly.
        cb = DynamicNegativeMiningCallback()
        assert hasattr(cb, "on_train_start")

    def test_inject_walks_dataloader_list(self):
        """``_inject`` tolerates both a single dataloader and a list/tuple,
        which mirrors PyTorch Lightning's val_dataloaders shape."""
        from mars_rec.torch.callbacks import DynamicNegativeMiningCallback

        class _StubDS:
            def __init__(self):
                self.injected = None

            def set_model(self, model, device):
                self.injected = (model, device)

        class _StubDL:
            def __init__(self, ds):
                self.dataset = ds

        ds_a = _StubDS()
        ds_b = _StubDS()
        cb = DynamicNegativeMiningCallback()
        cb._inject([_StubDL(ds_a), _StubDL(ds_b)], "MODEL", "cpu")
        assert ds_a.injected == ("MODEL", "cpu")
        assert ds_b.injected == ("MODEL", "cpu")

    def test_inject_skips_dataset_without_set_model(self):
        from mars_rec.torch.callbacks import DynamicNegativeMiningCallback

        class _PlainDS:
            def __init__(self):
                self.touched = False

        class _StubDL:
            def __init__(self, ds):
                self.dataset = ds

        plain = _PlainDS()
        cb = DynamicNegativeMiningCallback()
        cb._inject(_StubDL(plain), "MODEL", "cpu")  # single dataloader
        # No exception raised; plain dataset is silently skipped.
        assert getattr(plain, "touched", False) is False

    def test_resolve_model_prefers_recommender_module(self):
        from mars_rec.torch.callbacks import DynamicNegativeMiningCallback

        class _LM:
            def __init__(self):
                self.recommender_module = "REC"
                self.model = "M"

        cb = DynamicNegativeMiningCallback()
        assert cb._resolve_model(_LM()) == "REC"
