"""Tests for ``BasePrepareDataFrames.balance_dataset`` (random over/under-sampling).

The implementation replaced imbalanced-learn's ``RandomOverSampler`` /
``RandomUnderSampler``. The expected class counts below are the ones
imbalanced-learn 0.14 produced for the same inputs, so these tests pin the
sampling *semantics*. The exact rows drawn for a given seed differ from
imbalanced-learn's (different random generator), which is expected.
"""

from __future__ import annotations

import luigi
import numpy as np
import pandas as pd
import pytest

from mars_rec.data.task import BasePrepareDataFrames


class _BalancePrep(BasePrepareDataFrames):
    @property
    def dataset_dir(self):  # pragma: no cover - not used by balance_dataset
        return "/tmp/unused"

    @property
    def read_data_frame_path(self):  # pragma: no cover
        return None


@pytest.fixture(autouse=True)
def _fresh_luigi_instances():
    luigi.task_register.Register.clear_instance_cache()


@pytest.fixture
def df():
    # 10 / 4 / 2 rows of label 0 / 1 / 2; a non-default index to catch
    # label-vs-position mix-ups.
    return pd.DataFrame(
        {"label": [0] * 10 + [1] * 4 + [2] * 2, "group": ["a", "b"] * 8},
        index=range(100, 116),
    )


PROPORTIONS = {"label": {0: 1.0, 1: 0.5, 2: 0.25}}


def _counts(frame, column="label"):
    return frame[column].value_counts().sort_index().to_dict()


def _balance(df, **params):
    return _BalancePrep(**params).balance_dataset(df)


def test_none_returns_input_unchanged(df):
    assert _balance(df, sampling_strategy="none", balance_fields=["label"]) is df


def test_oversample_auto_grows_every_class_to_the_majority(df):
    out = _balance(df, sampling_strategy="oversample", balance_fields=["label"])
    assert _counts(out) == {0: 10, 1: 10, 2: 10}
    # Every original row is kept, in its original order, before the new draws.
    assert list(out.index[: len(df)]) == list(df.index)
    pd.testing.assert_frame_equal(out.loc[~out.index.duplicated()].sort_index(), df)


def test_undersample_auto_shrinks_every_class_to_the_minority(df):
    out = _balance(df, sampling_strategy="undersample", balance_fields=["label"])
    assert _counts(out) == {0: 2, 1: 2, 2: 2}
    assert not out.index.duplicated().any()  # drawn without replacement
    assert set(out.index) <= set(df.index)
    pd.testing.assert_frame_equal(out, df.loc[out.index])  # rows are not altered


def test_oversample_with_proportions(df):
    out = _balance(df, sampling_strategy="oversample", balance_fields=["label"], sampling_proportions=PROPORTIONS)
    assert _counts(out) == {0: 40, 1: 20, 2: 10}


def test_undersample_with_proportions_can_drop_a_class(df):
    out = _balance(df, sampling_strategy="undersample", balance_fields=["label"], sampling_proportions=PROPORTIONS)
    # Target for class 2 is int(0.25 * 2 / 1.0) == 0.
    assert _counts(out) == {0: 2, 1: 1}


def test_fields_are_balanced_in_sequence(df):
    out = _balance(df, sampling_strategy="oversample", balance_fields=["label", "group"])
    # The last field is balanced on the output of the first pass.
    counts = _counts(out, "group")
    assert counts["a"] == counts["b"]
    assert len(out) > 30


def test_same_seed_is_deterministic_and_seed_changes_the_draw(df):
    a = _balance(df, sampling_strategy="undersample", balance_fields=["label"], seed=7)
    luigi.task_register.Register.clear_instance_cache()
    b = _balance(df, sampling_strategy="undersample", balance_fields=["label"], seed=7)
    assert list(a.index) == list(b.index)

    draws = set()
    for seed in range(10):
        luigi.task_register.Register.clear_instance_cache()
        draws.add(tuple(_balance(df, sampling_strategy="undersample", balance_fields=["label"], seed=seed).index))
    assert len(draws) > 1


def test_works_with_a_duplicated_index(df):
    dup = df.set_axis([0] * len(df))
    out = _balance(dup, sampling_strategy="oversample", balance_fields=["label"])
    assert _counts(out) == {0: 10, 1: 10, 2: 10}


def test_invalid_targets_raise(df):
    task = _BalancePrep(sampling_strategy="undersample", balance_fields=["label"])
    positions = np.arange(len(df))
    with pytest.raises(ValueError, match="Cannot undersample"):
        task._resample_positions(positions, df["label"].to_numpy(), {0: 11}, np.random.default_rng(0))
    luigi.task_register.Register.clear_instance_cache()
    task = _BalancePrep(sampling_strategy="oversample", balance_fields=["label"])
    with pytest.raises(ValueError, match="Cannot oversample"):
        task._resample_positions(positions, df["label"].to_numpy(), {0: 3}, np.random.default_rng(0))
