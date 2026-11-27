"""A minimal mars-rec project, importable via dotted path.

Used by Round-2 lifecycle tests that need a real ``ProjectConfig`` for
``SupervisedModelTraining`` to dispatch against. Reads the dataset
directory from ``MARS_TEST_DATASET_DIR`` env var so each test can point
the prep task at its own ``tmp_path``.

Keep this file tiny — it's a fixture, not an example project. The
synthetic dataset (60 rows, 5 users × 12 items, binary label) is large
enough to split 60 / 20 / 20 across train/val/test, and small enough
for a 2-epoch LogisticRegression run to finish in ~5 seconds on CPU.
"""

from __future__ import annotations

import os

import pandas as pd

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.meta_config import Column, IOType, ProjectConfig


def _dataset_dir() -> str:
    # Late-bound so each test (which sets the env var via monkeypatch)
    # picks up its own path. Falling back to a stable default keeps the
    # module importable outside test context too.
    return os.environ.get("MARS_TEST_DATASET_DIR", "/tmp/mars_test_dataset")


class DummyPrep(BasePrepareDataFrames):
    """A BasePrepareDataFrames subclass with a hardcoded source DataFrame."""

    @property
    def dataset_dir(self) -> str:
        return _dataset_dir()

    @property
    def read_data_frame_path(self):
        return None  # we override read_data_frame() directly

    def read_data_frame(self) -> pd.DataFrame:
        # 60 rows: 5 users × 12 items, binary label parity-derived.
        rows = []
        for u in range(5):
            for i in range(12):
                rows.append(
                    {
                        "user_id": f"u{u}",
                        "item_id": f"i{i}",
                        "label": float((u + i) % 2),
                    }
                )
        return pd.DataFrame(rows)


dummy_project_config = ProjectConfig(
    base_dir="/tmp/mars_test_dataset",
    prepare_data_frames_task=DummyPrep,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    output_column=Column("label", IOType.NUMBER),
)
