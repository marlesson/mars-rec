"""
Synthetic implicit-feedback dataset: the offline "hello world" of mars-rec.

Nothing is downloaded. Users and items get random latent vectors, and each
user's interactions are drawn (without repetition) from a softmax over
``user · item + popularity``, so there is a real and learnable signal, and
popularity alone is a meaningful baseline to beat. The default size, 500
users x 200 items x 30 events, trains in a few seconds on CPU, which is why this
project backs the end-to-end test in ``tests/test_examples_e2e.py``.

Pedagogical highlights:
- The smallest possible ``BasePrepareDataFrames``: generation lives in an
  upstream ``luigi.Task`` and the prepare task only points at its CSV.
- Generation knobs are Luigi parameters, so changing one gives a new task
  id instead of silently reusing a stale CSV.
"""

import os
from typing import Optional

import luigi
import numpy as np
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "synthetic")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")


class GenerateSyntheticInteractions(luigi.Task):
    """Write ``(user_id, item_id, timestamp, label)`` positives to a CSV."""

    n_users: int = luigi.IntParameter(default=500)
    n_items: int = luigi.IntParameter(default=200)
    events_per_user: int = luigi.IntParameter(default=30)
    latent_dim: int = luigi.IntParameter(default=8)
    seed: int = luigi.IntParameter(default=42)

    def output(self):
        name = f"synthetic_{self.n_users}u_{self.n_items}i_{self.events_per_user}e_{self.latent_dim}d_{self.seed}.csv"
        return luigi.LocalTarget(os.path.join(DATASET_DIR, name))

    def run(self):
        rng = np.random.default_rng(self.seed)
        users = rng.normal(size=(self.n_users, self.latent_dim))
        items = rng.normal(size=(self.n_items, self.latent_dim))
        popularity = rng.normal(scale=1.0, size=self.n_items)

        affinity = users @ items.T / np.sqrt(self.latent_dim) * 2.0 + popularity  # [U, I]
        probs = np.exp(affinity - affinity.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        rows = []
        for u in range(self.n_users):
            # Distinct items per user: the evaluation removes items already
            # seen in train/val, so repeats would never count as held-out hits.
            for i in rng.choice(self.n_items, size=self.events_per_user, replace=False, p=probs[u]):
                rows.append((f"u{u}", f"i{i}"))
        df = pd.DataFrame(rows, columns=["user_id", "item_id"])
        # Shuffle, then give every event a unique increasing timestamp so the
        # temporal split mixes all users into train / val / test.
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        df["timestamp"] = np.arange(len(df), dtype=np.int64)
        df["label"] = 1.0

        os.makedirs(DATASET_DIR, exist_ok=True)
        df.to_csv(self.output().path, index=False)
        print(f"Synthetic dataset: {len(df)} events, {self.n_users} users, {self.n_items} items")


class PrepareSyntheticDataFrame(BasePrepareDataFrames):
    def requires(self):
        return GenerateSyntheticInteractions()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def timestamp_property(self) -> Optional[str]:
        return "timestamp"
