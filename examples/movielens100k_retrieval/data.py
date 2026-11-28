"""
MovieLens 100K as a candidate-retrieval task (two-tower + InfoNCE).

Every positive rating (>= 4) becomes one row: the user, the movie they rated
(the target) and the user's previous ``HISTORY_LEN`` positive movies. A
two-tower model learns a query embedding from (user, history) and an item
embedding, and is trained contrastively: the other targets in the batch are
the negatives, so no negative sampling is configured.

Dataset: https://grouplens.org/datasets/movielens/100k/
- 100K ratings, 943 users, 1682 movies (~55K positives with rating >= 4)
"""

import os
import zipfile
from typing import Optional

import luigi
import numpy as np
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "movielens100k_retrieval")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

HISTORY_LEN: int = 20  # previous positive movies kept per row
PAD_TOKEN: str = "-1"  # framework maps "-1" → index 2 (PAD slot)


class DownloadAndPrepareML100KRetrieval(luigi.Task):
    """Download MovieLens 100K and write one (user, history, target) row per positive."""

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "ml100k_retrieval.csv"))

    def run(self):
        from mars_rec.utils.download import download_file

        os.makedirs(DATASET_DIR, exist_ok=True)

        zip_path = os.path.join(DATASET_DIR, "ml-100k.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens 100K from {ML100K_URL}...")
            download_file(ML100K_URL, zip_path)

        extract_dir = os.path.join(DATASET_DIR, "ml-100k")
        if not os.path.exists(extract_dir):
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATASET_DIR)

        ratings = pd.read_csv(
            os.path.join(extract_dir, "u.data"),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        positives = ratings[ratings["rating"] >= 4].sort_values(["user_id", "timestamp", "item_id"])

        # Left-pad with PAD_TOKEN so the most recent movie is always last.
        rows = []
        for user_id, group in positives.groupby("user_id", sort=False):
            prior: list = []
            for item_id, ts in zip(group["item_id"].values, group["timestamp"].values):
                if prior:
                    recent = [str(i) for i in prior[-HISTORY_LEN:]]
                    rows.append(
                        (str(user_id), str(item_id), int(ts), [PAD_TOKEN] * (HISTORY_LEN - len(recent)) + recent)
                    )
                prior.append(int(item_id))

        out = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "history_item_ids"])
        # Python-literal list string, parsed back by the framework on read.
        out["history_item_ids"] = out["history_item_ids"].apply(
            lambda lst: "[" + ", ".join(f"'{v}'" for v in lst) + "]"
        )
        out = out.sort_values("timestamp").reset_index(drop=True)
        out["timestamp"] = out["timestamp"].astype(np.int64)

        out.to_csv(self.output().path, index=False)
        print(f"Wrote {len(out)} retrieval rows ({out['user_id'].nunique()} users, {out['item_id'].nunique()} movies)")


class PrepareML100KRetrievalDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareML100KRetrieval()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def timestamp_property(self) -> Optional[str]:
        return "timestamp"
