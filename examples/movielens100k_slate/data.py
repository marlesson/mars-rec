"""
MovieLens 100K with synthetic per-row candidate slates.

Demonstrates the framework's ``available_items_column_name`` feature: each
interaction row carries a list of candidate item ids, and negatives during
training are drawn from that slate (not from the full catalogue). This is
the standard "leave-one-out + 99 random negatives" evaluation format used
in the NCF paper and follow-ups — realistic, even though we synthesise the
slate here for demo purposes.

Why MovieLens 100K and not MIND-small (as originally planned): the public
MIND-small Azure blob is no longer accessible. MovieLens 100K is small
(~100K ratings, 943 users, 1682 movies), CPU-friendly, and reliably hosted
at grouplens.org. The slate-ranking lesson is identical.

This example is deliberately distinct from ``examples/movielens`` (which
uses MovieLens 1M with a different ``dataset_class``).
"""

import os
import zipfile
from typing import Optional

import luigi
import numpy as np
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "movielens100k_slate")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

SLATE_SIZE: int = 50  # 1 positive + (SLATE_SIZE - 1) random items
GENRE_DIM: int = 19  # ML-100K has 19 binary genre flags

METADATA_PATH = os.path.join(DATASET_DIR, "ml100k_item_metadata.csv")

# ML-100K item file column layout (from u.info / README)
_ML100K_ITEM_COLS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]
_GENRE_COLS = _ML100K_ITEM_COLS[5:]  # 19 binary flags


class DownloadAndPrepareML100KSlate(luigi.Task):
    def output(self):
        return (
            luigi.LocalTarget(os.path.join(DATASET_DIR, "ml100k_interactions.csv")),
            luigi.LocalTarget(METADATA_PATH),
        )

    def run(self):
        import math

        import requests
        from tqdm import tqdm

        os.makedirs(DATASET_DIR, exist_ok=True)

        zip_path = os.path.join(DATASET_DIR, "ml-100k.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens 100K from {ML100K_URL}...")
            r = requests.get(ML100K_URL, stream=True)
            total = int(r.headers.get("content-length", 0))
            block = 1024
            with open(zip_path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(block),
                    total=math.ceil(total / block) if total else 0,
                    unit="KB",
                    unit_scale=True,
                ):
                    f.write(chunk)

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
        items = pd.read_csv(
            os.path.join(extract_dir, "u.item"),
            sep="|",
            names=_ML100K_ITEM_COLS,
            encoding="latin-1",
        )

        # Item metadata: 19-dim binary genre vector per movie, stored as a
        # FLOAT_ARRAY in a separate per-item CSV (looked up at sample time).
        genre_matrix = items[_GENRE_COLS].astype(np.float32).values
        items["item_genre_vec"] = ["[" + ", ".join(f"{x:.1f}" for x in row) + "]" for row in genre_matrix]
        items[["item_id", "item_genre_vec"]].to_csv(METADATA_PATH, index=False)

        # Build per-row slate. For each rating, slate = [positive item] +
        # (SLATE_SIZE - 1) random other items. Drawing slate items globally
        # at random is the convention used in the NCF benchmark; one could
        # enrich it with same-genre / popularity constraints — left to the
        # reader.
        rng = np.random.default_rng(42)
        all_items = ratings["item_id"].unique()
        n_extra = SLATE_SIZE - 1

        slates = []
        for pos in ratings["item_id"].values:
            cands = rng.choice(all_items, size=n_extra + 1, replace=False)
            # Guarantee the positive is present in the slate
            if pos not in cands:
                cands[0] = pos
            cands = np.unique(cands)
            # If unique() shrunk the slate, top up
            while len(cands) < SLATE_SIZE:
                extra = rng.choice(all_items)
                if extra not in cands:
                    cands = np.append(cands, extra)
            slates.append(cands[:SLATE_SIZE].tolist())

        ratings["available_items"] = slates
        # Implicit label: rating >= 4 → 1
        ratings["label"] = (ratings["rating"] >= 4).astype(float)

        # Stringify the slate so the CSV round-trip works through
        # ``parallel_literal_eval``. ids are ints; quote them as ints so the
        # eval-back yields a clean list of ints.
        ratings["available_items"] = ratings["available_items"].apply(
            lambda lst: "[" + ", ".join(str(int(v)) for v in lst) + "]"
        )

        out = (
            ratings[["user_id", "item_id", "rating", "timestamp", "available_items", "label"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        out.to_csv(self.output()[0].path, index=False)
        print(f"Wrote {len(out)} interactions with slate size {SLATE_SIZE} and {len(items)} item-metadata rows.")


class PrepareML100KSlateDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareML100KSlate()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> str:
        return self.input()[0].path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return METADATA_PATH

    @property
    def timestamp_property(self) -> Optional[str]:
        return "timestamp"

    def create_available_items(self, df: pd.DataFrame) -> pd.DataFrame:
        # ``available_items`` is already populated by the upstream task; the
        # parent's default would otherwise overwrite it with a random slate.
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
