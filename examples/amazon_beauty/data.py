"""
Amazon Reviews — Beauty 5-core for content-based recommendation.

Demonstrates the framework's ``metadata_columns`` mechanism: per-item
content features (a 64-dim TF-IDF + TruncatedSVD embedding of the
review text) are stored once in a separate metadata CSV and looked up
by item index at sample time, instead of repeating the vector in every
interaction row.

Dataset: https://snap.stanford.edu/data/web-Amazon.html
- ~198K reviews of ~12K beauty products (5-core: each user/item has >=5)
- Fields used: reviewerID (user), asin (item), summary, overall (rating)
"""

import gzip
import json
import os
from typing import Optional

import luigi
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "amazon_beauty")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

REVIEWS_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"

TEXT_EMB_DIM: int = 64
MIN_USER_REVIEWS: int = 5
MIN_ITEM_REVIEWS: int = 5

METADATA_PATH = os.path.join(DATASET_DIR, "beauty_item_metadata.csv")


def _parse_jsonl_gz(path: str) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


class DownloadAndPrepareAmazonBeauty(luigi.Task):
    def output(self):
        return (
            luigi.LocalTarget(os.path.join(DATASET_DIR, "beauty_interactions.csv")),
            luigi.LocalTarget(METADATA_PATH),
        )

    def run(self):
        from mars_rec.utils.download import download_file

        os.makedirs(DATASET_DIR, exist_ok=True)

        gz_path = os.path.join(DATASET_DIR, "reviews_Beauty_5.json.gz")
        if not os.path.exists(gz_path):
            print(f"Downloading Beauty 5-core reviews from {REVIEWS_URL}...")
            download_file(REVIEWS_URL, gz_path)

        print("Parsing JSONL...")
        df = _parse_jsonl_gz(gz_path)

        keep = {"reviewerID", "asin", "overall", "unixReviewTime", "summary", "reviewText"}
        df = df[[c for c in df.columns if c in keep]].copy()
        df["summary"] = df["summary"].fillna("").astype(str)
        df["reviewText"] = df.get("reviewText", pd.Series([""] * len(df))).fillna("").astype(str)
        df["text"] = (df["summary"] + " " + df["reviewText"]).str.strip()
        df = df.dropna(subset=["reviewerID", "asin", "overall", "unixReviewTime"])
        df["overall"] = df["overall"].astype(float)
        df["unixReviewTime"] = df["unixReviewTime"].astype(np.int64)

        # 5-core filter (sample is already 5-core, but re-applying after any
        # NaN drops keeps the invariant)
        for _ in range(3):
            uc = df["reviewerID"].value_counts()
            ic = df["asin"].value_counts()
            df = df[
                df["reviewerID"].isin(uc[uc >= MIN_USER_REVIEWS].index)
                & df["asin"].isin(ic[ic >= MIN_ITEM_REVIEWS].index)
            ]

        print(f"  reviews={len(df)} users={df['reviewerID'].nunique()} items={df['asin'].nunique()}")

        # Build per-item text by concatenating review text for each asin —
        # this gives an item-level "document" for the TF-IDF model.
        item_text = (
            df.groupby("asin")["text"]
            .apply(lambda s: " ".join(s.values)[:5000])  # cap to keep TF-IDF fast
            .reset_index()
        )

        print(f"Fitting TF-IDF + SVD({TEXT_EMB_DIM}) over {len(item_text)} item documents...")
        vec = TfidfVectorizer(
            max_features=20_000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
        )
        tfidf = vec.fit_transform(item_text["text"].values)
        svd = TruncatedSVD(n_components=TEXT_EMB_DIM, random_state=42)
        emb = svd.fit_transform(tfidf).astype(np.float32)

        item_text["item_text_emb"] = ["[" + ", ".join(f"{x:.5f}" for x in row) + "]" for row in emb]
        metadata = item_text[["asin", "item_text_emb"]].rename(
            columns={"asin": "item_id"},
        )
        metadata.to_csv(METADATA_PATH, index=False)

        # Implicit label: positive if rating >= 4 (standard MovieLens-style
        # threshold; preserves enough negatives in the 1-5 scale).
        interactions = df.rename(
            columns={
                "reviewerID": "user_id",
                "asin": "item_id",
                "overall": "rating",
                "unixReviewTime": "timestamp",
            }
        )[["user_id", "item_id", "rating", "timestamp"]]
        interactions["label"] = (interactions["rating"] >= 4.0).astype(float)
        interactions = interactions.sort_values("timestamp").reset_index(drop=True)
        interactions.to_csv(self.output()[0].path, index=False)
        print(f"Wrote {len(interactions)} interactions and {len(metadata)} item-metadata rows.")


class PrepareAmazonBeautyDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareAmazonBeauty()

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
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
