"""
Last.fm HetRec 2011 (2K users) dataset for sequential next-item recommendation.

Downloads the HetRec 2011 Last.fm 2K dataset and turns the tagging events into
an ordered click-stream per user. For each event we attach the user's last
``HISTORY_LEN`` artist ids as a sequence column.

Dataset: https://grouplens.org/datasets/hetrec-2011/
- ~186K (user, artist, tag, timestamp) tagging events
- 1892 users, ~12K artists (after filtering)
- Used here as a proxy listen-stream: each row = one engagement event with
  a real Unix timestamp.

Pedagogical highlights:
- Demonstrates ``IOType.INDEXABLE_ARRAY`` with ``same_index_as`` so the
  history sequence shares the item vocabulary.
- Demonstrates ``dataset_split_method="time"`` for a temporal next-item task.
"""

import os
import zipfile
from typing import Optional

import luigi
import numpy as np
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "lastfm")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

HISTORY_LEN: int = 10  # number of past artist ids kept per event
MIN_USER_EVENTS: int = 5  # drop users with fewer than this many events
MIN_ARTIST_EVENTS: int = 3  # drop very long-tail artists
PAD_TOKEN: str = "-1"  # framework maps "-1" → index 2 (PAD slot)


class DownloadAndPrepareLastFM(luigi.Task):
    """Download the HetRec Last.fm 2K zip and prepare a single CSV."""

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "lastfm_sequential.csv"))

    def run(self):
        import math

        import requests
        from tqdm import tqdm

        os.makedirs(DATASET_DIR, exist_ok=True)

        zip_path = os.path.join(DATASET_DIR, "hetrec2011-lastfm-2k.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading HetRec Last.fm 2K from {LASTFM_URL}...")
            r = requests.get(LASTFM_URL, stream=True)
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

        extract_dir = os.path.join(DATASET_DIR, "lastfm")
        if not os.path.exists(extract_dir):
            print("Extracting...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

        events_path = os.path.join(extract_dir, "user_taggedartists-timestamps.dat")
        events = pd.read_csv(events_path, sep="\t", encoding="latin-1")
        events.columns = [c.lower() for c in events.columns]
        events = events.rename(columns={"userid": "user_id", "artistid": "artist_id"})
        # timestamps are milliseconds; convert to seconds for readability
        events["timestamp"] = (events["timestamp"] // 1000).astype(np.int64)
        # drop rows with invalid timestamps (a handful in this dataset have 0)
        events = events[events["timestamp"] > 0]

        # Collapse multiple tags for the same (user, artist, timestamp) to one
        # engagement event — the tag-id itself is not used by the model.
        events = events.drop_duplicates(subset=["user_id", "artist_id", "timestamp"])

        # Filter long-tail artists and inactive users for a CPU-friendly demo.
        artist_counts = events["artist_id"].value_counts()
        keep_artists = artist_counts[artist_counts >= MIN_ARTIST_EVENTS].index
        events = events[events["artist_id"].isin(keep_artists)]
        user_counts = events["user_id"].value_counts()
        keep_users = user_counts[user_counts >= MIN_USER_EVENTS].index
        events = events[events["user_id"].isin(keep_users)]

        events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Build a fixed-length history of the user's PRIOR artist ids for each
        # event. We pad with PAD_TOKEN at the LEFT so the most recent item is
        # always at position HISTORY_LEN-1 (lets the model use a "last-token"
        # readout without a length tensor).
        rows = []
        for _, group in events.groupby("user_id", sort=False):
            prior: list = []
            for user_id, artist_id, ts in zip(
                group["user_id"].values,
                group["artist_id"].values,
                group["timestamp"].values,
            ):
                if prior:
                    padded = [PAD_TOKEN] * (HISTORY_LEN - len(prior)) + [str(a) for a in prior[-HISTORY_LEN:]]
                    rows.append(
                        (
                            str(user_id),
                            str(artist_id),
                            int(ts),
                            padded,
                        )
                    )
                prior.append(int(artist_id))

        out = pd.DataFrame(
            rows,
            columns=["user_id", "artist_id", "timestamp", "history_artist_ids"],
        )
        out["label"] = 1.0
        # store the array column as a python-literal string so the framework's
        # ``parallel_literal_eval`` can parse it back on read.
        out["history_artist_ids"] = out["history_artist_ids"].apply(
            lambda lst: "[" + ", ".join(f"'{v}'" for v in lst) + "]"
        )

        out.to_csv(self.output().path, index=False)
        print(f"Last.fm sequential dataset prepared: {len(out)} events")
        print(f"  Users: {out['user_id'].nunique()}")
        print(f"  Artists: {out['artist_id'].nunique()}")
        print(
            "  Date range: "
            f"{pd.to_datetime(out['timestamp'].min(), unit='s').date()} → "
            f"{pd.to_datetime(out['timestamp'].max(), unit='s').date()}"
        )


class PrepareLastFMDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareLastFM()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return None

    @property
    def timestamp_property(self) -> Optional[str]:
        return "timestamp"

    def create_available_items(self, df: pd.DataFrame) -> pd.DataFrame:
        # negatives are sampled randomly from all items by the dataset_class —
        # no per-row candidate slate needed.
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
