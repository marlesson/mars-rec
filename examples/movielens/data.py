"""
MovieLens 1M dataset for recommendation/ranking.

Downloads the MovieLens 1M dataset and preprocesses it for implicit feedback
recommendation using the mars-rec framework.

Dataset: https://grouplens.org/datasets/movielens/1m/
- ~1M ratings from 6040 users on 3706 movies
- Ratings: 1-5 (converted to implicit: rating >= 4 -> positive)
- Features: user gender, age group, occupation; movie genres

User features are per-row ``NUMBER`` columns (they belong to the row's user,
which negative sampling never changes). Movie genres are a per-item
``FLOAT_ARRAY`` in a separate metadata CSV (``metadata_columns``): when the
dataset swaps the item id to build a negative, it re-looks-up the genres of
the *sampled* movie. Keeping genres as per-row columns instead would leak the
label — every negative would carry the genres of the positive movie.
"""

import os
import zipfile
from typing import Optional

import luigi
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "movielens")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

METADATA_PATH: str = os.path.join(DATASET_DIR, "movielens_1m_item_metadata.csv")

# Age encoding from MovieLens README
AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
GENDER_MAP = {"F": 0, "M": 1}

# The 18 MovieLens 1M genres, multi-hot encoded per movie
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRE_DIM: int = len(GENRES)


class DownloadAndPrepareMovieLens(luigi.Task):
    """Download MovieLens 1M and write the interactions + item-metadata CSVs."""

    def output(self):
        return (
            luigi.LocalTarget(os.path.join(DATASET_DIR, "movielens_1m.csv")),
            luigi.LocalTarget(METADATA_PATH),
        )

    def run(self):
        from mars_rec.utils.download import download_file

        os.makedirs(DATASET_DIR, exist_ok=True)

        # Download zip
        zip_path = os.path.join(DATASET_DIR, "ml-1m.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens 1M from {MOVIELENS_1M_URL}...")
            download_file(MOVIELENS_1M_URL, zip_path)

        # Extract
        extract_dir = os.path.join(DATASET_DIR, "ml-1m")
        if not os.path.exists(extract_dir):
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATASET_DIR)

        # --- Load ratings ---
        ratings = pd.read_csv(
            os.path.join(extract_dir, "ratings.dat"),
            sep="::",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )

        # --- Load users ---
        users = pd.read_csv(
            os.path.join(extract_dir, "users.dat"),
            sep="::",
            header=None,
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
            encoding="latin-1",
        )

        # --- Load movies ---
        movies = pd.read_csv(
            os.path.join(extract_dir, "movies.dat"),
            sep="::",
            header=None,
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

        # --- User features ---
        users["gender_encoded"] = users["gender"].map(GENDER_MAP).astype(float)
        users["age_encoded"] = users["age"].map(AGE_MAP).astype(float)
        # Normalize occupation (0-20 -> 0-1)
        users["occupation_norm"] = (users["occupation"] / 20.0).astype(float)

        # --- Movie features: multi-hot genre vector in a per-item metadata CSV ---
        genre_sets = movies["genres"].str.split("|").apply(set)
        movies["movie_genre_vec"] = [
            "[" + ", ".join("1.0" if g in genres else "0.0" for g in GENRES) + "]" for genres in genre_sets
        ]
        movies[["movie_id", "movie_genre_vec"]].to_csv(METADATA_PATH, index=False)

        # --- Merge user features (per-row) ---
        df = ratings.merge(
            users[["user_id", "gender_encoded", "age_encoded", "occupation_norm"]], on="user_id", how="left"
        )

        # --- Implicit feedback: rating >= 4 is positive ---
        df["label"] = (df["rating"] >= 4).astype(float)

        # Sort by timestamp for temporal split
        df = df.sort_values("timestamp").reset_index(drop=True)

        result = df[["user_id", "movie_id", "gender_encoded", "age_encoded", "occupation_norm", "timestamp", "label"]]

        result.to_csv(self.output()[0].path, index=False)
        print(f"MovieLens 1M dataset prepared: {len(result)} ratings, {len(movies)} item-metadata rows")
        print(f"  Users: {result['user_id'].nunique()}")
        print(f"  Movies: {result['movie_id'].nunique()}")
        print(f"  Positive rate (rating>=4): {result['label'].mean():.2%}")
        print(
            f"  Date range: {pd.to_datetime(result['timestamp'].min(), unit='s').date()} "
            f"to {pd.to_datetime(result['timestamp'].max(), unit='s').date()}"
        )


class PrepareMovieLensDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareMovieLens()

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

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        float_cols = [c for c in df.columns if c not in ("user_id", "movie_id", "timestamp")]
        for col in float_cols:
            df[col] = df[col].astype(float)
        return df
