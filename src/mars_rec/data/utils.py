"""Legacy bundled-dataset helpers (random, YooChoose, Trivago).

Carried over from the original mars-gym project; modern examples
under ``examples/`` define their own download tasks instead. Kept
here for backward compatibility with any project still pointing at
the public ``mars-gym-dataset`` Cloud Storage bucket.
"""
import abc
import luigi

import pandas as pd
import numpy as np
import os
from mars_rec.data.task import BasePrepareDataFrames
import requests
from tqdm import tqdm
import math
from mars_rec.utils import files

DATASETS = dict(
    random=["https://storage.googleapis.com/mars-gym-dataset/raw/random/dataset.csv"],
    yoochoose=[
        "https://storage.googleapis.com/mars-gym-dataset/raw/yoochoose/yoochoose-buys.dat"
    ],
    processed_yoochoose=[
        "https://storage.googleapis.com/mars-gym-dataset/process/yoochoose/dataset.csv"
    ],
    trivago_rio=[
        "https://storage.googleapis.com/mars-gym-dataset/raw/trivago/rio/train.csv",
        "https://storage.googleapis.com/mars-gym-dataset/raw/trivago/rio/item_metadata.csv",
    ],
    processed_trivago_rio=[
        "https://storage.googleapis.com/mars-gym-dataset/process/trivago/rio/interaction_dataset.csv",
        "https://storage.googleapis.com/mars-gym-dataset/process/trivago/rio/item_metadata_transform.csv",
    ],
)


def datasets():
    """Return the list of bundled dataset names known to :data:`DATASETS`."""
    return list(DATASETS.keys())


def load_dataset(name, cache=True, output_path=".", **kws):
    """Download (cached) and read the bundled dataset ``name``.

    Returns a list of DataFrames — one per file in the dataset entry.
    Files are cached under ``output_path/<name>/`` on first download
    and re-read from disk on subsequent calls unless ``cache=False``.
    Extra ``**kws`` are forwarded to :func:`pandas.read_csv`.
    """
    results = []
    for url in DATASETS[name]:
        output_file = os.path.join(output_path, name, os.path.basename(url))
        if not os.path.isfile(output_file) or not cache:
            r = requests.get(url, stream=True)

            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            wrote = 0
            os.makedirs(os.path.split(output_file)[0], exist_ok=True)
            with open(output_file, "wb") as f:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=math.ceil(total_size // block_size),
                    unit="KB",
                    unit_scale=True,
                ):
                    wrote = wrote + len(data)
                    f.write(data)
            if total_size != 0 and wrote != total_size:
                raise ConnectionError("ERROR, something went wrong")

        df = pd.read_csv(output_file, **kws)
        results.append(df)
    return results


class DownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    """Luigi task that downloads one of the bundled datasets to disk."""

    output_path: str = luigi.Parameter(default=files.OUTPUT_PATH)
    dataset: str = luigi.ChoiceParameter(choices=DATASETS.keys())

    def output(self):
        """One ``LocalTarget`` per file declared in :data:`DATASETS`."""
        return [
            luigi.LocalTarget(
                os.path.join(self.output_path, self.dataset, os.path.basename(p))
            )
            for p in DATASETS[self.dataset]
        ]

    def run(self):
        """Delegate to :func:`load_dataset` (downloads with caching)."""
        load_dataset(self.dataset, output_path=self.output_path)
