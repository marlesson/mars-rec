"""Luigi tasks for downloading raw data and producing train/val/test CSVs.

Every project's data pipeline starts here. A project subclasses
:class:`BasePrepareDataFrames`, points it at the raw-data Luigi task,
and overrides :meth:`BasePrepareDataFrames.read_data_frame_path` (and
optionally a few hooks). The framework then handles splitting,
balancing, filtering and writing the three split CSVs that the
training step consumes.

:class:`BaseDownloadDataset` is a thin helper for the common case of
"download one file by URL into ``output().path``" — projects with
multi-file or zip-based downloads typically write their own
``luigi.Task`` instead.
"""

import abc
import itertools
import math
import os
from typing import List, Tuple, Dict, Optional

import luigi
import numpy as np
import pandas as pd
import requests
import random
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm


class BaseDownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    """Abstract Luigi task that downloads a single file from ``url``.

    Subclasses implement :meth:`output` (return a ``LocalTarget``
    pointing at the destination path) and :attr:`url` (the source
    URL). The default :meth:`run` streams the response with a tqdm
    progress bar and raises ``ConnectionError`` if the byte count does
    not match the ``Content-Length`` header.

    Most real projects bypass this class because they fetch zip files
    or multiple resources — the BaseDownloadDataset is kept for the
    "one file, one URL" shortcut.
    """

    @abc.abstractmethod
    def output(self) -> luigi.LocalTarget:
        """Return the ``LocalTarget`` for the downloaded file."""

    @property
    @abc.abstractmethod
    def url(self) -> str:
        """URL of the file to download."""

    def run(self):
        """Stream the URL to ``self.output().path`` with a progress bar."""
        r = requests.get(self.url, stream=True)
        output_path = self.output().path

        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        wrote = 0
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        with open(output_path, "wb") as f:
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


class BasePrepareDataFrames(luigi.Task, metaclass=abc.ABCMeta):
    """Abstract Luigi task that produces ``train.csv``, ``val.csv``, ``test.csv``.

    Configured almost entirely via Luigi parameters (overridable from
    the CLI / training task), this base task handles:

    - reading the raw DataFrame (via :meth:`read_data_frame`),
    - applying ``eq_filters`` / ``neq_filters`` / ``isin_filters``,
    - splitting into train / val / test (random, time-based, column-
      stratified or k-fold),
    - optional over/under-sampling via ``imbalanced-learn``,
    - writing the three split CSVs to ``dataset_dir`` with hashed
      file names so the cache survives parameter changes.

    A project subclasses this and overrides at least:

    - :attr:`dataset_dir` — where to write the CSVs,
    - :attr:`read_data_frame_path` — path to the raw CSV produced by
      the upstream download task,
    - optionally :meth:`requires`, :attr:`timestamp_property`,
      :attr:`metadata_data_frame_path`, :meth:`transform_data_frame`,
      :meth:`create_available_items`.

    See ``examples/movielens/data.py`` for a minimal working example.
    """

    session_test_size: float = luigi.FloatParameter(default=0.10)
    test_size: float = luigi.FloatParameter(default=0.2)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.IntParameter(default=5)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "time", "column", "k_fold"], default="time")
    column_stratification: str = luigi.OptionalParameter(default=None)
    test_split_type: str = luigi.ChoiceParameter(choices=["random", "time"], default="random")
    item_column: str = luigi.Parameter(default="none")
    available_items_column_name: str = luigi.Parameter(default="available_items")
    n_splits: int = luigi.IntParameter(default=10)
    split_index: int = luigi.IntParameter(default=0)
    val_size: float = luigi.FloatParameter(default=0.2)
    sampling_strategy: str = luigi.ChoiceParameter(choices=["oversample", "undersample", "none"], default="none")
    balance_fields: List[str] = luigi.ListParameter(default=[])
    sampling_proportions: Dict[str, Dict[str, float]] = luigi.DictParameter(default={})
    use_sampling_in_validation: bool = luigi.BoolParameter(default=False)
    eq_filters: Dict[str, any] = luigi.DictParameter(default={})
    neq_filters: Dict[str, any] = luigi.DictParameter(default={})
    isin_filters: Dict[str, any] = luigi.DictParameter(default={})
    seed: int = luigi.IntParameter(default=42)

    VALIDATION_DATA = "VALIDATION_DATA"
    TRAIN_DATA = "TRAIN_DATA"
    TEST_GENERATOR = "TEST_GENERATOR"

    @property
    @abc.abstractmethod
    def dataset_dir(self) -> str:
        """Directory where the train / val / test CSVs are written.

        Subclass responsibility — the framework joins this with the
        hashed file names produced by :meth:`output`.
        """

    @property
    @abc.abstractmethod
    def read_data_frame_path(self) -> Optional[str]:
        """Path to the raw input CSV consumed by :meth:`read_data_frame`."""

    def read_data_frame(self) -> pd.DataFrame:
        """Load the raw DataFrame from :attr:`read_data_frame_path`.

        Override when the upstream task produces something other than a
        plain CSV (e.g. parquet) or when on-disk decoding requires
        special handling.
        """
        return pd.read_csv(self.read_data_frame_path)

    @property
    def stratification_property(self) -> str:
        """Column name used to stratify random splits / k-fold splits.

        Default ``None`` disables stratification. Set this in the
        subclass when ``test_split_type="random"`` should preserve a
        target distribution.
        """

    @property
    def timestamp_property(self) -> Optional[str]:
        """Column name used to sort rows for time-based splits.

        Required when ``dataset_split_method="time"`` or
        ``test_split_type="time"``. Returning ``None`` (the default)
        signals the splits should *not* be time-ordered.
        """
        return None

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        """Optional path to a per-item metadata CSV.

        When set, the file is appended as a 4th
        :class:`luigi.LocalTarget` in :meth:`output` and is read by
        ``SupervisedModelTraining`` to build the
        ``embeddings_for_metadata`` lookup consumed by
        :attr:`ProjectConfig.metadata_columns`. Return ``None`` to
        disable the metadata channel entirely.
        """
        return None

    def output(self) -> Tuple[luigi.LocalTarget, ...]:
        """Return the (train, val, test [, metadata]) ``LocalTarget`` tuple.

        File names are hashed on the Luigi parameters so that toggling
        ``val_size``, ``test_split_type``, ``seed`` etc. produces a
        distinct cache key without manual invalidation.
        """
        task_hash = self.task_id
        if self.dataset_split_method == "k_fold":
            output = (
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "train_[%dof%d]_test=%s_%d_%s_%s.csv"
                        % (
                            self.split_index + 1,
                            self.n_splits,
                            self.test_split_type,
                            self.seed,
                            self.sampling_strategy,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "val_[%dof%d]_test=%s_%d_%s.csv"
                        % (
                            self.split_index + 1,
                            self.n_splits,
                            self.test_split_type,
                            self.seed,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "test_%.2f_test=%s_%d_%s.csv" % (self.test_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
            )
        else:
            output = (
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "train_%.2f_test=%s_%d_%s_%s.csv"
                        % (
                            self.val_size,
                            self.test_split_type,
                            self.seed,
                            self.sampling_strategy,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "val_%.2f_test=%s_%d_%s.csv" % (self.val_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "test_%.2f_test=%s_%d_%s.csv" % (self.test_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
            )
        if self.metadata_data_frame_path:
            return output + (luigi.LocalTarget(self.metadata_data_frame_path),)

        return output

    def run(self):
        """Read the raw DataFrame, filter, split and write the CSVs.

        The ``eq_filters`` / ``neq_filters`` / ``isin_filters``
        DictParameters are applied in order before the split, so
        downstream caller code can keep the same parameter hash across
        different filter combinations.
        """
        os.makedirs(self.dataset_dir, exist_ok=True)

        df = self.read_data_frame()

        for field, value in self.eq_filters.items():
            df = df[df[field] == value]

        for field, value in self.neq_filters.items():
            df = df[df[field] != value]

        for field, value in self.isin_filters.items():
            df = df[df[field].isin(value)]

        self.create_available_items(df)

        self.train_df, self.val_df, self.test_df = self.split_dataset(df)

        self.transform_data_frame(self.train_df, data_key=self.TRAIN_DATA).to_csv(self.output()[0].path, index=False)
        self.transform_data_frame(self.val_df, data_key=self.VALIDATION_DATA).to_csv(self.output()[1].path, index=False)
        self.transform_data_frame(self.test_df, data_key=self.TEST_GENERATOR).to_csv(self.output()[2].path, index=False)

    def split_dataset(self, df):
        """Split ``df`` into train / val / test according to the Luigi params.

        - ``test_size > 0``: carves out a held-out test set first,
          either randomly or time-ordered (controlled by
          ``test_split_type``).
        - ``val_size > 0``: carves out a validation set from the
          remaining train rows, with the strategy chosen by
          ``dataset_split_method`` (holdout / column / time / k-fold).
        - If ``sampling_strategy != "none"``: applies over/under-
          sampling on train (and optionally val) via the
          ``imbalanced-learn`` samplers.
        """
        if self.test_size:
            if self.test_split_type == "random":
                train_df, test_df = self.random_train_test_split(df, test_size=self.test_size)
            else:
                train_df, test_df = self.time_train_test_split(df, test_size=self.test_size)
        else:
            train_df, test_df = df, df[:0]

        if self.val_size:
            if self.dataset_split_method == "holdout":
                train_df, val_df = self.random_train_test_split(train_df, test_size=self.val_size)
            elif self.dataset_split_method == "column":
                train_df, val_df = self.column_train_test_split(train_df, test_size=self.val_size)
            elif self.dataset_split_method == "time":
                train_df, val_df = self.time_train_test_split(train_df, test_size=self.val_size)
            else:
                train_df, val_df = self.kfold_split(train_df)
        else:
            train_df, val_df = train_df, train_df[:0]

        if self.sampling_strategy != "none":
            train_df = self.balance_dataset(train_df)
            if self.use_sampling_in_validation:
                val_df = self.balance_dataset(val_df)

        return train_df, val_df, test_df

    def create_available_items(self, df: pd.DataFrame) -> None:
        """Populate the per-row candidate-slate column when missing.

        Default behaviour: when the column named by
        ``available_items_column_name`` is absent, attach a list of up
        to 99 randomly sampled items plus the row's positive item.

        Override in the subclass to skip this entirely (return early)
        or to build the slate from a domain-specific source (popularity
        cohort, same-genre items, etc.).
        """
        if self.available_items_column_name not in df.columns:

            def add_items(item, item_unique):
                items = random.sample(item_unique, min(99, len(item_unique)))
                items.append(item)
                items = np.unique(items).tolist()
                return items

            item_unique = list(df[self.item_column].drop_duplicates().values)
            df[self.available_items_column_name] = df.apply(
                lambda row: add_items(row[self.item_column], item_unique), axis=1
            )

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        """Hook: transform each split before it is written to CSV.

        Receives the train / val / test DataFrame (identified by
        ``data_key`` set to one of :attr:`TRAIN_DATA`,
        :attr:`VALIDATION_DATA`, :attr:`TEST_GENERATOR`) and returns
        the DataFrame to persist. Default is a no-op pass-through.
        """
        return df

    def kfold_split(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """K-fold stratified split returning the ``split_index``-th fold.

        Uses ``sklearn.model_selection.StratifiedKFold`` with
        ``n_splits`` and ``stratification_property``.
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        train_indices, val_indices = next(
            itertools.islice(
                skf.split(
                    df,
                    df[self.stratification_property] if self.stratification_property else None,
                ),
                self.split_index,
                self.split_index + 1,
            )
        )
        train_df, test_df = df.iloc[train_indices], df.iloc[val_indices]
        return train_df, test_df

    def column_train_test_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by partitioning the unique values of ``column_stratification``.

        Useful when rows sharing a key (e.g. ``user_id`` or
        ``session_id``) must end up entirely on one side of the split.
        """
        # Cast to numpy: in pandas ≥ 3.0 (pyarrow-backed strings),
        # ``.unique()`` returns a ``pyarrow.ChunkedArray`` that sklearn's
        # ``train_test_split`` cannot slice (raises
        # ``TypeError: only integer scalar arrays can be converted to a
        # scalar index``).
        keys = np.asarray(df[self.column_stratification].unique())
        keys_train, keys_test = train_test_split(keys, test_size=test_size, random_state=self.seed)
        return (
            df[df[self.column_stratification].isin(keys_train)],
            df[df[self.column_stratification].isin(keys_test)],
        )

    def time_train_test_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based split: oldest ``1 - test_size`` rows go to train.

        Sorts on :attr:`timestamp_property` first; the split is purely
        positional (no shuffle) so the test set is always the most
        recent slice.
        """
        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        size = len(df)
        cut = int(size - size * test_size)

        return df.iloc[:cut], df.iloc[cut:]

    def random_train_test_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Plain ``train_test_split`` shuffle, optionally stratified.

        Stratification kicks in when
        :attr:`stratification_property` is set; otherwise it's a
        uniform shuffle keyed on ``seed``.
        """
        return train_test_split(
            df,
            test_size=test_size,
            stratify=df[self.stratification_property] if self.stratification_property else None,
            random_state=self.seed,
        )

    def _create_sampling_strategy(self, df: pd.DataFrame, balance_field: str):
        if balance_field in self.sampling_proportions:
            field_unique_values, counts = np.unique(df[balance_field], return_counts=True)
            if self.sampling_strategy == "oversample":
                min_num_samples = max(counts)
                min_proportion = min(self.sampling_proportions[balance_field].values())
                return {
                    field_value: int(
                        self.sampling_proportions[balance_field][field_value] * min_num_samples / min_proportion
                    )
                    for field_value in field_unique_values
                }
            elif self.sampling_strategy == "undersample":
                max_num_samples = min(counts)
                max_proportion = max(self.sampling_proportions[balance_field].values())
                return {
                    field_value: int(
                        self.sampling_proportions[balance_field][field_value] * max_num_samples / max_proportion
                    )
                    for field_value in field_unique_values
                }
        return "auto"

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply over- or under-sampling on ``df`` for each ``balance_fields`` entry.

        Iterates the configured ``balance_fields`` and resamples row
        indices using ``RandomOverSampler`` / ``RandomUnderSampler``
        from imbalanced-learn. Sampling proportions can be customised
        per field through the ``sampling_proportions`` DictParameter.
        Returns ``df`` unchanged when ``sampling_strategy`` is ``"none"``.
        """
        sampler = dict(oversample=RandomOverSampler, undersample=RandomUnderSampler)

        random_sampler_cls = sampler.get(self.sampling_strategy)

        if random_sampler_cls is None:
            return df
        index_resampled = df.index
        for balance_field in self.balance_fields:
            random_sampler = random_sampler_cls(
                sampling_strategy=self._create_sampling_strategy(df, balance_field),
                random_state=self.seed,
            )
            index_resampled, _ = random_sampler.fit_resample(
                np.array(index_resampled).reshape(-1, 1),
                df.loc[index_resampled][balance_field],
            )
            index_resampled = index_resampled.flatten()

        return df.loc[index_resampled]
