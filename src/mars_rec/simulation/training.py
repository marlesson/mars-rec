"""Supervised training pipeline — the heart of the framework.

This module wires together every other component of mars-rec. The
public entry point is :class:`SupervisedModelTraining`, a Luigi task
that:

1. Depends on the project's :class:`BasePrepareDataFrames` subclass to
   produce ``train.csv`` / ``val.csv`` / ``test.csv``.
2. Builds an ``index_mapping`` over every ``INDEXABLE`` /
   ``INDEXABLE_ARRAY`` column.
3. Instantiates the configured PyTorch :class:`Dataset`,
   :class:`RecommenderModule` and the project's loss function.
4. Wraps the model in a ``RecommenderLightningModule`` and runs the
   ``pytorch_lightning.Trainer`` with EarlyStopping, ModelCheckpoint,
   CSVLogger and (when ``mars-rec[tensorboard]`` is installed)
   TensorBoardLogger callbacks.
5. Evaluates on the held-out test set and saves every artifact
   (``params.json``, ``metadata.json``, ``model/``, ``training/`` and
   ``eval/test/``) under
   ``output/models/SupervisedModelTraining/results/<task_id>/``; the layout
   is defined in :mod:`mars_rec.utils.files`.

The class hierarchy is::

    _BaseModelTraining        (data / index_mapping / Dataset plumbing)
    └── TorchModelTraining    (Lightning module / Trainer / training loop)
        └── SupervisedModelTraining   (user-facing public entry point)

Two helper functions are also exposed:

- :func:`load_torch_model_training_from_task_dir` /
  :func:`load_torch_model_training_from_task_id` — re-hydrate a
  trained run for inference / evaluation.
- :func:`run_evaluate` — re-evaluate a trained run on the original
  test split or on an external file (CSV or parquet). Backs the
  ``mars-rec evaluate`` CLI command.
"""

import abc
import gc
import json
import logging
import os
import pickle
import platform
import random
import shutil
import sys
import time
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Type, Dict, Iterable, List, Optional, Tuple, Any
import luigi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn.init import xavier_normal_
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mars_rec.data.candidate_features import (
    FN_PARAM as CANDIDATE_FN_PARAM,
    LOOKUP_PARAM as CANDIDATE_LOOKUP_PARAM,
    CandidateFeatureFn,
    resolve_candidate_lookup,
    validate_candidate_feature_fn,
)
from mars_rec.data.dataset import (
    preprocess_interactions_data_frame,
    preprocess_metadata_data_frame,
    literal_eval_array_columns,
    InteractionsDataset,
)
from mars_rec.evaluation import (
    DEFAULT_KS,
    InteractionHistory,
    build_ranking_queries,
    evaluate_rankings,
    positive_mask,
)
from mars_rec.filtering import AvailableItemsColumnProvider, CandidateProvider
from mars_rec.meta_config import IOType, ProjectConfig
from mars_rec.model.abstract import RecommenderModule
from mars_rec.torch.data import NoAutoCollationDataLoader, FasterBatchSampler
from mars_rec.torch.init import lecun_normal_init, he_init
from mars_rec.torch.lightning_module import (
    EmbeddingRecommenderLightningModule,
    RecommenderLightningModule,
)
from mars_rec.torch.loss import (
    BCEWithLogitsLossSmooth,
    BPRLoss,
    CosineSimilarityLoss,
    HardNegInfoNCELoss,
    ImplicitFeedbackBCELoss,
    InfoNCELoss,
    FocalLoss,
    LambdaRankLoss,
    RankNetLoss,
    DummyLoss,
)
from mars_rec.torch.optimizer import RAdam
from mars_rec.torch.summary import summary
from mars_rec.utils.files import (
    CHECKPOINT_NAME,
    DEFAULT_EVAL_NAME,
    ensure_parent_dir,
    find_checkpoint,
    get_architecture_path,
    get_eval_dir,
    get_eval_plot_path,
    get_eval_results_path,
    get_history_path,
    get_index_mapping_path,
    get_metadata_path,
    get_model_dir,
    get_params,
    get_params_path,
    get_plot_history_path,
    get_sample_train_keys_path,
    get_sample_train_path,
    get_sample_train_scores_path,
    get_summary_path,
    get_task_dir,
    get_tensorboard_logdir,
    get_test_set_predictions_path,
    get_success_marker_path,
    get_training_logs_dir,
    is_complete_run,
    list_artifacts,
    validate_eval_name,
)
from mars_rec.utils.index_mapping import (
    create_index_mapping,
    create_index_mapping_from_arrays,
    transform_with_indexing,
    map_array,
    real_indices,
)
from mars_rec.utils.plot import (
    plot_history,
    plot_scores,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from mars_rec.utils.logging import banner, setup_logging
from mars_rec.utils.reflection import load_attr

setup_logging()
logger = logging.getLogger(__name__)


def _tensorboard_available() -> bool:
    """Whether Lightning's ``TensorBoardLogger`` has a backend to write with."""
    import importlib.util

    return any(importlib.util.find_spec(name) is not None for name in ("tensorboard", "tensorboardX"))


# Suppress internal Lightning deprecation about LeafSpec (fixed in future PL version)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

TORCH_OPTIMIZERS = dict(
    adam=Adam,
    adamw=AdamW,
    rmsprop=RMSprop,
    sgd=SGD,
    adadelta=Adadelta,
    adagrad=Adagrad,
    adamax=Adamax,
    radam=RAdam,
)
TORCH_LR_SCHEDULERS = dict(
    reduce_on_plateau=ReduceLROnPlateau,
    cosine_annealing=CosineAnnealingLR,
    cosine_warm_restarts=CosineAnnealingWarmRestarts,
    step=StepLR,
)
TORCH_LOSS_FUNCTIONS = dict(
    mse=nn.MSELoss,
    nll=nn.NLLLoss,
    bce=nn.BCELoss,
    bce_logits=nn.BCEWithLogitsLoss,
    bce_logits_smooth=BCEWithLogitsLossSmooth,
    bpr=BPRLoss,
    ranknet=RankNetLoss,
    lambdarank=LambdaRankLoss,
    ce=nn.CrossEntropyLoss,
    mlm=nn.MultiLabelMarginLoss,
    implicit_feedback_bce=ImplicitFeedbackBCELoss,
    focal_loss=FocalLoss,
    dummy=DummyLoss,
    # Embedding-similarity losses — use with EmbeddingRecommenderLightningModule
    infonce=InfoNCELoss,
    cosine_similarity=CosineSimilarityLoss,
    hardneg_infonce=HardNegInfoNCELoss,
)

# Loss functions that require EmbeddingRecommenderLightningModule (pred_emb, target_emb).
_EMBEDDING_LOSSES = {"infonce", "cosine_similarity", "hardneg_infonce"}
TORCH_ACTIVATION_FUNCTIONS = dict(relu=F.relu, selu=F.selu, tanh=torch.tanh, sigmoid=torch.sigmoid, linear=F.linear)
TORCH_WEIGHT_INIT = dict(lecun_normal=lecun_normal_init, he=he_init, xavier_normal=xavier_normal_)
TORCH_DROPOUT_MODULES = dict(dropout=nn.Dropout, alpha=nn.AlphaDropout)

SEED = 42

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

TRAIN_DATA = "train_data"
VAL_DATA = "val_data"
TEST_DATA = "test_data"


class _BaseModelTraining(luigi.Task, metaclass=abc.ABCMeta):
    """Internal base: data preparation, index mapping, Dataset plumbing.

    Holds every Luigi parameter that controls the *data* side of the
    pipeline (filters, splits, sampling, dataset_class). Not meant to
    be instantiated directly — use :class:`SupervisedModelTraining`.

    Subclasses expose cached properties (``train_data_frame``,
    ``val_data_frame``, ``test_data_frame``, ``index_mapping``,
    ``train_dataset``, …) that perform the heavy lifting on first
    access; :meth:`invalidate_cache` discards them so a re-run of the
    same Luigi task in the same Python process picks up fresh data.
    """

    project: str = luigi.Parameter(
        description="Should be like config.my_project_config",
    )

    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.IntParameter(default=5)
    session_test_size: float = luigi.FloatParameter(default=0.10)
    test_size: float = luigi.FloatParameter(default=0.2)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "column", "time", "k_fold"], default="time")
    test_split_type: str = luigi.ChoiceParameter(choices=["random", "time"], default="random")
    val_size: float = luigi.FloatParameter(default=0.2)
    n_splits: int = luigi.IntParameter(default=5)
    split_index: int = luigi.IntParameter(default=0)
    data_frames_preparation_extra_params: dict = luigi.DictParameter(default={})
    sampling_strategy: str = luigi.ChoiceParameter(choices=["oversample", "undersample", "none"], default="none")
    balance_fields: List[str] = luigi.ListParameter(default=[])
    sampling_proportions: Dict[str, Dict[str, float]] = luigi.DictParameter(default={})
    use_sampling_in_validation: bool = luigi.BoolParameter(default=False)
    eq_filters: Dict[str, any] = luigi.DictParameter(default={})
    neq_filters: Dict[str, any] = luigi.DictParameter(default={})
    isin_filters: Dict[str, any] = luigi.DictParameter(default={})
    seed: int = luigi.IntParameter(default=SEED)
    observation: str = luigi.Parameter(default="")
    load_index_mapping_path: str = luigi.OptionalParameter(default=None)
    negative_proportion: float = luigi.FloatParameter(0.0)

    def invalidate_cache(self):
        """Delete all cached outputs for this task (the whole :attr:`task_dir`) and its data preparation dependency."""
        # Invalidate data preparation outputs
        prep = self.prepare_data_frames
        for target in prep.output():
            if hasattr(target, "path") and os.path.exists(target.path):
                os.remove(target.path)
        # Invalidate own output
        if os.path.exists(self.task_dir):
            shutil.rmtree(self.task_dir, ignore_errors=True)

    @property
    def cache_attrs(self):
        return [
            "_test_dataset",
            "_val_dataset",
            "_train_dataset",
            "_test_data_frame",
            "_val_data_frame",
            "_train_data_frame",
            "_metadata_data_frame",
            "_cached_trained_module",
            "_cached_dataset_forward",
        ]

    def requires(self):
        return self.prepare_data_frames

    @property
    def prepare_data_frames(self):
        return self.project_config.prepare_data_frames_task(
            session_test_size=self.session_test_size,
            sample_size=self.sample_size,
            minimum_interactions=self.minimum_interactions,
            test_size=self.test_size,
            dataset_split_method=self.dataset_split_method,
            test_split_type=self.test_split_type,
            val_size=self.val_size,
            n_splits=self.n_splits,
            split_index=self.split_index,
            sampling_strategy=self.sampling_strategy,
            sampling_proportions=self.sampling_proportions,
            balance_fields=self.balance_fields or self.project_config.default_balance_fields,
            item_column=self.project_config.item_column.name,
            available_items_column_name=self.project_config.available_items_column_name,
            use_sampling_in_validation=self.use_sampling_in_validation,
            eq_filters=self.eq_filters,
            neq_filters=self.neq_filters,
            isin_filters=self.isin_filters,
            seed=self.seed,
            **self.data_frames_preparation_extra_params,
        )

    @property
    def task_dir(self) -> str:
        """Directory this run writes its artifacts to (see :mod:`mars_rec.utils.files`)."""
        return get_task_dir(self.__class__, self.task_id)

    def output(self):
        """The ``_SUCCESS`` marker, written only once the run has finished.

        A directory left behind by a run that was killed mid-way therefore no
        longer counts as done. Use :attr:`task_dir` for the run directory.
        """
        return luigi.LocalTarget(get_success_marker_path(self.task_dir))

    def complete(self) -> bool:
        """Done when ``_SUCCESS`` exists (or for a finished run of an older version, see
        :func:`~mars_rec.utils.files.is_complete_run`)."""
        return is_complete_run(self.task_dir)

    @property
    def project_config(self) -> ProjectConfig:
        if not hasattr(self, "_project_config"):
            self._project_config = deepcopy(load_attr(self.project, ProjectConfig))
        return self._project_config

    def _save_params(self):
        with open(get_params_path(self.task_dir), "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

    @property
    def train_data_frame_path(self) -> str:
        return self.input()[0].path

    @property
    def val_data_frame_path(self) -> str:
        return self.input()[1].path

    @property
    def test_data_frame_path(self) -> str:
        return self.input()[2].path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        if len(self.input()) > 3:
            return self.input()[3].path
        else:
            return None

    @property
    def metadata_data_frame(self) -> Optional[pd.DataFrame]:
        if not hasattr(self, "_metadata_data_frame"):
            self._metadata_data_frame = (
                pd.read_csv(self.metadata_data_frame_path) if self.metadata_data_frame_path else None
            )
            if self._metadata_data_frame is not None:
                literal_eval_array_columns(self._metadata_data_frame, self.project_config.metadata_columns)
            transform_with_indexing(self._metadata_data_frame, self.index_mapping, self.project_config)
        return self._metadata_data_frame

    @property
    def embeddings_for_metadata(self) -> Optional[Dict[str, np.ndarray]]:
        if not hasattr(self, "_embeddings_for_metadata"):
            self._embeddings_for_metadata = (
                preprocess_metadata_data_frame(self.metadata_data_frame, self.project_config)
                if self.metadata_data_frame is not None
                else None
            )
        return self._embeddings_for_metadata

    @property
    def dataset_read_columns(self) -> List[str]:
        except_columns = [c.name for c in self.project_config.metadata_columns]
        columns = [c.name for c in self.project_config.all_columns if c.name not in except_columns]
        if self.project_config.available_items_column_name:
            columns += [self.project_config.available_items_column_name]
        # Include sample_keys_columns so the raw (pre-indexing) values for
        # session_id / user_id / etc. survive the read; transform_with_indexing
        # ignores columns it does not own, so extras pass through unchanged.
        sample_keys = list(getattr(self.project_config, "sample_keys_columns", None) or [])
        for c in sample_keys:
            if c not in columns:
                columns.append(c)
        return columns

    @property
    def train_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_train_data_frame"):
            logger.info("Loading train data frame: %s", self.train_data_frame_path)
            self._train_data_frame = preprocess_interactions_data_frame(
                pd.read_csv(self.train_data_frame_path, usecols=self.dataset_read_columns), self.project_config
            )

            # Snapshot the RAW values of sample_keys_columns BEFORE
            # transform_with_indexing rewrites e.g. user_id / track_id to
            # integer indices. The snapshot is indexed identically to
            # _train_data_frame (same pandas index), so downstream we can
            # slice it with the same sample_idx as the post-indexed df and
            # recover the original UUID-style identifiers per row.
            sample_keys = list(getattr(self.project_config, "sample_keys_columns", None) or [])
            keys_present = [c for c in sample_keys if c in self._train_data_frame.columns]
            if keys_present:
                self._train_data_frame_raw_keys = self._train_data_frame[keys_present].copy()

            transform_with_indexing(self._train_data_frame, self.index_mapping, self.project_config)

        return self._train_data_frame

    @property
    def val_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_val_data_frame"):
            logger.info("Loading val data frame: %s", self.val_data_frame_path)
            self._val_data_frame = preprocess_interactions_data_frame(
                pd.read_csv(self.val_data_frame_path, usecols=self.dataset_read_columns), self.project_config
            )

            transform_with_indexing(self._val_data_frame, self.index_mapping, self.project_config)

        return self._val_data_frame

    @property
    def test_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_test_data_frame"):
            logger.info("Loading test data frame: %s", self.test_data_frame_path)
            self._test_data_frame = preprocess_interactions_data_frame(
                pd.read_csv(self.test_data_frame_path, usecols=self.dataset_read_columns), self.project_config
            )

            transform_with_indexing(self._test_data_frame, self.index_mapping, self.project_config)

        return self._test_data_frame

    def get_data_frame_for_indexing(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.read_csv(self.train_data_frame_path, usecols=self.dataset_read_columns),
                pd.read_csv(self.val_data_frame_path, usecols=self.dataset_read_columns),
            ]
        ).drop_duplicates()

    def get_data_frame_interactions(self) -> pd.DataFrame:
        """Distinct raw ``(user, item)`` pairs of train + val.

        Used to flag test rows whose pair was seen in training. Reads only
        the two key columns, so array columns are never parsed here.
        """
        key_columns = [self.project_config.user_column.name, self.project_config.item_column.name]
        return (
            pd.concat(
                [
                    pd.read_csv(self.train_data_frame_path, usecols=key_columns),
                    pd.read_csv(self.val_data_frame_path, usecols=key_columns),
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    @property
    def index_mapping_path(self) -> Optional[str]:
        if self.load_index_mapping_path:
            return get_index_mapping_path(self.load_index_mapping_path)
        return get_index_mapping_path(self.source_task_dir)

    @property
    def index_mapping(self) -> Dict[str, Dict[Any, int]]:
        if not hasattr(self, "_index_mapping"):
            logger.info("Building index_mapping (path=%s)", self.index_mapping_path)

            if os.path.exists(self.index_mapping_path):
                with open(self.index_mapping_path, "rb") as f:
                    self._index_mapping = pickle.load(f)
                logger.info(
                    "  reusing %d existing column mappings",
                    len(self._index_mapping),
                )
            else:
                self._index_mapping = {}

            keys_in_map = list(self._index_mapping.keys())
            project_all_columns = [c for c in self.project_config.all_columns if c.name not in keys_in_map]

            new_scalar = [c.name for c in project_all_columns if c.type == IOType.INDEXABLE and not c.same_index_as]
            new_array = [
                c.name for c in project_all_columns if c.type == IOType.INDEXABLE_ARRAY and not c.same_index_as
            ]

            # Reading and parsing train+val is the expensive part: skip it
            # when the reused mapping already covers every indexable column.
            self._creating_index_mapping = True
            df = (
                preprocess_interactions_data_frame(self.get_data_frame_for_indexing(), self.project_config)
                if new_scalar or new_array
                else None
            )

            for column in project_all_columns:
                if column.type == IOType.INDEXABLE and not column.same_index_as:
                    self._index_mapping[column.name] = create_index_mapping(df[column.name].values)

            self._index_mapping.update(
                {
                    column.name: create_index_mapping_from_arrays(df[column.name].values)
                    for column in project_all_columns
                    if column.type == IOType.INDEXABLE_ARRAY and not column.same_index_as
                }
            )

            for column in project_all_columns:
                if column.same_index_as:
                    self._index_mapping[column.name] = self._index_mapping[column.same_index_as]

            del self._creating_index_mapping
            del df

            with open(ensure_parent_dir(get_index_mapping_path(self.source_task_dir)), "wb") as f:
                pickle.dump(self._index_mapping, f)

            if new_scalar or new_array:
                logger.info(
                    "  indexed %d scalar + %d array columns (total=%d)",
                    len(new_scalar),
                    len(new_array),
                    len(self._index_mapping),
                )

        return self._index_mapping

    @property
    def reverse_index_mapping(self) -> Dict[str, Dict[int, Any]]:
        """``{column: {int_index: raw_value}}``, the inverse of :attr:`index_mapping`.

        Index ``0`` (unknown) maps back to ``0``. Built once per
        ``index_mapping`` object: the candidate sampler reads it for every
        test row, and rebuilding it each time costs O(catalog) per row.
        """
        index_mapping = self.index_mapping
        cached = self.__dict__.get("_reverse_index_mapping_cache")
        if cached is None or cached[0] is not index_mapping:
            rev = {key: {value_: key_ for key_, value_ in mapping.items()} for key, mapping in index_mapping.items()}
            for key in rev:
                rev[key][0] = 0
            cached = (index_mapping, rev)
            self._reverse_index_mapping_cache = cached
        return cached[1]

    @property
    def train_dataset(self) -> Dataset:
        if not hasattr(self, "_train_dataset"):
            emb = self.embeddings_for_metadata
            if emb is not None:
                shapes = {k: v.shape for k, v in emb.items()}
                print(f"[META] embeddings_for_metadata keys={list(emb.keys())} shapes={shapes}", flush=True)
            else:
                print(
                    f"[META] embeddings_for_metadata is None! metadata_path={self.metadata_data_frame_path} input_len={len(self.input())}",
                    flush=True,
                )
            self._train_dataset = self.project_config.dataset_class(
                data_frame=self.train_data_frame,
                embeddings_for_metadata=emb,
                project_config=self.project_config,
                index_mapping=self.index_mapping,
                negative_proportion=self.negative_proportion,
                data_key=TRAIN_DATA,
                **self.project_config.dataset_extra_params,
            )
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        if not hasattr(self, "_val_dataset"):
            self._val_dataset = self.project_config.dataset_class(
                data_frame=self.val_data_frame,
                embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config,
                index_mapping=self.index_mapping,
                negative_proportion=self.negative_proportion,
                data_key=VAL_DATA,
                **self.project_config.dataset_extra_params,
            )
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        if not hasattr(self, "_test_dataset"):
            self._test_dataset = self.project_config.dataset_class(
                data_frame=self.test_data_frame,
                embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config,
                index_mapping=self.index_mapping,
                negative_proportion=0.0,
                data_key=TEST_DATA,
                **self.project_config.dataset_extra_params,
            )
        return self._test_dataset

    @property
    def n_users(self) -> int:
        if not hasattr(self, "_n_users"):
            self._n_users = max(self.index_mapping[self.project_config.user_column.name].values()) + 1
        return self._n_users

    @property
    def n_items(self) -> int:
        if not hasattr(self, "_n_items"):
            self._n_items = max(self.index_mapping[self.project_config.item_column.name].values()) + 1
        return self._n_items

    def index_mapping_max_value(self, key: str) -> int:
        return max(self.index_mapping[key].values())

    @abc.abstractmethod
    def train(self):
        pass

    def cache_cleanup(self, keep: Iterable[str] = ()):
        """Drop the cached data frames / datasets listed in :attr:`cache_attrs`.

        :param keep: attribute names to leave in place (e.g. the validation
            split when it is about to be evaluated).
        """
        keep = set(keep)
        for a in self.cache_attrs:
            if a not in keep and hasattr(self, a):
                delattr(self, a)

    def seed_everything(self):
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def before_run(self):
        self.index_mapping

    def run(self):
        os.makedirs(self.task_dir, exist_ok=True)

        self.seed_everything()
        self.before_run()
        self._save_params()

        try:
            self.train()
        except Exception:
            shutil.rmtree(self.task_dir)
            raise
        finally:
            gc.collect()

        # Last step: from now on Luigi (and ``complete()``) treat the run as done.
        with open(get_success_marker_path(self.task_dir), "w"):
            pass


class TorchModelTraining(_BaseModelTraining, metaclass=abc.ABCMeta):
    """Internal base: PyTorch / Lightning training loop on top of the data plumbing.

    Adds every Luigi parameter that controls *training* (model class,
    optimiser, scheduler, loss, batch size, epochs, callbacks, …) and
    implements :meth:`train` / :meth:`evaluate`.
    :class:`SupervisedModelTraining` is the concrete subclass exposed
    to users.
    """

    recommender_module_class: str = luigi.Parameter(
        description="Should be like mars_rec.model.base_model.LogisticRegression",
    )
    recommender_extra_params: Dict[str, Any] = luigi.DictParameter(default={})

    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda", "mps"], default=DEFAULT_DEVICE)

    batch_size: int = luigi.IntParameter(default=500)
    epochs: int = luigi.IntParameter(default=100)
    optimizer: str = luigi.ChoiceParameter(choices=TORCH_OPTIMIZERS.keys(), default="adam")
    optimizer_params: dict = luigi.DictParameter(default={})
    learning_rate: float = luigi.FloatParameter(1e-3)
    # Optional LR scheduler. Empty string = legacy behaviour (no scheduler).
    # ``lr_scheduler_params`` is forwarded to the scheduler ctor; for
    # ``reduce_on_plateau`` the special keys ``monitor`` (default ``val_loss``),
    # ``interval`` (default ``epoch``), and ``frequency`` (default 1) are
    # consumed by Lightning rather than by the scheduler itself.
    lr_scheduler: str = luigi.ChoiceParameter(choices=list(TORCH_LR_SCHEDULERS.keys()) + [""], default="")
    lr_scheduler_params: dict = luigi.DictParameter(default={})
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="bce")
    loss_function_params: dict = luigi.DictParameter(default={})
    gradient_norm_clipping: float = luigi.FloatParameter(default=0.0)
    gradient_norm_clipping_type: float = luigi.IntParameter(default=2)
    early_stopping_patience: int = luigi.IntParameter(default=5)
    early_stopping_min_delta: float = luigi.FloatParameter(default=1e-3)
    monitor_metric: str = luigi.Parameter(default="val_loss")
    monitor_mode: str = luigi.Parameter(default="min")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)
    # Lightning precision string (``"32-true"``, ``"16-mixed"``, ``"bf16-mixed"``).
    # Empty string = auto: ``"16-mixed"`` on CUDA, ``"32-true"`` elsewhere.
    # T4/L4/A100 all benefit from FP16 mixed; bf16 only on Ampere+.
    precision: str = luigi.Parameter(default="")

    # Cap on the ranking queries scored against the full catalog (and on the
    # ROC-AUC rows); ``None`` evaluates every query.
    sample_size_eval: int = luigi.OptionalIntParameter(default=None)
    ranking_metrics: bool = luigi.BoolParameter(default=False)

    metrics = luigi.ListParameter(default=["loss"])

    # Optional ``"module.path:func_name"`` reference to a callable that
    # builds per-row sampling weights for the training DataLoader. Called
    # once per epoch with ``(train_dataset, train_data_frame)`` and must
    # return a 1-D tensor / numpy array of length ``len(train_dataset)``.
    # When set, the training sampler switches from uniform shuffle to
    # ``torch.multinomial`` over those weights — useful for matching a
    # downstream evaluation distribution. Default ""
    # = legacy uniform shuffle.
    train_sample_weights_provider: str = luigi.Parameter(default="")

    @property
    def resources(self):
        return {"cuda": 1} if self.device == "cuda" else {}

    def _get_accelerator(self) -> str:
        if self.device == "cuda":
            return "gpu"
        elif self.device == "mps":
            return "mps"
        return "cpu"

    def _get_devices(self):
        if self.device == "cuda":
            return [self.device_id]
        return "auto"

    @property
    def device_id(self) -> Optional[int]:
        """CUDA device index used when ``device == "cuda"``, else ``None``.

        Always the first visible GPU. To run tasks in parallel on different
        GPUs, give each process its own ``CUDA_VISIBLE_DEVICES``.
        """
        return 0 if self.device == "cuda" else None

    @property
    def module_class(self) -> Type[RecommenderModule]:
        if not hasattr(self, "_module_class"):
            self._module_class = load_attr(self.recommender_module_class, RecommenderModule)
        return self._module_class

    @property
    def all_recommender_extra_params(self) -> Dict[str, Any]:
        return self.recommender_extra_params

    def create_module(self) -> nn.Module:
        """Instantiate the model (``recommender_module_class``).

        The class receives ``project_config``, ``index_mapping`` and every key of
        ``recommender_extra_params`` as keyword arguments. Override to build the
        model differently (e.g. to load pretrained weights).
        """
        return self.module_class(
            project_config=self.project_config,
            index_mapping=self.index_mapping,
            **self.all_recommender_extra_params,
        )

    def train(self):
        """Fit the model and write the run artifacts.

        Builds the data loaders and the model, writes ``summary.txt``,
        ``sample_train.csv`` and ``architecture.svg``, runs ``trainer.fit``, then
        writes ``history.csv`` and ``plot_history.jpg`` and calls
        :meth:`after_fit`, :meth:`evaluate` and the metadata/cleanup steps.
        """
        self._train_start_time = time.time()

        if self.device == "cuda":
            torch.cuda.set_device(self.device_id)

        banner("Training", logger=logger)
        logger.info(
            "task=%s | model=%s | device=%s",
            self.task_id,
            self.recommender_module_class.split(".")[-1],
            self.device,
        )
        self._split_stats = self._compute_split_stats()
        logger.info(
            "splits: train=%d | val=%d | test=%d",
            self._split_stats["train_size"],
            self._split_stats["val_size"],
            self._split_stats["test_size"],
        )
        logger.info(
            "training: epochs=%s | batch_size=%s | lr=%s | loss=%s | optimizer=%s",
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.loss_function,
            self.optimizer,
        )

        train_loader = self.get_train_generator()
        val_loader = self.get_val_generator()
        module = self.create_module()

        summary_path = ensure_parent_dir(get_summary_path(self.task_dir))
        with open(summary_path, "w") as summary_file:
            with redirect_stdout(summary_file):
                sample_input = self.get_sample_batch()
                summary(module, sample_input)
            summary(module, sample_input)

        # Sample with a fixed seed so sample_train.csv and the optional
        # sample_train_keys.csv reference the SAME 100 rows. self.seed is
        # already used to seed training, so reusing it keeps the sample
        # reproducible per task_id.
        sample_data = self.train_data_frame.sample(
            100,
            replace=True,
            random_state=int(self.seed),
        )
        sample_data.to_csv(ensure_parent_dir(get_sample_train_path(self.task_dir)))

        # Companion artifact: raw (pre-indexing) values of the project-declared
        # sample_keys_columns, sliced at the same sample_idx so each row in
        # sample_train.csv has a row_idx match in sample_train_keys.csv. The
        # raw snapshot was captured in train_data_frame before
        # transform_with_indexing — see the `_train_data_frame_raw_keys`
        # assignment. Skipped silently when the project did not declare
        # sample_keys_columns; downstream --val-prediction modes hard-fail
        # with an actionable message in that case.
        if hasattr(self, "_train_data_frame_raw_keys"):
            sample_keys = self._train_data_frame_raw_keys.loc[sample_data.index].copy()
            sample_keys.insert(0, "row_idx", range(len(sample_keys)))
            sample_keys.to_csv(
                ensure_parent_dir(get_sample_train_keys_path(self.task_dir)),
                index=False,
            )

        # Architecture diagram — always emit ``architecture.svg`` next to the
        # checkpoint. trace=True captures real tensor shapes via forward hooks.
        # Failures are logged and swallowed: a diagram never blocks training.
        from mars_rec.visualization import generate_architecture_diagram

        viz_params = {
            "project": self.project,
            "recommender_module_class": self.recommender_module_class,
            "loss_function": self.loss_function,
            "recommender_extra_params": self.recommender_extra_params,
        }
        try:
            generate_architecture_diagram(
                module,
                self.project_config,
                viz_params,
                get_architecture_path(self.task_dir),
                index_mapping=self.index_mapping,
                trace=True,
                device=self.device,
            )
        except Exception:  # noqa: BLE001 — never let viz crash training
            logger.warning("Architecture diagram generation failed — continuing", exc_info=True)

        lit_module = self.create_lightning_module(module)
        log_steps = max(1, min(50, len(train_loader)))
        trainer = self.create_trainer(log_every_n_steps=log_steps)

        try:
            trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user — finalising current state.")

        self._convert_history_csv(trainer)

        history_df = pd.read_csv(get_history_path(self.task_dir))

        model_name = self.recommender_module_class.split(".")[-1]
        plot_meta = self._plot_metadata()
        history_fig = plot_history(
            history_df,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=plot_meta,
        )
        history_fig.savefig(
            ensure_parent_dir(get_plot_history_path(self.task_dir)),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(history_fig)

        self.after_fit()
        self.evaluate()
        self._save_eval_results(self._eval_results_dict())
        self._save_metadata()
        self.cache_cleanup()

    def get_sample_batch(self):
        """Return the inputs of the first training sample, used for ``summary.txt``."""
        return default_convert(self.train_dataset[0][0])

    def after_fit(self):
        """Hook called right after ``trainer.fit``, before :meth:`evaluate`.

        Scores the ``sample_train.csv`` rows and writes ``sample_train_scores.csv``
        (a no-op when ``sample_train.csv`` is absent). Subclasses that override
        it should call ``super().after_fit()`` to keep this behaviour.
        """
        try:
            self._save_sample_train_scores()
        except Exception:  # noqa: BLE001 — log + continue, don't abort training
            logger.exception("sample_train_scores save failed — continuing")

    def _compute_split_stats(self) -> Dict[str, int]:
        """Row counts of the three splits plus distinct users / items in train."""
        train_df = self.train_data_frame
        stats = {
            "train_size": len(train_df),
            "val_size": len(self.val_data_frame),
            "test_size": len(self.test_data_frame),
        }
        if self.project_config.item_column:
            stats["n_items"] = len(train_df[self.project_config.item_column.name].unique())
        if self.project_config.user_column:
            stats["n_users"] = len(train_df[self.project_config.user_column.name].unique())
        return stats

    def _save_metadata(self):
        """Save comprehensive metadata.json with all information needed to
        register the training run and replicate the experiment."""
        from datetime import datetime, timezone

        output_dir = self.task_dir
        train_duration = time.time() - getattr(self, "_train_start_time", time.time())

        # --- Model info --- (reuse the loaded model; building one only to count is costly)
        module = getattr(self, "_cached_trained_module", None)
        if module is None:
            module = self.create_module()
        n_parameters = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        # --- Training history (best epoch) ---
        history_path = get_history_path(output_dir)
        best_epoch = None
        best_metrics = {}
        history_summary = {}
        if os.path.exists(history_path):
            hist_df = pd.read_csv(history_path)
            if len(hist_df) > 0:
                history_summary["total_epochs"] = len(hist_df)
                # Find best epoch by monitor metric
                monitor = self.monitor_metric
                if monitor in hist_df.columns:
                    valid_vals = hist_df[monitor].dropna()
                    if len(valid_vals) > 0:
                        if self.monitor_mode == "min":
                            best_epoch = int(valid_vals.idxmin())
                        else:
                            best_epoch = int(valid_vals.idxmax())
                        best_metrics = {
                            col: float(hist_df.loc[best_epoch, col])
                            for col in hist_df.columns
                            if col != "epoch" and not pd.isna(hist_df.loc[best_epoch, col])
                        }
                        history_summary["best_epoch"] = best_epoch
                        history_summary["best_metrics"] = best_metrics
                # Last epoch metrics
                last = hist_df.iloc[-1]
                history_summary["last_epoch_metrics"] = {
                    col: float(last[col]) for col in hist_df.columns if col != "epoch" and not pd.isna(last[col])
                }

        # --- Data splits info ---
        # Recorded by train() while the splits were loaded: evaluate() has
        # released them by now, and reloading only to count rows is costly.
        split_stats = getattr(self, "_split_stats", None) or self._compute_split_stats()
        data_info = {
            "train_size": split_stats["train_size"],
            "val_size": split_stats["val_size"],
            "test_size": split_stats["test_size"],
            "n_features": len(self.project_config.input_columns),
            "input_columns": [col.name for col in self.project_config.input_columns],
            "output_column": self.project_config.output_column.name,
        }
        for key in ("n_items", "n_users"):
            if key in split_stats:
                data_info[key] = split_stats[key]

        # --- Artifacts list ---
        artifacts = list_artifacts(output_dir) if os.path.isdir(output_dir) else []

        # --- Build metadata ---
        metadata = {
            "run": {
                "task_id": self.task_id,
                "task_class": self.__class__.__name__,
                "started_at": datetime.fromtimestamp(self._train_start_time, tz=timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(train_duration, 2),
                "seed": self.seed,
                "observation": self.observation if hasattr(self, "observation") else "",
            },
            "project": {
                "name": self.project,
                "dataset_split_method": self.dataset_split_method,
                "test_split_type": self.test_split_type,
                "test_size": self.test_size,
                "val_size": self.val_size,
            },
            "model": {
                "class": self.recommender_module_class,
                "class_name": self.recommender_module_class.split(".")[-1],
                "n_parameters": n_parameters,
                "n_trainable_parameters": n_trainable,
                "extra_params": dict(self.recommender_extra_params),
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer,
                "optimizer_params": dict(self.optimizer_params) if self.optimizer_params else {},
                "learning_rate": self.learning_rate,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": dict(self.lr_scheduler_params) if self.lr_scheduler_params else {},
                "loss_function": self.loss_function,
                "loss_function_params": dict(self.loss_function_params) if self.loss_function_params else {},
                "metrics": list(self.metrics),
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "monitor_metric": self.monitor_metric,
                "monitor_mode": self.monitor_mode,
                "gradient_norm_clipping": self.gradient_norm_clipping,
                "device": self.device,
            },
            "data": data_info,
            "history": history_summary,
            "evaluation": getattr(self, "_eval_results", {}),
            "ranking_metrics": getattr(self, "_ranking_metrics", {}),
            "ranking_baseline": getattr(self, "_ranking_baseline_metrics", {}),
            "ranking_protocol": getattr(self, "_ranking_protocol", {}),
            "roc_auc_metrics": getattr(self, "_roc_auc_metrics", {}),
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "pytorch_lightning": pl.__version__,
                "platform": platform.platform(),
            },
            "artifacts": artifacts,
        }

        metadata_path = get_metadata_path(output_dir)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata saved → %s", metadata_path)
        if best_epoch is not None:
            logger.info(
                "Best epoch: %d (%s=%.4f)",
                best_epoch,
                self.monitor_metric,
                best_metrics.get(self.monitor_metric, float("nan")),
            )
        logger.info("Training duration: %.1fs", train_duration)

    @property
    def source_task_dir(self) -> str:
        """The actual task directory on disk.

        When a task is reconstructed for re-evaluation, Luigi may compute
        a different ``task_id`` hash, so :attr:`task_dir` may not match
        the real directory. Set ``_source_task_dir`` to override.
        """
        return getattr(self, "_source_task_dir", None) or self.task_dir

    @property
    def eval_output_dir(self) -> str:
        """Directory where evaluation artifacts (metrics, predictions, plots) are saved.

        Defaults to ``eval/test/`` inside ``source_task_dir``; :func:`run_evaluate`
        overrides it via ``_eval_output_dir`` for named / external evaluations.
        """
        return getattr(self, "_eval_output_dir", None) or get_eval_dir(self.source_task_dir, DEFAULT_EVAL_NAME)

    def _eval_results_dict(self) -> Dict[str, Any]:
        """Metrics of the latest evaluation, in the ``metrics.json`` format."""
        results: Dict[str, Any] = {}
        if hasattr(self, "_eval_results"):
            results["val_metrics"] = self._eval_results
        if hasattr(self, "_ranking_metrics"):
            results["ranking_metrics"] = self._ranking_metrics
            results["ranking_baseline"] = getattr(self, "_ranking_baseline_metrics", {})
            results["ranking_protocol"] = getattr(self, "_ranking_protocol", {})
        if hasattr(self, "_roc_auc_metrics"):
            results["roc_auc_metrics"] = self._roc_auc_metrics
        return results

    def _save_eval_results(self, results: Dict[str, Any]) -> str:
        """Merge ``results`` into ``metrics.json`` of :attr:`eval_output_dir` and return its path.

        Merging (instead of overwriting) keeps the test metrics when only the
        validation split is re-evaluated, and vice versa.
        """
        path = ensure_parent_dir(get_eval_results_path(self.eval_output_dir))
        merged: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as f:
                merged = json.load(f)
        merged.update(results)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        logger.info("Eval results saved → %s", path)
        return path

    def _evaluate_on_dataloader(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run Lightning validation on any dataloader and return metrics dict."""
        module = self._get_cached_trained_module()
        lit_module = self.create_lightning_module(module)
        trainer = pl.Trainer(
            accelerator=self._get_accelerator(),
            devices=self._get_devices(),
            logger=False,
            enable_progress_bar=True,
        )
        results = trainer.validate(lit_module, dataloaders=dataloader)
        return results[0] if results else {}

    def evaluate(self):
        """Evaluate the trained model on the validation split.

        Logs every metric of ``metrics`` and keeps the results for
        ``metadata.json``. Test-set ranking metrics are computed earlier, in
        :meth:`SupervisedModelTraining.after_fit`.
        """
        # Free the train / test splits before evaluating, but keep the
        # validation split (about to be scored) and the loaded model.
        self.cache_cleanup(keep=("_val_data_frame", "_val_dataset", "_cached_trained_module"))
        val_loader = self.get_val_generator()
        banner("Evaluation (validation set)", logger=logger)
        self._eval_results = self._evaluate_on_dataloader(val_loader)
        for metric, value in self._eval_results.items():
            try:
                logger.info("  %-28s %.4f", metric, float(value))
            except (TypeError, ValueError):
                logger.info("  %-28s %s", metric, value)

    def create_lightning_module(self, module: nn.Module) -> RecommenderLightningModule:
        """Wrap ``module`` in a Lightning module with the configured loss, optimizer and scheduler.

        Embedding losses (``infonce``, ``cosine_similarity``, ``hardneg_infonce``) select
        :class:`mars_rec.torch.lightning_module.EmbeddingRecommenderLightningModule`;
        every other loss uses :class:`mars_rec.torch.lightning_module.RecommenderLightningModule`.
        """
        loss_fn = self._get_loss_function()
        optimizer = self._get_optimizer(module)
        lr_scheduler_config = self._get_lr_scheduler_config(optimizer)
        cls = (
            EmbeddingRecommenderLightningModule
            if self.loss_function in _EMBEDDING_LOSSES
            else RecommenderLightningModule
        )
        return cls(
            module,
            optimizer,
            loss_fn,
            self.metrics,
            loss_name=self.loss_function,
            lr_scheduler_config=lr_scheduler_config,
        )

    def create_trainer(self, log_every_n_steps: int = 50) -> pl.Trainer:
        """Build the ``pl.Trainer`` (callbacks, loggers, accelerator, precision).

        Override to add Lightning callbacks or change trainer flags.
        """
        callbacks = self._get_callbacks()
        loggers = self._get_loggers()

        trainer_kwargs = dict(
            max_epochs=self.epochs,
            callbacks=callbacks,
            logger=loggers,
            accelerator=self._get_accelerator(),
            devices=self._get_devices(),
            enable_progress_bar=True,
            use_distributed_sampler=False,
            log_every_n_steps=log_every_n_steps,
            num_sanity_val_steps=0,
            precision=self._resolve_precision(),
        )

        if self.gradient_norm_clipping:
            trainer_kwargs["gradient_clip_val"] = self.gradient_norm_clipping
            trainer_kwargs["gradient_clip_algorithm"] = "norm"

        return pl.Trainer(**trainer_kwargs)

    def _resolve_precision(self) -> str:
        if self.precision:
            return self.precision
        return "16-mixed" if self.device == "cuda" else "32-true"

    def _get_loss_function(self):
        params = dict(self.loss_function_params) if self.loss_function_params else {}
        # nn.BCEWithLogitsLoss requires `pos_weight` as a torch.Tensor (not a
        # Python float), so allow users to pass a scalar via --loss-function-params
        # and convert it here.
        if self.loss_function in {"bce_logits", "bce_logits_smooth", "bce"} and "pos_weight" in params:
            pw = params["pos_weight"]
            if not isinstance(pw, torch.Tensor):
                params["pos_weight"] = torch.tensor([float(pw)], dtype=torch.float32)
        return TORCH_LOSS_FUNCTIONS[self.loss_function](**params)

    def _get_optimizer(self, module) -> Optimizer:
        return TORCH_OPTIMIZERS[self.optimizer](module.parameters(), lr=self.learning_rate, **self.optimizer_params)

    def _get_lr_scheduler_config(self, optimizer: Optimizer) -> Optional[dict]:
        if not self.lr_scheduler:
            return None
        params = dict(self.lr_scheduler_params) if self.lr_scheduler_params else {}
        # Lightning-only keys — pop before passing the rest to the scheduler ctor.
        monitor = params.pop("monitor", "val_loss")
        interval = params.pop("interval", "epoch")
        frequency = int(params.pop("frequency", 1))
        scheduler_cls = TORCH_LR_SCHEDULERS[self.lr_scheduler]
        scheduler = scheduler_cls(optimizer, **params)
        cfg: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }
        # ReduceLROnPlateau is the only built-in that needs Lightning to push
        # a monitored metric into ``scheduler.step(metric)``.
        if self.lr_scheduler == "reduce_on_plateau":
            cfg["monitor"] = monitor
        return cfg

    def _get_callbacks(self):
        callbacks = [
            *self._get_extra_callbacks(),
            ModelCheckpoint(
                dirpath=get_model_dir(self.task_dir),
                filename=CHECKPOINT_NAME,
                # Always overwrite ``weights.ckpt`` instead of writing ``weights-v1.ckpt``.
                enable_version_counter=False,
                save_top_k=1,
                monitor=self.monitor_metric,
                mode=self.monitor_mode,
            ),
            EarlyStopping(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                monitor=self.monitor_metric,
                mode=self.monitor_mode,
            ),
        ]

        # Auto-inject the DNS callback when the project uses a dataset that
        # exposes ``set_model`` (DNS-style score-driven hard negatives).
        # Without this the sampler silently falls back to uniform random.
        ds_cls = self.project_config.dataset_class
        if hasattr(ds_cls, "set_model"):
            from mars_rec.torch.callbacks import DynamicNegativeMiningCallback

            callbacks.append(DynamicNegativeMiningCallback())

        return callbacks

    def _get_loggers(self):
        # CSVLogger is the source of ``history.csv`` and always runs.
        loggers = [CSVLogger(save_dir=get_training_logs_dir(self.task_dir), name="lightning_logs", version="")]
        # TensorBoard logging is optional (``pip install "mars-rec[tensorboard]"``):
        # Lightning's TensorBoardLogger needs ``tensorboard`` or ``tensorboardX``.
        if _tensorboard_available():
            loggers.append(
                TensorBoardLogger(
                    save_dir=os.path.dirname(get_tensorboard_logdir(self.task_id)),
                    name=os.path.basename(get_tensorboard_logdir(self.task_id)),
                )
            )
        else:
            logger.debug("tensorboard/tensorboardX not installed: TensorBoard logging disabled")
        return loggers

    def _get_extra_callbacks(self):
        return []

    def _convert_history_csv(self, trainer: pl.Trainer):
        """Convert Lightning CSVLogger output to the format expected by plot_history."""
        csv_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                csv_logger = logger
                break

        if csv_logger is None:
            return

        log_dir = csv_logger.log_dir
        metrics_file = os.path.join(log_dir, "metrics.csv")

        if not os.path.exists(metrics_file):
            return

        df = pd.read_csv(metrics_file)

        if "epoch" not in df.columns:
            return

        # Lightning logs train and val metrics on separate rows per step.
        # Group by epoch and take the last non-NaN value for each metric.
        epoch_df = df.groupby("epoch").last().reset_index()

        # Drop step column if present
        if "step" in epoch_df.columns:
            epoch_df = epoch_df.drop(columns=["step"])

        epoch_df.to_csv(ensure_parent_dir(get_history_path(self.task_dir)), index=False)

    def get_trained_module(self) -> nn.Module:
        """Rebuild the model and load the best ``weights*.ckpt`` of this run, in eval mode.

        :raises FileNotFoundError: when the task directory has no checkpoint.
        """
        module = self.create_module().to(self.torch_device)

        # Find Lightning checkpoint
        ckpt_path = self._find_best_checkpoint()
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.torch_device, weights_only=False)
            # Lightning saves with 'module.' prefix from RecommenderLightningModule.
            # Filter to keep only the recommender's tensors — discards top-level
            # buffers like `loss_fn.pos_weight` (registered when BCEWithLogitsLoss
            # gets a pos_weight tensor) that don't belong to the recommender.
            state_dict = {
                k[len("module.") :]: v for k, v in checkpoint["state_dict"].items() if k.startswith("module.")
            }
            module.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.source_task_dir}")

        module.eval()
        return module

    def _get_cached_trained_module(self) -> nn.Module:
        """:meth:`get_trained_module`, loaded from the checkpoint once per task instance.

        Every post-training step (sample scores, test predictions, ROC-AUC,
        validation) scores with the same weights, so they share one copy
        instead of rebuilding the model and re-reading the checkpoint each
        time. Only call it after ``trainer.fit`` has written the checkpoint.
        Released by :meth:`cache_cleanup`.
        """
        if not hasattr(self, "_cached_trained_module"):
            self._cached_trained_module = self.get_trained_module().to(self.torch_device)
        # Lightning's validate() may leave the module in train mode.
        return self._cached_trained_module.eval()

    def _forward_dataset(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score every row of ``dataset`` with the trained model.

        Returns ``(scores, targets)`` on CPU, concatenated along the batch
        dimension (``scores`` is ``[N]`` or ``[N, C]``). The result for the
        last dataset scored is cached, so the test-set plots and ROC-AUC,
        which read the same split, run the forward pass once.
        """
        cached = self.__dict__.get("_cached_dataset_forward")
        if cached is not None and cached[0] is dataset:
            return cached[1]

        batch_sampler = FasterBatchSampler(dataset, self.batch_size, shuffle=False)
        generator = NoAutoCollationDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )
        model = self._get_cached_trained_module()

        all_scores: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        with torch.no_grad():
            for x, y in generator:
                input_params = x if isinstance(x, (list, tuple)) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]
                all_scores.append(model.recommendation_score(*input_params).cpu())
                target = y[0] if isinstance(y, (list, tuple)) else y
                all_targets.append(target.cpu())

        result = (torch.cat(all_scores, dim=0), torch.cat(all_targets, dim=0))
        self._cached_dataset_forward = (dataset, result)
        return result

    def _find_best_checkpoint(self) -> Optional[str]:
        """Find the best Lightning checkpoint file (see :func:`~mars_rec.utils.files.find_checkpoint`)."""
        return find_checkpoint(self.source_task_dir)

    def _save_sample_train_scores(self) -> None:
        """Score the 100 rows in ``sample_train.csv`` with the trained model
        and persist ``sample_train_scores.csv``.

        Companion to ``sample_train.csv`` + ``sample_train_keys.csv`` used by
        downstream ``--val-prediction`` validation flows: they let inference
        scripts re-build the same input rows from the original session data,
        score them through the same checkpoint, and verify the submission
        pipeline reproduces the training-time logit per row.

        Saves the raw model logit (no sigmoid / clipping), so any downstream
        comparison can apply the same transformation deterministically on
        both sides if probabilities are preferred. Skipped silently when
        ``sample_train.csv`` is not present (e.g. ``run()`` aborted before
        the sample was written).
        """
        sample_path = get_sample_train_path(self.task_dir)
        if not os.path.exists(sample_path):
            return

        df_sample = pd.read_csv(sample_path)
        # The CSV uses the pandas index as its first (unnamed) column; drop
        # so the on-disk row order matches the in-memory frame.
        if df_sample.columns[0].startswith("Unnamed"):
            df_sample = df_sample.iloc[:, 1:].reset_index(drop=True)

        # sample_train.csv values come from train_data_frame AFTER
        # transform_with_indexing, but to_csv writes INDEXABLE_ARRAY cells
        # as Python repr strings (e.g. "[2, 5, 0, 0]"). Re-running
        # preprocess_interactions_data_frame parses those back into lists,
        # matching the in-memory shape InteractionsDataset expects.
        df_sample = preprocess_interactions_data_frame(df_sample, self.project_config)

        sample_dataset = InteractionsDataset(
            df_sample,
            self.embeddings_for_metadata,
            self.project_config,
            self.index_mapping,
        )
        batch_sampler = FasterBatchSampler(sample_dataset, self.batch_size, shuffle=False)
        generator = NoAutoCollationDataLoader(
            sample_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,  # 100 rows — spawn overhead dominates
            pin_memory=False,
        )

        model = self._get_cached_trained_module()

        all_scores: List[float] = []
        with torch.no_grad():
            for x, _ in generator:
                input_params = x if isinstance(x, (list, tuple)) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]
                scores_tensor = model.recommendation_score(*input_params)
                all_scores.extend(scores_tensor.cpu().numpy().reshape(-1).tolist())

        if len(all_scores) != len(df_sample):
            logger.warning(
                "sample_train_scores: %d scores produced for %d rows — "
                "row_idx alignment may be off; writing what we have.",
                len(all_scores),
                len(df_sample),
            )

        out_path = ensure_parent_dir(get_sample_train_scores_path(self.task_dir))
        pd.DataFrame(
            {
                "row_idx": list(range(len(all_scores))),
                "train_logit": all_scores,
            }
        ).to_csv(out_path, index=False)
        logger.info("sample_train_scores saved → %s", out_path)

    @property
    def torch_device(self) -> torch.device:
        if not hasattr(self, "_torch_device"):
            if self.device == "cuda":
                self._torch_device = torch.device(f"cuda:{self.device_id}")
            elif self.device == "mps":
                self._torch_device = torch.device("mps")
            else:
                self._torch_device = torch.device("cpu")
        return self._torch_device

    @property
    def _effective_num_workers(self) -> int:
        return self.generator_workers

    def _resolve_train_sample_weights(self) -> Optional[Any]:
        """Import ``train_sample_weights_provider`` and call it.

        Format: ``"<module.path>:<callable_name>"``. The callable is
        invoked with ``(train_dataset, train_data_frame)`` and must
        return a 1-D tensor / numpy array of length ``len(train_dataset)``.
        Errors during import or invocation are surfaced loudly — a silent
        fallback to uniform sampling would mask a misconfigured run.
        """
        ref = (self.train_sample_weights_provider or "").strip()
        if not ref:
            return None
        if ":" not in ref:
            raise ValueError(f"train_sample_weights_provider must be 'module.path:func', got {ref!r}")
        import importlib

        module_path, func_name = ref.rsplit(":", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return func(self.train_dataset, self.train_data_frame)

    def get_train_generator(self) -> DataLoader:
        """DataLoader over the training dataset (shuffled, or weighted by ``train_sample_weights_provider``)."""
        sample_weights = self._resolve_train_sample_weights()
        batch_sampler = FasterBatchSampler(
            self.train_dataset,
            self.batch_size,
            shuffle=True if sample_weights is None else False,
            sample_weights=sample_weights,
        )
        return NoAutoCollationDataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self._effective_num_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

    def get_val_generator(self) -> Optional[DataLoader]:
        """DataLoader over the validation dataset, or ``None`` when the split is empty."""
        if len(self.val_data_frame) == 0:
            return None
        batch_sampler = FasterBatchSampler(self.val_dataset, self.batch_size, shuffle=False)
        return NoAutoCollationDataLoader(
            self.val_dataset,
            batch_sampler=batch_sampler,
            num_workers=self._effective_num_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

    def get_test_generator(self) -> DataLoader:
        """DataLoader over the test dataset (not shuffled)."""
        batch_sampler = FasterBatchSampler(self.test_dataset, self.batch_size, shuffle=False)
        return NoAutoCollationDataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self._effective_num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )


class SupervisedModelTraining(TorchModelTraining):
    """Public Luigi task: train a recommender end-to-end and evaluate it.

    Instantiate it (programmatically or via ``mars-rec run
    supervised``) with two required parameters:

    - ``project`` — dotted path to a :class:`ProjectConfig` instance
      (e.g. ``"examples.movielens.config.movielens"``).
    - ``recommender_module_class`` — dotted path to a
      :class:`RecommenderModule` subclass.

    The remaining parameters are documented in the README's
    "Training parameters" section and follow the same names as the
    public Luigi attributes (``epochs``, ``batch_size``,
    ``learning_rate``, ``loss_function``, ``negative_proportion``,
    ``device``, ``early_stopping_patience``, ``metrics``,
    ``ranking_metrics``, ``test_size``, ``val_size``,
    ``dataset_split_method``, ``test_split_type``, …).

    Programmatic usage::

        import luigi
        from mars_rec.simulation.training import SupervisedModelTraining

        task = SupervisedModelTraining(
            project="examples.movielens.config.movielens",
            recommender_module_class="examples.movielens.model.MovieLensNCF",
            recommender_extra_params={"n_factors": 32, "hidden_size": 128},
            epochs=10, batch_size=512, loss_function="bce_logits",
            negative_proportion=0.5, device="cpu",
        )
        luigi.build([task], local_scheduler=True)

    Output directory: ``output/models/SupervisedModelTraining/results/<task_id>/``.
    The ``task_id`` is hashed from every Luigi parameter, so changing
    any of them (including ``observation``) gives a fresh result
    directory while reuses are free.
    """

    @property
    def obs_columns(self) -> List[str]:
        if not hasattr(self, "_obs_columns"):
            self._obs_columns = [self.project_config.user_column.name] + [
                column.name for column in self.project_config.other_input_columns
            ]
        return self._obs_columns

    def _lazy_candidate_provider(self):
        """Build (and cache) the CandidateProvider declared by the project, if any."""
        if hasattr(self, "_cached_candidate_provider"):
            return self._cached_candidate_provider
        # ``available_items_column_name`` already has a dedicated branch below,
        # so we deliberately skip the legacy fallback here — pass an explicit
        # factory check instead.
        factory = getattr(
            self.project_config,
            "candidate_provider_factory",
            None,
        )
        if factory is None:
            self._cached_candidate_provider = None
        else:
            self._cached_candidate_provider = factory(
                self.project_config,
                self.index_mapping,
                self.project_config.dataset_extra_params or {},
            )
        return self._cached_candidate_provider

    def _full_catalog_idx(self) -> np.ndarray:
        """Cached int64 array of every real catalog item idx (see ``real_indices``)."""
        if hasattr(self, "_cached_full_catalog_idx"):
            return self._cached_full_catalog_idx
        item_map = self.index_mapping.get(
            self.project_config.item_column.name,
            {},
        )
        self._cached_full_catalog_idx = real_indices(item_map)
        return self._cached_full_catalog_idx

    def _plot_metadata(self):
        """Return metadata dict for plot annotations."""
        return {
            "lr": self.learning_rate,
            "bs": self.batch_size,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "loss": self.loss_function,
        }

    def _plot_scores(self, scores):
        model_name = self.recommender_module_class.split(".")[-1]
        fig = plot_scores(
            scores,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=self._plot_metadata(),
        )
        fig.savefig(
            ensure_parent_dir(get_eval_plot_path(self.eval_output_dir, "scores")),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _get_test_row_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model on the test dataset to get one score per row.

        Returns (y_true, y_scores) as numpy arrays.
        """
        scores, targets = self._forward_dataset(self.test_dataset)
        y_true = targets.numpy().reshape(-1)
        y_scores = scores.numpy().reshape(-1)

        # Only apply sigmoid if the loss function expects logits (model outputs raw logits)
        logit_loss_functions = {
            "bce_logits",
            "bce_logits_smooth",
            "ce",
            "focal_loss",
            "bpr",
            "ranknet",
            "lambdarank",
        }
        if self.loss_function in logit_loss_functions:
            y_scores = 1 / (1 + np.exp(-y_scores))  # sigmoid for probabilities

        # Clamp to [0, 1] for probability-based scores
        y_scores = np.clip(y_scores, 0.0, 1.0)

        return y_true, y_scores

    def _is_binary_classification(self) -> bool:
        """Check if the task is binary classification based on target values.

        Requires BOTH classes (0 and 1) to be present in the test labels.
        A test set with only positives (e.g. ``negative_proportion=0.0``)
        produces degenerate confusion/ROC/PR plots, so we treat it as
        non-binary here and skip those figures.
        """
        output_col = self.project_config.output_column.name
        unique_vals = set(self.test_data_frame[output_col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            return False
        return {0, 1}.issubset({int(v) for v in unique_vals})

    def _plot_classification_curves(self):
        """Generate confusion matrix, ROC curve, and PR curve for binary classification."""
        if not self._is_binary_classification():
            return

        y_true, y_scores = self._get_test_row_scores()

        # Defensive guard: even if _is_binary_classification() passed at the
        # data-frame level, the actually-evaluated batches may collapse to a
        # single class (e.g. all positives). CM/ROC/PR are meaningless then.
        if len(np.unique(y_true)) < 2:
            print(
                "[plots] Skipping confusion-matrix/ROC/PR: only one class "
                f"present in test y_true (unique={np.unique(y_true).tolist()})."
            )
            return

        model_name = self.recommender_module_class.split(".")[-1]
        meta = self._plot_metadata()
        out_dir = self.eval_output_dir

        # Confusion Matrix
        fig = plot_confusion_matrix(
            y_true,
            y_scores,
            threshold=0.5,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=meta,
        )
        fig.savefig(ensure_parent_dir(get_eval_plot_path(out_dir, "confusion_matrix")), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ROC Curve
        fig = plot_roc_curve(
            y_true,
            y_scores,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=meta,
        )
        fig.savefig(ensure_parent_dir(get_eval_plot_path(out_dir, "roc_curve")), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Precision-Recall Curve
        fig = plot_precision_recall_curve(
            y_true,
            y_scores,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=meta,
        )
        fig.savefig(
            ensure_parent_dir(get_eval_plot_path(out_dir, "precision_recall_curve")), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    @property
    def _resolved_candidate_feature_lookup(self) -> Optional[Dict[str, Any]]:
        """Return the ``candidate_feature_lookup`` dict (resolved once, cached).

        Uses the same resolver as ``InteractionsDataset`` and
        ``score_candidates_bce``. The ranking evaluation reads its keys to
        leave item-dependent columns out of the query context.
        """
        if not hasattr(self, "_cached_candidate_feature_lookup"):
            raw = (self.project_config.dataset_extra_params or {}).get(CANDIDATE_LOOKUP_PARAM)
            self._cached_candidate_feature_lookup = resolve_candidate_lookup(raw, self.index_mapping)
        return self._cached_candidate_feature_lookup

    @property
    def _candidate_feature_fn(self) -> Optional[CandidateFeatureFn]:
        """Return the project's ``candidate_feature_fn`` hook, if any."""
        return validate_candidate_feature_fn((self.project_config.dataset_extra_params or {}).get(CANDIDATE_FN_PARAM))

    def _fill_hist_columns(self, ob_df: pd.DataFrame) -> pd.DataFrame:
        if self.project_config.hist_view_column_name not in ob_df:
            ob_df[self.project_config.hist_view_column_name] = 0
        if self.project_config.hist_output_column_name not in ob_df:
            ob_df[self.project_config.hist_output_column_name] = 0
        return ob_df

    def _ranking_context_columns(self) -> List[str]:
        """Input columns that define a ranking query's context (see :mod:`mars_rec.evaluation`).

        Every non-item input column, minus the ones rewritten per candidate
        through ``candidate_feature_lookup`` (they describe the item, not the
        context). The candidate slate column is part of the context: rows with
        different slates rank different candidates.
        """
        item_col = self.project_config.item_column.name
        item_dependent = {k for k in (self._resolved_candidate_feature_lookup or {}) if isinstance(k, str)}
        columns = [
            c.name
            for c in self.project_config.other_input_columns
            if c.name != item_col and c.name not in item_dependent
        ]
        slate = self.project_config.available_items_column_name
        if slate and slate in self.test_data_frame.columns:
            columns.append(slate)
        return columns

    def _ranking_history(self) -> InteractionHistory:
        """Indexed train + validation interactions: seen items and popularity."""
        cfg = self.project_config
        user_col, item_col, out_col = cfg.user_column.name, cfg.item_column.name, cfg.output_column.name
        frames = []
        for path in (self.train_data_frame_path, self.val_data_frame_path):
            header = pd.read_csv(path, nrows=0).columns
            frames.append(
                pd.read_csv(path, usecols=[c for c in dict.fromkeys([user_col, item_col, out_col]) if c in header])
            )
        interactions = pd.concat(frames, ignore_index=True)

        def to_index(column: str) -> np.ndarray:
            values = interactions[column]
            if values.dtype != object:
                values = values.astype(str)
            return values.map(self.index_mapping[column]).fillna(0).to_numpy(dtype=np.int64)

        item_map = self.index_mapping[item_col]
        return InteractionHistory.from_interactions(
            users=to_index(user_col),
            items=to_index(item_col),
            positive=positive_mask(interactions, item_col, out_col),
            n_items=max(item_map.values()) + 1 if item_map else 1,
        )

    def _ranking_candidate_provider(self, query_rows: pd.DataFrame) -> Tuple[Optional[CandidateProvider], str]:
        """Candidate pool for the ranking evaluation and its protocol label.

        A project ``CandidateProvider`` (when enabled with ``--prefilter``)
        wins, then the per-row ``available_items`` slate, then the full
        catalog. ``query_rows`` gets the slate mapped to item indices in place.
        """
        if getattr(self, "_enable_prefilter", False):
            provider = self._lazy_candidate_provider()
            if provider is not None:
                return provider, provider.name()
        slate = self.project_config.available_items_column_name
        if slate and slate in query_rows.columns:
            item_map = self.index_mapping[self.project_config.item_column.name]
            query_rows[slate] = [
                [i for i in map_array(list(items), item_map) if i > 0]
                if isinstance(items, (list, tuple, np.ndarray))
                else []
                for items in query_rows[slate]
            ]
            return AvailableItemsColumnProvider(slate), "available_items"
        return None, "full_catalog"

    def _score_ranking_queries(
        self, query_rows: pd.DataFrame, provider: Optional[CandidateProvider]
    ) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """Score each query row against its candidates with the trained model."""
        from mars_rec.inference import score_candidates_bce, score_candidates_ce, score_candidates_emb

        model = self._get_cached_trained_module()
        model.eval()
        common = dict(
            df_processed=query_rows,
            project_config=self.project_config,
            index_mapping=self.index_mapping,
            full_catalog=self._full_catalog_idx(),
            candidate_provider=provider,
            device=str(self.torch_device),
            show_progress=len(query_rows) > 1,
        )
        if self.loss_function in _EMBEDDING_LOSSES and hasattr(model, "get_target_emb"):
            return score_candidates_emb(model, **common)
        if not self.project_config.item_is_input:
            return score_candidates_ce(model, **common)
        return score_candidates_bce(model, embeddings_for_metadata=self.embeddings_for_metadata, **common)

    def _save_test_set_predictions(self) -> None:
        """Rank the catalog for every test query and save predictions (+ ranking metrics).

        Follows the protocol of :mod:`mars_rec.evaluation`: positive test rows
        grouped into queries, full-catalog (or slate / provider) candidates,
        seen items excluded, random tie-break, per-user averaging.
        ``sample_size_eval`` caps the number of queries.
        """
        banner("Test-set predictions", logger=logger)
        self._plot_classification_curves()
        if not self.ranking_metrics:
            logger.info("ranking_metrics=False: skipping the top-k ranking evaluation")
            return

        cfg = self.project_config
        test_df = self.test_data_frame
        item_col = cfg.item_column.name
        relevant_col = cfg.output_column.name if not cfg.item_is_input else item_col

        queries = build_ranking_queries(
            test_df,
            user_column=cfg.user_column.name,
            item_column=relevant_col,
            output_column=cfg.output_column.name,
            context_columns=self._ranking_context_columns(),
        )
        n_queries = len(queries)
        queries = queries.sample(self.sample_size_eval, seed=self.seed)
        logger.info(
            "Ranking queries: %d (from %d positive / %d test rows)%s",
            len(queries),
            queries.n_positive_rows,
            queries.n_rows,
            f", sampled from {n_queries} (sample_size_eval)" if len(queries) < n_queries else "",
        )

        query_rows = self._fill_hist_columns(test_df.iloc[queries.positions].reset_index(drop=True).copy())
        provider, candidate_source = self._ranking_candidate_provider(query_rows)
        if provider is None:
            n_pairs = len(query_rows) * len(self._full_catalog_idx())
            logger.info(
                "  scoring %d queries × %d items (full catalog)", len(query_rows), len(self._full_catalog_idx())
            )
            if n_pairs > 50_000_000:
                logger.warning(
                    "  %d (query, item) pairs to score — set sample_size_eval to evaluate fewer queries", n_pairs
                )
        scores, candidates = self._score_ranking_queries(query_rows, provider)

        history = self._ranking_history()
        result = evaluate_rankings(
            queries,
            scores,
            candidates,
            ks=DEFAULT_KS,
            history=history,
            n_catalog_items=len(self._full_catalog_idx()),
            candidate_source=candidate_source,
            seed=self.seed,
        )
        if result.protocol["n_relevant_seen_dropped"]:
            logger.info(
                "  %d relevant items were already seen in train/val and were dropped (%d queries left empty)",
                result.protocol["n_relevant_seen_dropped"],
                result.protocol["n_queries_dropped"],
            )

        # Score distribution plot: a bounded sample of every ranked score.
        flat = np.concatenate([np.asarray(s, dtype=np.float32).ravel() for s in scores]) if scores else np.empty(0)
        if len(flat) > 200_000:
            flat = np.random.default_rng(self.seed).choice(flat, size=200_000, replace=False)
        self._plot_scores(flat.tolist())

        self._ranking_metrics = result.metrics
        self._ranking_baseline_metrics = result.baseline_metrics
        self._ranking_protocol = result.protocol
        if result.metrics:
            banner("Ranking metrics", logger=logger)
            logger.info("  %-20s %10s %10s", "metric", "model", "popularity")
            for metric_name, value in result.metrics.items():
                baseline = result.baseline_metrics.get(metric_name)
                logger.info("  %-20s %10.4f %10s", metric_name, value, "-" if baseline is None else f"{baseline:.4f}")

        self._roc_auc_metrics = self._compute_roc_auc()
        if self._roc_auc_metrics:
            banner("ROC-AUC", logger=logger)
            for metric_name, value in self._roc_auc_metrics.items():
                logger.info("  %-20s %.4f", metric_name, value)

        rev_items = self.reverse_index_mapping[item_col]
        rev_users = self.reverse_index_mapping[cfg.user_column.name]
        predictions = pd.DataFrame(
            {
                cfg.user_column.name: [rev_users.get(int(u), u) for u in result.queries.users],
                "relevant_items": [[rev_items.get(int(i), i) for i in rel] for rel in result.queries.relevant],
                "sorted_actions": [[rev_items.get(int(i), i) for i in items] for items in result.top_items],
                "action_scores": [np.round(s, 6).tolist() for s in result.top_scores],
                "n_candidates": result.n_candidates,
            }
        )
        predictions_path = ensure_parent_dir(get_test_set_predictions_path(self.eval_output_dir))
        predictions.to_csv(predictions_path, index=False)
        logger.info("Test predictions saved → %s", predictions_path)

    def _get_roc_auc_dataset(self) -> Dataset:
        """Return the dataset used for ROC-AUC inference.

        Builds a fresh subsampled dataset when ``sample_size_eval`` is
        set and smaller than the full test set; otherwise reuses
        ``self.test_dataset``. The subsample uses ``random_state=self.seed``.
        """
        if self.sample_size_eval and len(self.test_data_frame) > self.sample_size_eval:
            df = self.test_data_frame.sample(
                self.sample_size_eval,
                random_state=self.seed,
            )
            logger.info(
                "ROC-AUC subsampling %d/%d test rows (sample_size_eval, seed=%d)",
                len(df),
                len(self.test_data_frame),
                self.seed,
            )
            return self.project_config.dataset_class(
                data_frame=df,
                embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config,
                index_mapping=self.index_mapping,
                negative_proportion=0.0,
                data_key=TEST_DATA,
                **self.project_config.dataset_extra_params,
            )
        return self.test_dataset

    def _compute_roc_auc(self) -> Dict[str, float]:
        """Compute ROC-AUC on the test set.

        For multi-class models (output shape [B, C]), uses One-vs-Rest
        macro-averaged ROC-AUC with softmax probabilities.
        For binary models (output shape [B]), uses standard ROC-AUC.

        Honours ``sample_size_eval``: when set and smaller than the test
        set, runs the forward pass on a deterministic subsample (same
        ``random_state=self.seed`` used by ``_save_test_set_predictions``)
        so enabling ``--ranking-metrics`` doesn't double the test-time cost.
        """
        from sklearn.metrics import roc_auc_score

        # Same forward pass as the test-set plots when no subsample is drawn.
        all_logits, all_targets = self._forward_dataset(self._get_roc_auc_dataset())  # [N, C] or [N], [N]
        y_true = all_targets.numpy().astype(int)

        is_multiclass = all_logits.dim() == 2 and all_logits.size(1) > 1
        results = {}

        try:
            if is_multiclass:
                # Softmax → probabilities [N, C]
                y_probs = torch.softmax(all_logits, dim=1).numpy()
                n_classes = y_probs.shape[1]

                # Filter to classes actually present in y_true
                present_classes = np.unique(y_true)
                if len(present_classes) < 2:
                    logger.warning("ROC-AUC skipped: fewer than 2 classes in test set")
                    return {}

                # One-hot encode targets for present classes only
                y_true_onehot = np.zeros((len(y_true), n_classes), dtype=int)
                y_true_onehot[np.arange(len(y_true)), y_true] = 1

                results["roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_true_onehot[:, present_classes],
                        y_probs[:, present_classes],
                        multi_class="ovr",
                        average="macro",
                    )
                )
            else:
                # Binary: sigmoid → probability
                if len(np.unique(y_true)) < 2:
                    logger.warning("ROC-AUC skipped: only one class present in y_true (undefined)")
                    return {}

                logit_losses = {"bce_logits", "bce_logits_smooth", "ce", "focal_loss"}
                if self.loss_function in logit_losses:
                    y_scores = torch.sigmoid(all_logits).numpy()
                else:
                    y_scores = all_logits.numpy()
                y_scores = np.clip(y_scores, 0.0, 1.0)

                results["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            logger.exception("ROC-AUC computation failed: %s", e)

        return results

    def after_fit(self):
        """Run the base hook, then score the test split.

        When ``test_size > 0`` it writes ``eval/test/predictions.csv`` and, with
        ``ranking_metrics=True``, computes NDCG / HitRate / MRR / MAP and ROC-AUC
        for ``metadata.json``.
        """
        super().after_fit()
        if self.test_size > 0:
            self._save_test_set_predictions()


def load_torch_model_training_from_task_dir(model_cls: Type[TorchModelTraining], task_dir: str) -> TorchModelTraining:
    """Rebuild a training-task instance from its ``params.json`` on disk.

    Reads the parameters persisted at ``task_dir/params.json`` and
    instantiates ``model_cls`` with them, giving back a Luigi task
    object equivalent (parameter-wise) to the one that originally
    produced ``task_dir``. Useful to drive inference or re-evaluation
    without re-running training.
    """
    return model_cls(**get_params(task_dir))


def load_torch_model_training_from_task_id(model_cls: Type[TorchModelTraining], task_id: str) -> TorchModelTraining:
    """Like :func:`load_torch_model_training_from_task_dir` but by ``task_id``.

    Resolves ``task_id`` to its on-disk directory under
    ``output/models/<model_cls.__name__>/results/`` and forwards to
    the path-based loader.
    """
    task_dir = get_task_dir(model_cls, task_id)
    return load_torch_model_training_from_task_dir(model_cls, task_dir)


def _ranking_results(task: "SupervisedModelTraining") -> Dict[str, Any]:
    """The ranking / ROC-AUC entries a finished ``_save_test_set_predictions`` left on ``task``."""
    return {
        "ranking_metrics": getattr(task, "_ranking_metrics", {}),
        "ranking_baseline": getattr(task, "_ranking_baseline_metrics", {}),
        "ranking_protocol": getattr(task, "_ranking_protocol", {}),
        "roc_auc_metrics": getattr(task, "_roc_auc_metrics", {}),
    }


def run_evaluate(
    task_dir: str,
    metrics: Optional[List[str]] = None,
    ranking_metrics: Optional[bool] = None,
    sample_size_eval: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    eval_split: str = "all",
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    enable_prefilter: bool = False,
    eval_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-run post-training evaluation on an already-trained model.

    Supports two flows controlled by the ``data_path`` parameter:

    **Flow 1** (``data_path=None``): Re-evaluate using the original training
    splits (val/test CSVs managed by Luigi).

    **Flow 2** (``data_path="path/to/file.csv"``): Evaluate on a new external
    dataset. The file is preprocessed with the same pipeline used during
    training (``preprocess_raw_data`` from ``inference.py``), then injected
    into the task so the entire evaluation pipeline runs identically.

    Args:
        task_dir: Full path to the task result directory.
        metrics: Override the list of Lightning validation metrics
            (e.g. ``["loss", "binary_accuracy", "f1_score"]``).
        ranking_metrics: Override whether to run the top-k ranking evaluation
            (see :mod:`mars_rec.evaluation`).
        sample_size_eval: Override the maximum number of ranking queries
            (and ROC-AUC rows) evaluated.
        batch_size: Override the batch size.
        device: Override the device (``"cpu"``, ``"cuda"``, ``"mps"``).
        eval_split: Which splits to evaluate (Flow 1 only).
            ``"val"`` / ``"test"`` / ``"all"`` (default).
            Ignored when ``data_path`` is set.
        data_path: Path to an external CSV or parquet file for evaluation
            (Flow 2). Must contain the same columns as the training data,
            including the target column. If ``None``, uses original splits.
        output_dir: Directory to save evaluation artifacts (``metrics.json``,
            ``predictions.csv``, ``plots/``). Defaults to ``eval/<eval_name>/``
            inside the task directory.
        enable_prefilter: Consult ``project_config.candidate_provider_factory``
            when generating ranking-metric candidates.
        eval_name: Name of this evaluation, i.e. the ``eval/<eval_name>/``
            sub-directory it is written to. Defaults to ``"test"`` for the
            original splits and to the file name of ``data_path`` (without
            extension) for external data. ``"test"`` is reserved for the
            original splits. Only the ``"test"`` evaluation (without
            ``output_dir``) refreshes ``metadata.json``.

    Returns:
        A dict with ``"val_metrics"``, ``"ranking_metrics"``, and
        ``"roc_auc_metrics"`` keys (present depending on what was computed).
    """
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    if eval_name is None:
        eval_name = DEFAULT_EVAL_NAME if data_path is None else os.path.splitext(os.path.basename(data_path))[0]
    validate_eval_name(eval_name)
    if data_path is not None and eval_name == DEFAULT_EVAL_NAME and output_dir is None:
        raise ValueError(
            f"The evaluation name {DEFAULT_EVAL_NAME!r} is reserved for the original test split; "
            "pass another eval_name (--name) for external data."
        )

    params = get_params(task_dir)

    # Apply overrides
    if metrics is not None:
        params["metrics"] = metrics
    if ranking_metrics is not None:
        params["ranking_metrics"] = ranking_metrics
    if sample_size_eval is not None:
        params["sample_size_eval"] = sample_size_eval
    if batch_size is not None:
        params["batch_size"] = batch_size
    if device is not None:
        params["device"] = device

    task = SupervisedModelTraining(**params)

    # Point the task to the actual directory on disk (Luigi may compute
    # a different task_id hash from the re-loaded params).
    task._source_task_dir = task_dir

    # Opt-in candidate prefilter: rank the project's
    # ``candidate_provider_factory`` pool instead of the full catalog.
    task._enable_prefilter = bool(enable_prefilter)

    # Set output directory for eval artifacts: eval/<eval_name>/ unless overridden.
    effective_output_dir = output_dir or get_eval_dir(task_dir, eval_name)
    os.makedirs(effective_output_dir, exist_ok=True)
    task._eval_output_dir = effective_output_dir

    results: Dict[str, Any] = {}

    # --- Flow 2: inject external data ---
    if data_path is not None:
        from mars_rec.inference import preprocess_raw_data, load_trained_model

        banner("Evaluation on external data", logger=logger)
        logger.info("Loading external data: %s", data_path)
        if data_path.endswith(".parquet"):
            df_raw = pd.read_parquet(data_path)
        else:
            df_raw = pd.read_csv(data_path)
        logger.info("  rows=%d", len(df_raw))

        # Load project_config and index_mapping from the task dir
        _, project_config, index_mapping, _ = load_trained_model(task_dir, device=params.get("device", "cpu"))

        logger.info("Preprocessing external data ...")
        df_processed, dataset = preprocess_raw_data(df_raw, project_config, index_mapping)
        logger.info("  dataset size=%d", len(dataset))

        # Inject into task so all evaluation methods use external data
        task._test_data_frame = df_processed
        task._test_dataset = dataset
        # Override index_mapping to use the one from disk (avoid Luigi dependency)
        task._index_mapping = index_mapping

        # Point-wise metrics via Lightning
        banner("External data — point-wise metrics", logger=logger)
        test_loader = task.get_test_generator()
        task._eval_results = task._evaluate_on_dataloader(test_loader)
        results["val_metrics"] = task._eval_results
        for metric, value in task._eval_results.items():
            try:
                logger.info("  %-28s %.4f", metric, float(value))
            except (TypeError, ValueError):
                logger.info("  %-28s %s", metric, value)

        # Ranking + ROC-AUC + plots (reuses the injected test data)
        banner("External data — predictions & ranking", logger=logger)
        task._save_test_set_predictions()
        results.update(_ranking_results(task))

    # --- Flow 1: original splits ---
    else:
        if eval_split in ("val", "all"):
            banner("Re-evaluating on VALIDATION set", logger=logger)
            task.evaluate()
            results["val_metrics"] = getattr(task, "_eval_results", {})

        if eval_split in ("test", "all"):
            banner("Re-evaluating on TEST set", logger=logger)
            task._save_test_set_predictions()
            results.update(_ranking_results(task))

    # --- Save results ---
    task._save_eval_results(results)

    # metadata.json summarises the canonical test evaluation only, so an
    # external or custom-named evaluation never overwrites it.
    if output_dir is None and eval_name == DEFAULT_EVAL_NAME:
        metadata_path = get_metadata_path(task_dir)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if "val_metrics" in results:
            metadata["evaluation"] = results["val_metrics"]
        if results.get("ranking_metrics"):
            for key in ("ranking_metrics", "ranking_baseline", "ranking_protocol"):
                metadata[key] = results.get(key, {})
        if results.get("roc_auc_metrics"):
            metadata["roc_auc_metrics"] = results["roc_auc_metrics"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata updated → %s", metadata_path)

    task.cache_cleanup()

    return results
