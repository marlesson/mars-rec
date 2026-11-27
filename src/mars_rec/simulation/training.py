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
   (``weights.ckpt``, ``history.csv``, ``params.json``,
   ``metadata.json``, ``index_mapping.pkl``,
   ``test_set_predictions.csv``, training-curve plots, …) under
   ``output/models/SupervisedModelTraining/results/<task_id>/``.

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
import hashlib
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
from typing import Type, Dict, List, Optional, Tuple, Any, cast
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
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mars_rec.data.dataset import (
    preprocess_interactions_data_frame,
    preprocess_metadata_data_frame,
    literal_eval_array_columns,
    InteractionsDataset,
)
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
    get_training_logs_dir,
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
        """Delete all cached outputs for this task and its data preparation dependency."""
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
        return luigi.LocalTarget(self.task_dir)

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
        return self.get_data_frame_for_indexing()

    @property
    def index_mapping_path(self) -> Optional[str]:
        if self.load_index_mapping_path:
            return get_index_mapping_path(self.load_index_mapping_path)
        return get_index_mapping_path(self.source_task_dir)

    @property
    def index_mapping(self) -> Dict[str, Dict[Any, int]]:
        if not hasattr(self, "_index_mapping"):
            logger.info("Building index_mapping (path=%s)", self.index_mapping_path)

            self._creating_index_mapping = True
            df = preprocess_interactions_data_frame(self.get_data_frame_for_indexing(), self.project_config)

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

            with open(get_index_mapping_path(self.source_task_dir), "wb") as f:
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
        rev = {key: {value_: key_ for key_, value_ in mapping.items()} for key, mapping in self.index_mapping.items()}
        # add nothing 0
        for key, mapping in self.index_mapping.items():
            rev[key][0] = 0
        return rev

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

    def cache_cleanup(self):
        for a in self.cache_attrs:
            if hasattr(self, a):
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
        logger.info(
            "splits: train=%d | val=%d | test=%d",
            len(self.train_data_frame),
            len(self.val_data_frame),
            len(self.test_data_frame),
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

        summary_path = get_summary_path(self.task_dir)
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
        sample_data.to_csv(get_sample_train_path(self.task_dir))

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
                get_sample_train_keys_path(self.task_dir),
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
            get_plot_history_path(self.task_dir),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(history_fig)

        self.after_fit()
        self.evaluate()
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

    def _save_metadata(self):
        """Save comprehensive metadata.json with all information needed to
        register the training run and replicate the experiment."""
        from datetime import datetime, timezone

        output_dir = self.task_dir
        train_duration = time.time() - getattr(self, "_train_start_time", time.time())

        # --- Model info ---
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
        data_info = {
            "train_size": len(self.train_data_frame),
            "val_size": len(self.val_data_frame),
            "test_size": len(self.test_data_frame),
            "n_features": len(self.project_config.input_columns),
            "input_columns": [col.name for col in self.project_config.input_columns],
            "output_column": self.project_config.output_column.name,
        }
        if self.project_config.item_column:
            data_info["n_items"] = len(self.train_data_frame[self.project_config.item_column.name].unique())
        if self.project_config.user_column:
            data_info["n_users"] = len(self.train_data_frame[self.project_config.user_column.name].unique())

        # --- Artifacts list ---
        artifacts = (
            sorted(fname for fname in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fname)))
            if os.path.isdir(output_dir)
            else []
        )

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
        """Directory where evaluation artifacts (plots, predictions) are saved.

        Defaults to ``source_task_dir``, but can be overridden via
        ``_eval_output_dir`` for external-data evaluation.
        """
        return getattr(self, "_eval_output_dir", None) or get_eval_dir(self.source_task_dir, DEFAULT_EVAL_NAME)

    def _evaluate_on_dataloader(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run Lightning validation on any dataloader and return metrics dict."""
        module = self.get_trained_module()
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
        self.cache_cleanup()
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

        epoch_df.to_csv(get_history_path(self.task_dir), index=False)

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

        model = self.get_trained_module()
        model.to(self.torch_device)
        model.eval()

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

        out_path = get_sample_train_scores_path(self.task_dir)
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
    def unique_items(self) -> List[int]:
        if not hasattr(self, "_unique_items"):
            self._unique_items = list(self.index_mapping[self.project_config.item_column.name].keys())[1:-1]
            self._unique_items = [x for x in self._unique_items if str(x) != "nan"]
        return self._unique_items

    @property
    def obs_columns(self) -> List[str]:
        if not hasattr(self, "_obs_columns"):
            self._obs_columns = [self.project_config.user_column.name] + [
                column.name for column in self.project_config.other_input_columns
            ]
        return self._obs_columns

    def _get_candidate_items(self, ob: dict) -> List[Any]:
        # Deterministic per-row RNG: stable across runs/processes, and unique
        # per (user, item) so different rows still get different negative samples.
        # Uses md5 to derive an integer seed — faster than random.Random(str)
        # which internally calls sha512 on the string representation.
        user_col = self.project_config.user_column.name
        item_col = self.project_config.item_column.name
        user_val = ob.get(user_col, "")
        item_val = ob.get(item_col, "")
        seed_int = int.from_bytes(
            hashlib.md5(f"{self.seed}|{user_val}|{item_val}".encode()).digest()[:8],
            "little",
        )
        rng = random.Random(seed_int)

        # Prefer a CandidateProvider when the project declares one AND the
        # caller opted in via ``self._enable_prefilter``. The opt-in keeps
        # legacy post-train ranking metrics byte-identical for everyone who
        # hasn't explicitly asked for the filter (mars-rec evaluate
        # exposes the flag via --prefilter; submission scripts via their own
        # CLI flag). Provider returns framework-indexed item idx; we map back
        # to raw ids because _prepare_for_prediction expects raw values for
        # INDEXABLE item columns.
        if getattr(self, "_enable_prefilter", False):
            provider = self._lazy_candidate_provider()
            if provider is not None:
                full_catalog = self._full_catalog_idx()
                cand_idx = provider.get_candidates(ob, full_catalog)
                if len(cand_idx) > 0:
                    rev = self.reverse_index_mapping[self.project_config.item_column.name]
                    items = [rev[int(i)] for i in cand_idx if int(i) in rev]
                    if items:
                        return items

        if self.project_config.available_items_column_name and self.project_config.available_items_column_name in ob:
            items = ob[self.project_config.available_items_column_name]
            items = rng.sample(items, len(items))
        else:
            if (
                ob[self.project_config.item_column.name]
                in self.reverse_index_mapping[self.project_config.item_column.name]
            ):
                ob_item = self.reverse_index_mapping[self.project_config.item_column.name][
                    ob[self.project_config.item_column.name]
                ]
                items = rng.sample(self.unique_items, min(101, len(self.unique_items)))
                items.append(ob_item)
                items = list(np.unique(items))
            else:
                items = rng.sample(self.unique_items, min(100, len(self.unique_items)))

        return items

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

    def _get_cached_trained_module(self) -> nn.Module:
        """Return the trained module, loading from checkpoint only once."""
        if not hasattr(self, "_cached_trained_module"):
            self._cached_trained_module = self.get_trained_module()
            self._cached_trained_module.to(self.torch_device)
        return self._cached_trained_module

    def _get_item_scores(self, ob_dataset: Dataset) -> np.ndarray:
        """Score items using the trained model. Returns a 1-D numpy array."""
        batch_sampler = FasterBatchSampler(ob_dataset, self.batch_size, shuffle=False)
        generator = NoAutoCollationDataLoader(
            ob_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

        model = self._get_cached_trained_module()
        model.eval()
        score_arrays = []

        with torch.no_grad():
            for i, (x, _) in tqdm(enumerate(generator), total=len(generator), disable=(len(generator) <= 1)):
                input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]

                scores_tensor: torch.Tensor = model.recommendation_score(*input_params)
                score_arrays.append(scores_tensor.cpu().numpy().ravel())

        return np.concatenate(score_arrays) if score_arrays else np.array([], dtype=np.float32)

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
            get_eval_plot_path(self.eval_output_dir, "scores"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _get_test_row_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model on the test dataset to get one score per row.

        Returns (y_true, y_scores) as numpy arrays.
        """
        test_dataset = self.test_dataset
        batch_sampler = FasterBatchSampler(test_dataset, self.batch_size, shuffle=False)
        generator = NoAutoCollationDataLoader(
            test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

        model = self.get_trained_module()
        model.to(self.torch_device)
        model.eval()

        all_scores = []
        all_targets = []

        with torch.no_grad():
            for x, y in generator:
                input_params = x if isinstance(x, (list, tuple)) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]
                scores_tensor = model.recommendation_score(*input_params)
                all_scores.extend(scores_tensor.cpu().numpy().reshape(-1).tolist())

                target = y[0] if isinstance(y, (list, tuple)) else y
                all_targets.extend(target.cpu().numpy().reshape(-1).tolist())

        y_true = np.array(all_targets)
        y_scores = np.array(all_scores)

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
        fig.savefig(get_eval_plot_path(out_dir, "confusion_matrix"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ROC Curve
        fig = plot_roc_curve(
            y_true,
            y_scores,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=meta,
        )
        fig.savefig(get_eval_plot_path(out_dir, "roc_curve"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Precision-Recall Curve
        fig = plot_precision_recall_curve(
            y_true,
            y_scores,
            model_name=model_name,
            task_id=self.task_id,
            extra_info=meta,
        )
        fig.savefig(get_eval_plot_path(out_dir, "precision_recall_curve"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    @property
    def _resolved_candidate_feature_lookup(self) -> Optional[Dict[str, Any]]:
        """Return the candidate-feature lookup dict (cached, deferred resolve).

        Reads ``project_config.dataset_extra_params["candidate_feature_lookup"]``.
        Mirrors ``InteractionsDataset._resolve_candidate_lookup`` so that the
        eval-time 101-candidate ranking step (``_create_ob_data_frame``) and
        the train-time negative samplers apply the **same** override and stay
        in sync. Returns ``None`` when the config didn't supply one — every
        non-enhanced project keeps current behavior.
        """
        if hasattr(self, "_cached_candidate_feature_lookup"):
            return self._cached_candidate_feature_lookup
        raw = (self.project_config.dataset_extra_params or {}).get("candidate_feature_lookup")
        if raw is None:
            self._cached_candidate_feature_lookup = None
            return None
        if isinstance(raw, dict):
            self._cached_candidate_feature_lookup = raw
            return raw
        if not callable(raw):
            self._cached_candidate_feature_lookup = None
            return None
        import inspect as _inspect

        try:
            sig = _inspect.signature(raw)
            arity = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.kind
                    in (
                        _inspect.Parameter.POSITIONAL_ONLY,
                        _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and p.default is _inspect.Parameter.empty
                ]
            )
        except (TypeError, ValueError):
            arity = 0
        materialised = raw(self.index_mapping) if arity >= 1 else raw()
        self._cached_candidate_feature_lookup = materialised if isinstance(materialised, dict) else None
        return self._cached_candidate_feature_lookup

    def _create_ob_data_frame(self, ob: dict, item_indices: List[int]) -> pd.DataFrame:
        data = [{**ob, self.project_config.item_column.name: item_index} for item_index in item_indices]
        ob_df = pd.DataFrame(columns=self.obs_columns + [self.project_config.item_column.name], data=data)

        ob_df = self._fill_hist_columns(ob_df)

        # Per-candidate target-feature override — same semantics as the
        # negative sampler hook in ``InteractionsDataset._apply_candidate_features``.
        # Without this, every candidate in the 101-candidate ranking pool
        # inherits the positive row's popularity / album_id / tag_ids /
        # release_year_bucket / duration_bucket, and the model "ranks" by
        # detecting feature-mismatch — a shortcut that doesn't transfer to
        # full-catalog scoring.
        lookup = self._resolved_candidate_feature_lookup
        if lookup:
            item_idx_arr = np.asarray(item_indices, dtype=np.int64)
            for col in ("popularity", "release_year_bucket", "duration_bucket", "album_id"):
                if col in ob_df.columns and col in lookup:
                    ob_df[col] = lookup[col][item_idx_arr]
            if "tag_ids" in ob_df.columns and "tag_ids" in lookup:
                ob_df["tag_ids"] = [lookup["tag_ids"][i] for i in item_idx_arr]
            # Computed features — mirror the dataset hook (vectorized)
            if (
                "artist_repeat_count" in ob_df.columns
                and "prev_artist_ids" in ob_df.columns
                and "_artist_per_idx" in lookup
            ):
                cand_artists = np.asarray(
                    lookup["_artist_per_idx"][item_idx_arr],
                    dtype=np.int64,
                )
                pad_artist = lookup.get("_pad_artist", 0)
                prev_values = ob_df["prev_artist_ids"].values
                # Vectorized: stack prev arrays into a matrix and compare
                # All rows share the same prev_artist_ids (same observation),
                # so we can broadcast efficiently.
                if len(prev_values) > 0:
                    prev_matrix = np.array(prev_values.tolist(), dtype=np.int64)
                    cand_col = cand_artists.reshape(-1, 1)
                    # Mask invalid candidates
                    valid_mask = (cand_artists > 0) & (cand_artists != pad_artist)
                    counts = (prev_matrix == cand_col).sum(axis=1)
                    arc_vals = np.where(valid_mask, counts, 0)
                else:
                    arc_vals = np.zeros(len(ob_df), dtype=np.int64)
                ob_df["artist_repeat_count"] = arc_vals
            if "album_continuation" in ob_df.columns and "prev_album_ids" in ob_df.columns and "album_id" in lookup:
                cand_albums = np.asarray(
                    lookup["album_id"][item_idx_arr],
                    dtype=np.int64,
                )
                pad_album = lookup.get("_pad_album", 0)
                prev_values = ob_df["prev_album_ids"].values
                # Vectorized: find last valid album in each prev sequence
                if len(prev_values) > 0:
                    prev_matrix = np.array(prev_values.tolist(), dtype=np.int64)
                    # Find last non-pad, non-zero album per row (scan from right)
                    n_rows, seq_len = prev_matrix.shape
                    # Create mask of valid positions
                    valid_pos = (prev_matrix > 0) & (prev_matrix != pad_album)
                    # Get index of last valid position per row
                    # Use argmax on flipped valid_pos to find first True from right
                    flipped = valid_pos[:, ::-1]
                    has_valid = valid_pos.any(axis=1)
                    last_idx = seq_len - 1 - flipped.argmax(axis=1)
                    # Extract last valid album
                    last_albums = prev_matrix[np.arange(n_rows), last_idx]
                    last_albums = np.where(has_valid, last_albums, 0)
                    # Compare with candidate
                    valid_cand = (cand_albums > 0) & (cand_albums != pad_album)
                    cont_vals = np.where(valid_cand & (last_albums == cand_albums), 1, 0)
                else:
                    cont_vals = np.zeros(len(ob_df), dtype=np.int64)
                ob_df["album_continuation"] = cont_vals

        if self.project_config.output_column.name not in ob_df.columns:
            ob_df[self.project_config.output_column.name] = 1
        for auxiliar_output_column in self.project_config.auxiliar_output_columns:
            if auxiliar_output_column.name not in ob_df.columns:
                ob_df[auxiliar_output_column.name] = 0

        return ob_df

    def _fill_hist_columns(self, ob_df: pd.DataFrame) -> pd.DataFrame:
        if self.project_config.hist_view_column_name not in ob_df:
            ob_df[self.project_config.hist_view_column_name] = 0
        if self.project_config.hist_output_column_name not in ob_df:
            ob_df[self.project_config.hist_output_column_name] = 0
        return ob_df

    def _prepare_for_prediction(
        self, obs: List[Dict[str, Any]]
    ) -> Tuple[List[List[Any]], List[List[int]], List[List[float]]]:
        """Prepare observations for prediction."""
        n_obs = len(obs)
        log_every = max(1, n_obs // 10)
        t0 = time.time()

        items_list: List[List[Any]] = []
        for i, ob in enumerate(obs):
            items_list.append(self._get_candidate_items(ob))
            if (i + 1) % log_every == 0 or (i + 1) == n_obs:
                logger.info(
                    "  candidates: %d/%d (%.1fs)",
                    i + 1,
                    n_obs,
                    time.time() - t0,
                )

        # Hoist the mapping dict to avoid repeated dict lookup per observation
        if self.project_config.item_column.type in [
            IOType.INDEXABLE,
            IOType.INDEXABLE_ARRAY,
        ]:
            item_map = self.index_mapping[self.project_config.item_column.name]
            item_indices_list = [map_array(items, item_map) for items in items_list]
        else:
            item_indices_list = cast(List[List[int]], items_list)

        t1 = time.time()
        ob_dfs: List[pd.DataFrame] = []
        for i, (ob, item_indices) in enumerate(zip(obs, item_indices_list)):
            ob_dfs.append(self._create_ob_data_frame(ob, item_indices))
            if (i + 1) % log_every == 0 or (i + 1) == n_obs:
                logger.info(
                    "  ob_data_frames: %d/%d (%.1fs)",
                    i + 1,
                    n_obs,
                    time.time() - t1,
                )
        ob_lengths = [len(df) for df in ob_dfs]
        # No ``copy=False``: it is deprecated since pandas 3.0, where
        # Copy-on-Write already defers the copy.
        combined_df = pd.concat(ob_dfs, ignore_index=True)

        logger.info(
            "  combined rows=%d → running model inference",
            len(combined_df),
        )
        obs_dataset = InteractionsDataset(
            combined_df,
            self.embeddings_for_metadata,
            self.project_config,
            self.index_mapping,
        )

        t2 = time.time()
        all_item_scores = self._get_item_scores(obs_dataset)
        logger.info(
            "  inference: %d scores in %.1fs",
            len(all_item_scores),
            time.time() - t2,
        )

        # Slice numpy array into per-observation lists (numpy slicing is O(1) view)
        item_scores_list: List[List[float]] = []
        offset = 0
        for length in ob_lengths:
            item_scores_list.append(all_item_scores[offset : offset + length].tolist())
            offset += length

        return items_list, item_indices_list, item_scores_list

    def _save_test_set_predictions(self) -> None:
        """Save test set predictions using the trained model directly."""
        banner("Test-set predictions", logger=logger)

        if self.sample_size_eval and len(self.test_data_frame) > self.sample_size_eval:
            obs: List[Dict[str, Any]] = self.test_data_frame.sample(
                self.sample_size_eval, random_state=self.seed
            ).to_dict("records")
            logger.info(
                "Sampling %d/%d test rows (sample_size_eval, seed=%d)",
                len(obs),
                len(self.test_data_frame),
                self.seed,
            )
        else:
            obs: List[Dict[str, Any]] = self.test_data_frame.to_dict("records")
            logger.info("Scoring all %d test rows", len(obs))

        items_list, item_indices_list, item_scores_list = self._prepare_for_prediction(obs)

        # Rank by score (descending)
        sorted_actions_list = []
        for items, scores in zip(items_list, item_scores_list):
            sorted_pairs = sorted(zip(scores, items), reverse=True)
            sorted_actions_list.append([item for _, item in sorted_pairs])

        action_scores_list = [list(reversed(sorted(action_scores))) for action_scores in item_scores_list]

        del obs

        # Create evaluation file
        if hasattr(self, "_external_raw_df"):
            df = self._external_raw_df.copy()
        else:
            df = pd.read_csv(self.test_data_frame_path)
        if self.sample_size_eval and len(self.test_data_frame) > self.sample_size_eval:
            df = df.sample(self.sample_size_eval, random_state=self.seed)

        df["sorted_actions"] = sorted_actions_list
        df["action_scores"] = action_scores_list

        # Join with train interaction information
        df_train = self.get_data_frame_interactions()[
            [self.project_config.user_column.name, self.project_config.item_column.name]
        ].drop_duplicates()
        df_train["trained"] = 1
        df = df.merge(
            df_train, on=[self.project_config.user_column.name, self.project_config.item_column.name], how="left"
        )
        df["trained"] = df["trained"].fillna(0)

        # Add indexed information
        df["item_indexed"] = df[self.project_config.item_column.name].apply(
            lambda i: self.index_mapping[self.project_config.item_column.name][str(i)] > 0
        )

        scores = [score for item_scores in item_scores_list for score in item_scores]
        self._plot_scores(scores)
        self._plot_classification_curves()

        # Ranking metrics (only when explicitly enabled)
        if self.ranking_metrics:
            self._ranking_metrics = self._compute_ranking_metrics(df, sorted_actions_list)
            if self._ranking_metrics:
                banner("Ranking metrics", logger=logger)
                for metric_name, value in self._ranking_metrics.items():
                    logger.info("  %-20s %.4f", metric_name, value)

            self._roc_auc_metrics = self._compute_roc_auc()
            if self._roc_auc_metrics:
                banner("ROC-AUC", logger=logger)
                for metric_name, value in self._roc_auc_metrics.items():
                    logger.info("  %-20s %.4f", metric_name, value)

        predictions_path = get_test_set_predictions_path(self.eval_output_dir)
        df.to_csv(predictions_path, index=False)
        logger.info("Test predictions saved → %s", predictions_path)

    def _get_roc_auc_dataset(self) -> Dataset:
        """Return the dataset used for ROC-AUC inference.

        Builds a fresh subsampled dataset when ``sample_size_eval`` is
        set and smaller than the full test set; otherwise reuses
        ``self.test_dataset``. The subsample uses ``random_state=self.seed``
        to match the obs sampled by ``_save_test_set_predictions``.
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

        test_dataset = self._get_roc_auc_dataset()
        batch_sampler = FasterBatchSampler(test_dataset, self.batch_size, shuffle=False)
        generator = NoAutoCollationDataLoader(
            test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

        model = self.get_trained_module()
        model.to(self.torch_device)
        model.eval()

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for x, y in generator:
                input_params = x if isinstance(x, (list, tuple)) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]
                logits = model.recommendation_score(*input_params).cpu()
                all_logits.append(logits)

                target = y[0] if isinstance(y, (list, tuple)) else y
                all_targets.append(target.cpu())

        all_logits = torch.cat(all_logits, dim=0)  # [N, C] or [N]
        all_targets = torch.cat(all_targets, dim=0)  # [N]
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

    def _compute_ranking_metrics(
        self,
        df: pd.DataFrame,
        sorted_actions_list: List[List[Any]],
        ks: List[int] = [5, 10, 20],
    ) -> Dict[str, float]:
        """Compute ranking metrics (NDCG, HitRate, MRR, MAP) per user.

        For each test observation, the ground-truth item is the item_id
        in the test row. We check where it appears in the ranked list
        of candidate items produced by the model.
        """
        item_col = self.project_config.item_column.name
        ground_truth_items = df[item_col].astype(str).tolist()

        if len(ground_truth_items) != len(sorted_actions_list):
            return {}

        results = {}

        for k in ks:
            ndcg_scores = []
            hit_scores = []
            rr_scores = []
            ap_scores = []

            for gt_item, ranked_items in zip(ground_truth_items, sorted_actions_list):
                ranked_k = [str(x) for x in ranked_items[:k]]

                if gt_item in ranked_k:
                    rank_pos = ranked_k.index(gt_item) + 1  # 1-indexed
                    # NDCG: relevance is 1 for the correct item
                    ndcg_scores.append(1.0 / np.log2(rank_pos + 1))
                    hit_scores.append(1.0)
                    rr_scores.append(1.0 / rank_pos)
                    ap_scores.append(1.0 / rank_pos)
                else:
                    ndcg_scores.append(0.0)
                    hit_scores.append(0.0)
                    rr_scores.append(0.0)
                    ap_scores.append(0.0)

            results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
            results[f"hit_rate@{k}"] = float(np.mean(hit_scores))
            results[f"mrr@{k}"] = float(np.mean(rr_scores))
            results[f"map@{k}"] = float(np.mean(ap_scores))

        return results

    def after_fit(self):
        """Run the base hook, then score the test split.

        When ``test_size > 0`` it writes ``test_set_predictions.csv`` and, with
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
        ranking_metrics: Override whether to compute ranking metrics
            (NDCG, HitRate, MRR, MAP).
        sample_size_eval: Override the test-set sample size for evaluation.
        batch_size: Override the batch size.
        device: Override the device (``"cpu"``, ``"cuda"``, ``"mps"``).
        eval_split: Which splits to evaluate (Flow 1 only).
            ``"val"`` / ``"test"`` / ``"all"`` (default).
            Ignored when ``data_path`` is set.
        data_path: Path to an external CSV or parquet file for evaluation
            (Flow 2). Must contain the same columns as the training data,
            including the target column. If ``None``, uses original splits.
        output_dir: Directory to save evaluation artifacts (plots,
            predictions CSV). Defaults to the task directory.

    Returns:
        A dict with ``"val_metrics"``, ``"ranking_metrics"``, and
        ``"roc_auc_metrics"`` keys (present depending on what was computed).
    """
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

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

    # Opt-in candidate prefilter for ranking metric paths. Default False keeps
    # post-train + re-eval byte-identical to the legacy behaviour; passing
    # True consults ``project_config.candidate_provider_factory`` from inside
    # ``_get_candidate_items``.
    task._enable_prefilter = bool(enable_prefilter)

    # Set output directory for eval artifacts
    effective_output_dir = output_dir or task_dir
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
        task._external_raw_df = df_raw
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
        results["ranking_metrics"] = getattr(task, "_ranking_metrics", {})
        results["roc_auc_metrics"] = getattr(task, "_roc_auc_metrics", {})

    # --- Flow 1: original splits ---
    else:
        if eval_split in ("val", "all"):
            banner("Re-evaluating on VALIDATION set", logger=logger)
            task.evaluate()
            results["val_metrics"] = getattr(task, "_eval_results", {})

        if eval_split in ("test", "all"):
            banner("Re-evaluating on TEST set", logger=logger)
            task._save_test_set_predictions()
            results["ranking_metrics"] = getattr(task, "_ranking_metrics", {})
            results["roc_auc_metrics"] = getattr(task, "_roc_auc_metrics", {})

    # --- Save results ---
    eval_results_path = get_eval_results_path(effective_output_dir)
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Eval results saved → %s", eval_results_path)

    # Also update metadata.json in the task dir (if saving there)
    if effective_output_dir == task_dir:
        metadata_path = get_metadata_path(task_dir)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if "val_metrics" in results:
            metadata["evaluation"] = results["val_metrics"]
        if results.get("ranking_metrics"):
            metadata["ranking_metrics"] = results["ranking_metrics"]
        if results.get("roc_auc_metrics"):
            metadata["roc_auc_metrics"] = results["roc_auc_metrics"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata updated → %s", metadata_path)

    task.cache_cleanup()

    return results
