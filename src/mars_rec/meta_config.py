"""Central configuration types for a mars-rec project.

Every project (e.g. a directory under ``examples/``)
declares a :class:`ProjectConfig` instance describing **which columns
exist, what type they have, which Luigi task prepares the data, and
which PyTorch ``Dataset`` class wraps it**. The training entry point
(:class:`mars_rec.simulation.training.SupervisedModelTraining`)
takes a dotted path to that instance via its ``project`` parameter
and drives the rest of the pipeline from it.

The two supporting types are :class:`Column` (a typed column
descriptor) and :class:`IOType` / :class:`RecommenderType` (enums for
the column dtype and the high-level recommender flavour).

See ``examples/movielens/config.py`` for the canonical minimal example.
"""

from enum import Enum, auto
from typing import Callable, List, Type, Dict, Optional, TYPE_CHECKING

from torch.utils.data import Dataset
import numpy as np

from mars_rec.data.task import BasePrepareDataFrames

if TYPE_CHECKING:
    from mars_rec.filtering import CandidateProvider


class IOType(Enum):
    """Declared dtype for a :class:`Column`.

    Members:

    - ``INDEXABLE``        — categorical scalar mapped to ``int64`` via
      ``utils.index_mapping`` (user_id, item_id, …).
    - ``NUMBER``           — numeric scalar stored as ``float32``.
    - ``INDEXABLE_ARRAY``  — list of categorical ids mapped through the
      column's vocabulary (each row holds a sequence of indices).
    - ``FLOAT_ARRAY``      — fixed-shape numeric vector stored as ``float32``
      (typical for precomputed text / image embeddings).
    - ``INT_ARRAY``        — fixed-shape integer vector (``int64``).
    - ``SEQUENCE``         — alias of ``INT_ARRAY`` reserved for sequential
      model inputs (kept distinct so downstream consumers can branch
      on intent rather than dtype).

    The :attr:`dtype` property returns the numpy dtype the framework
    will materialise the column with in :class:`InteractionsDataset`.
    """

    INDEXABLE = auto()
    NUMBER = auto()
    INDEXABLE_ARRAY = auto()
    FLOAT_ARRAY = auto()
    INT_ARRAY = auto()
    SEQUENCE = auto()

    @property
    def dtype(self):
        """Numpy dtype the framework casts the column to at load time."""
        return {
            self.INDEXABLE.name: np.int64,
            self.NUMBER.name: np.float32,
            self.FLOAT_ARRAY.name: np.float32,
            self.INT_ARRAY.name: np.int64,
            self.SEQUENCE.name: np.int64,
        }[self.name]


class RecommenderType(Enum):
    """High-level flavour tag for a recommender project.

    Purely informational — the framework does not branch on it today, but
    it is persisted alongside the project config so downstream tooling
    (visualisation, leaderboards, etc.) can categorise runs without
    inspecting the model code.
    """

    USER_BASED_COLLABORATIVE_FILTERING = auto()
    ITEM_BASED_COLLABORATIVE_FILTERING = auto()
    CONTENT_BASED = auto()


class Column(object):
    """Declarative description of a single column in the project schema.

    A ``Column`` carries everything the framework needs to load, index
    and feed the value through the model:

    :param name: Column name in the source DataFrame.
    :param type: :class:`IOType` declaring the dtype and how the column
        is materialised at sample time.
    :param same_index_as: When given, makes this column reuse the
        vocabulary (and the resulting integer ids) of another column.
        Typical use: declare a history-of-items column as
        ``Column("history_ids", IOType.INDEXABLE_ARRAY, same_index_as="item_id")``
        so the history shares the item embedding space.
    """

    def __init__(self, name: str, type: IOType, same_index_as: str = None) -> None:
        self.name = name
        self.type = type
        self.same_index_as = same_index_as


class ProjectConfig(object):
    """Central configuration object describing a recommendation project.

    A :class:`ProjectConfig` declares the **data contract** (which columns
    feed the model, what they mean, how they are indexed, how negatives
    are sampled) and is the single value loaded by
    :class:`mars_rec.simulation.training.SupervisedModelTraining`
    via the ``project`` dotted-path parameter.

    The most important parameters:

    :param base_dir: Output root for this project's artifacts under
        ``output/<base_dir>``.
    :param prepare_data_frames_task: Luigi task class (subclass of
        :class:`mars_rec.data.task.BasePrepareDataFrames`) that
        materialises the train / val / test CSVs.
    :param dataset_class: PyTorch ``Dataset`` subclass that wraps the
        CSVs at training time. The framework ships several variants in
        :mod:`mars_rec.data.dataset` (uniform negative sampling,
        slate negatives, hard negatives, …).
    :param user_column: The ``IOType.INDEXABLE`` column that identifies
        the *querying entity*. Often a real user id but can be any
        indexable entity (e.g. ``session_id`` for anonymous sessions —
        see ``examples/online_retail``).
    :param item_column: The ``IOType.INDEXABLE`` column for the
        candidate/positive item.
    :param other_input_columns: Additional columns fed to ``forward()``
        in declaration order, after ``user_column`` and ``item_column``.
    :param output_column: The label column. Usually ``IOType.NUMBER``
        (0/1 implicit feedback).
    :param recommender_type: Informational tag — see
        :class:`RecommenderType`.

    Less common parameters (defaults are fine for most projects):

    :param dataset_extra_params: Forwarded to ``dataset_class.__init__``.
    :param user_is_input: When False, ``user_column`` is loaded but not
        passed to ``forward()`` (e.g. content-only models).
    :param item_is_input: Same idea for the item column.
    :param hist_view_column_name: Conventional column name used by data
        prep helpers; override only if your CSV uses a different name.
    :param hist_output_column_name: Same as ``hist_view_column_name``.
    :param timestamp_column_name: Same as ``hist_view_column_name``.
    :param available_items_column_name: Name of the per-row candidate
        slate column, consumed by
        :class:`InteractionsWithNegativeItemGenerationByAvailableItemsDataset`.
        Set to ``None`` to disable slate-based sampling.
    :param n_users_column: Name of the (optional) user-cardinality
        column persisted alongside the train CSV.
    :param n_items_column: Same for the item cardinality.
    :param default_balance_fields: Columns used for over/under-sampling
        when ``sampling_strategy != "none"``.
    :param metadata_columns: Per-item content features stored once in a
        separate metadata CSV and looked up by item index at sample
        time (e.g. precomputed text/image embeddings). See
        ``examples/amazon_beauty`` for a working two-tower example.
    :param auxiliar_output_columns: Additional targets returned by
        ``__getitem__`` (used by multi-task models).
    :param possible_negative_indices_columns: Per-column override of
        the negative-sampling pool, used by the popularity / hard-mining
        dataset variants.
    :param candidate_provider_factory: Optional factory producing a
        :class:`mars_rec.filtering.CandidateProvider`. When set,
        the training / eval / inference paths consult it to filter the
        candidate set before scoring — generalises
        ``available_items_column_name`` to dynamic providers.
    :param sample_keys_columns: Pre-indexing columns to persist
        alongside ``sample_train.csv`` so inference paths can resolve
        sampled rows back to their original keys.
    """

    def __init__(
        self,
        base_dir: str,
        prepare_data_frames_task: Type[BasePrepareDataFrames],
        dataset_class: Type[Dataset],
        user_column: Column,
        item_column: Column,
        other_input_columns: List[Column],
        output_column: Column,
        recommender_type: RecommenderType = RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
        dataset_extra_params: dict = None,
        user_is_input: bool = True,
        item_is_input: bool = True,
        hist_view_column_name: str = "hist_view",
        hist_output_column_name: str = "hist_output",
        timestamp_column_name: str = "timestamp",
        available_items_column_name: str = "available_items",
        n_users_column: str = "n_users",
        n_items_column: str = "n_items",
        default_balance_fields: List[str] = None,
        metadata_columns: List[Column] = None,
        auxiliar_output_columns: List[Column] = None,
        possible_negative_indices_columns: Dict[str, List[str]] = None,
        candidate_provider_factory: Optional[Callable[..., "CandidateProvider"]] = None,
        sample_keys_columns: Optional[List[str]] = None,
    ) -> None:
        self.base_dir = base_dir
        self.prepare_data_frames_task = prepare_data_frames_task
        self.dataset_class = dataset_class
        self.dataset_extra_params = dataset_extra_params if dataset_extra_params is not None else {}
        self.user_column = user_column
        self.item_column = item_column
        self.other_input_columns = other_input_columns
        self.output_column = output_column
        self.user_is_input = user_is_input
        self.item_is_input = item_is_input
        self.hist_view_column_name = hist_view_column_name
        self.hist_output_column_name = hist_output_column_name
        self.timestamp_column_name = timestamp_column_name
        self.available_items_column_name = available_items_column_name
        self.auxiliar_output_columns = auxiliar_output_columns if auxiliar_output_columns is not None else []
        self.recommender_type = recommender_type
        self.n_users_column = n_users_column
        self.n_items_column = n_items_column
        self.default_balance_fields = default_balance_fields if default_balance_fields is not None else []
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        self.possible_negative_indices_columns = possible_negative_indices_columns
        # Optional factory ``(project_config, index_mapping, dataset_extras) ->
        # CandidateProvider``. When set, the inference / eval paths consult it
        # to pre-filter candidates before scoring (see mars_rec.filtering).
        # Default ``None`` preserves the legacy behaviour in every consumer.
        self.candidate_provider_factory = candidate_provider_factory
        # Optional list of columns from the RAW (pre-``transform_with_indexing``)
        # train DataFrame to persist as identity keys alongside
        # ``sample_train.csv``. Consumed by training to write
        # ``sample_train_keys.csv`` and by inference ``--val-prediction`` modes
        # to look up the original session/user/item each sampled row referred
        # to. Default ``None`` disables the artifact entirely; downstream
        # validation flags should hard-fail with a clear message when the
        # project did not opt in.
        self.sample_keys_columns = sample_keys_columns

    @property
    def input_columns(self) -> List[Column]:
        """Columns the model's ``forward()`` receives, in declaration order.

        The order is: ``user_column`` (if ``user_is_input``), ``item_column``
        (if ``item_is_input``), then ``other_input_columns``, then the
        ``metadata_columns`` (when ``item_is_input``; metadata is keyed by
        the item index). Match this order in your model's ``forward``
        signature.
        """
        input_columns: List[Column] = []
        if self.user_is_input:
            input_columns.append(self.user_column)
        if self.item_is_input:
            self._item_input_index = len(input_columns)
            input_columns.append(self.item_column)
        input_columns.extend(self.other_input_columns)
        if self.item_is_input:
            input_columns.extend(self.metadata_columns)
        return input_columns

    @property
    def all_columns(self) -> List[Column]:
        """Every column declared in the project, including outputs.

        Used by the indexing machinery to discover which columns need a
        vocabulary built. Order is intentional but not part of the public
        contract — consumers should look up by name.
        """
        return [
            self.user_column,
            self.item_column,
            *self.other_input_columns,
            *self.auxiliar_output_columns,
            *self.metadata_columns,
            self.output_column,
        ]

    def get_column_by_name(self, name: str) -> Optional[Column]:
        """Return the :class:`Column` whose ``name`` matches, or ``None``."""
        for column in self.all_columns:
            if column.name == name:
                return column
        return None
