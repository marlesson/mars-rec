"""Abstract base class for every model in mars-rec.

Every model class referenced via the
``recommender_module_class`` training parameter must inherit from
:class:`RecommenderModule`. The framework instantiates the subclass
with ``project_config`` and ``index_mapping``, plus any keyword
arguments passed through ``recommender_extra_params``.
"""

import abc
from typing import Dict, Any

import torch.nn as nn

from mars_rec.meta_config import ProjectConfig


class RecommenderModule(nn.Module, metaclass=abc.ABCMeta):
    """Base class shared by every recommender model in the framework.

    Subclasses inherit two conveniences computed from the
    ``index_mapping`` at construction time:

    - ``self._n_users`` — size of the user vocabulary (use as the
      ``num_embeddings`` of ``nn.Embedding``).
    - ``self._n_items`` — size of the item vocabulary.

    Both include the 3 reserved slots (unknown, NaN, pad) the indexing
    machinery prepends, so an embedding sized with these values is
    always safe regardless of how the data was prepared.

    The training loop calls the subclass's ``forward(*args)`` with one
    positional argument per entry in
    :attr:`ProjectConfig.input_columns`, in declaration order.

    :param project_config: The :class:`ProjectConfig` of the project
        being trained. Stored on ``self._project_config`` for
        downstream introspection (e.g. ``forward`` can branch on
        column types).
    :param index_mapping: ``{column_name: {raw_value: int_index}}``
        produced by ``utils.index_mapping.create_index_mapping``. Kept
        on ``self._index_mapping`` so subclasses can recover the
        original values during debugging.
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
    ) -> None:
        super().__init__()
        self._index_mapping = index_mapping
        self._project_config = project_config
        self._n_users = max(index_mapping[project_config.user_column.name].values()) + 1
        self._n_items = max(index_mapping[project_config.item_column.name].values()) + 1

    def recommendation_score(self, *args):
        """Score interface for inference / eval — defaults to ``forward``.

        Override when the *training* forward returns auxiliary outputs
        (multi-task models) but ranking metrics should be computed
        against a single scalar score.
        """
        return self.forward(*args)
