"""
Project configuration for MovieLens 100K two-tower retrieval with InfoNCE.

The contract for embedding losses (``infonce``, ``cosine_similarity``,
``hardneg_infonce``), which select ``EmbeddingRecommenderLightningModule``:

- ``item_is_input=False``: the query tower never sees the target movie.
- ``output_column`` is the item column (``INDEXABLE``), so each batch carries
  the target ids that the model embeds with ``get_target_emb``.
- ``InteractionsDataset`` (no negative sampling): InfoNCE uses the other
  targets of the batch as negatives.
"""

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig, RecommenderType
from examples.movielens100k_retrieval.data import BASE_DIR, HISTORY_LEN, PrepareML100KRetrievalDataFrame

movielens100k_retrieval = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareML100KRetrievalDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    item_is_input=False,
    other_input_columns=[
        Column("history_item_ids", IOType.INDEXABLE_ARRAY, same_index_as="item_id"),
    ],
    output_column=Column("item_id", IOType.INDEXABLE),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

MOVIELENS100K_RETRIEVAL_HISTORY_LEN = HISTORY_LEN
