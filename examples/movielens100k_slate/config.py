"""
Project configuration for MovieLens 100K slate-ranking.

Two framework features in one project:
- ``available_items_column_name`` + ``InteractionsWithNegativeItemGenerationByAvailableItemsDataset``:
    negatives are drawn from the per-row candidate slate, not the whole
    catalogue. This matches the "leave-one-out + 99 random negatives"
    benchmark format used in many recsys papers.
- ``metadata_columns``: each item carries a 19-dim binary genre vector,
    looked up by item index at sample time.
"""

from mars_rec.data.dataset import (
    InteractionsWithNegativeItemGenerationByAvailableItemsDataset,
)
from mars_rec.meta_config import (
    ProjectConfig,
    Column,
    IOType,
    RecommenderType,
)
from examples.movielens100k_slate.data import (
    BASE_DIR,
    PrepareML100KSlateDataFrame,
    GENRE_DIM,
)

movielens100k_slate = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareML100KSlateDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationByAvailableItemsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    metadata_columns=[
        Column("item_genre_vec", IOType.FLOAT_ARRAY),
    ],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name="available_items",
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

MOVIELENS100K_GENRE_DIM = GENRE_DIM
