"""
Project configuration for Last.fm sequential next-item recommendation.
"""

from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import (
    ProjectConfig,
    Column,
    IOType,
    RecommenderType,
)
from examples.lastfm.data import BASE_DIR, PrepareLastFMDataFrame, HISTORY_LEN

# history_artist_ids shares the vocabulary of artist_id (the item column) via
# ``same_index_as`` — the framework reuses the same int-id mapping for both,
# which is what lets the model embed both with one shared item embedding.
lastfm = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareLastFMDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("artist_id", IOType.INDEXABLE),
    other_input_columns=[
        Column(
            "history_artist_ids",
            IOType.INDEXABLE_ARRAY,
            same_index_as="artist_id",
        ),
    ],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

# Re-export so the model can read the configured history length without
# importing the data module directly.
LASTFM_HISTORY_LEN = HISTORY_LEN
