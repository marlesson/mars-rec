"""
Project configuration for Amazon Beauty content-based recommendation.

Demonstrates ``metadata_columns``: the ``item_text_emb`` vector lives in a
separate per-item metadata CSV and is attached to each interaction at
sample time via the item index. This is the right pattern any time you
have a dense per-item feature (text embedding, image embedding, …) that
would otherwise blow up the interaction CSV.
"""

from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import (
    ProjectConfig,
    Column,
    IOType,
    RecommenderType,
)
from examples.amazon_beauty.data import (
    BASE_DIR,
    PrepareAmazonBeautyDataFrame,
    TEXT_EMB_DIM,
)

amazon_beauty = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareAmazonBeautyDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    metadata_columns=[
        Column("item_text_emb", IOType.FLOAT_ARRAY),
    ],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.CONTENT_BASED,
)

AMAZON_BEAUTY_TEXT_EMB_DIM = TEXT_EMB_DIM
