"""
Project configuration for UCI Online Retail session-based recommendation.

The ``user_column`` is bound to ``session_id`` (the invoice number) — there
is no stable customer identifier per row. Demonstrates that ``user_column``
in mars-rec can be any indexable entity, not just a literal user.
"""

from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import (
    ProjectConfig,
    Column,
    IOType,
    RecommenderType,
)
from examples.online_retail.data import (
    BASE_DIR,
    PrepareOnlineRetailDataFrame,
    HISTORY_LEN,
)

online_retail = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareOnlineRetailDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("session_id", IOType.INDEXABLE),
    item_column=Column("stock_code", IOType.INDEXABLE),
    other_input_columns=[
        Column(
            "history_stock_codes",
            IOType.INDEXABLE_ARRAY,
            same_index_as="stock_code",
        ),
    ],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

ONLINE_RETAIL_HISTORY_LEN = HISTORY_LEN
