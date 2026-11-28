"""
Project configuration for the synthetic offline example.

The minimal ``ProjectConfig``: a user column, an item column and a label.
Negatives are drawn uniformly from the catalogue at training time.
"""

from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig, RecommenderType
from examples.synthetic.data import BASE_DIR, PrepareSyntheticDataFrame

synthetic = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareSyntheticDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
