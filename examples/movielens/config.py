"""
Project configuration for MovieLens 1M recommendation/ranking.
"""
from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import (
    ProjectConfig, Column, IOType, RecommenderType
)
from examples.movielens.data import BASE_DIR, PrepareMovieLensDataFrame, TOP_GENRES, genre_col_name

# Genre columns as NUMBER features
_genre_columns = [
    Column(genre_col_name(g), IOType.NUMBER)
    for g in TOP_GENRES
]

movielens = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareMovieLensDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("movie_id", IOType.INDEXABLE),
    other_input_columns=[
        Column("gender_encoded", IOType.NUMBER),
        Column("age_encoded", IOType.NUMBER),
        Column("occupation_norm", IOType.NUMBER),
    ] + _genre_columns,
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
