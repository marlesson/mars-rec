"""
Project configuration for MovieLens 1M recommendation/ranking.

User demographics are per-row ``NUMBER`` inputs; movie genres are a per-item
``FLOAT_ARRAY`` declared in ``metadata_columns``, so negative sampling
re-looks-up the genres of the sampled movie (see ``data.py``).
"""

from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import ProjectConfig, Column, IOType, RecommenderType
from examples.movielens.data import BASE_DIR, GENRE_DIM, PrepareMovieLensDataFrame

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
    ],
    metadata_columns=[
        Column("movie_genre_vec", IOType.FLOAT_ARRAY),
    ],
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

MOVIELENS_GENRE_DIM = GENRE_DIM
