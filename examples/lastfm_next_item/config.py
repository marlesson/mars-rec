"""
Project configuration for Last.fm next-item prediction with full-softmax CE.

Same data as ``examples/lastfm`` (the prepare task and its cached CSVs are
reused), but a different training contract:

- ``item_is_input=False``: the model never sees the target artist. It reads
  only the user and the history, and scores *every* artist at once.
- ``output_column`` is the item column itself (``INDEXABLE``), so the target
  is the artist's index in the vocabulary, which is what
  ``CrossEntropyLoss`` expects for ``[batch, n_items]`` logits.
- ``InteractionsDataset`` (no negative sampling): the softmax over the whole
  catalogue already treats every other artist as a negative.
"""

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig, RecommenderType
from examples.lastfm.data import BASE_DIR, HISTORY_LEN, PrepareLastFMDataFrame

lastfm_next_item = ProjectConfig(
    base_dir=BASE_DIR,
    prepare_data_frames_task=PrepareLastFMDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("artist_id", IOType.INDEXABLE),
    item_is_input=False,
    other_input_columns=[
        Column("history_artist_ids", IOType.INDEXABLE_ARRAY, same_index_as="artist_id"),
    ],
    output_column=Column("artist_id", IOType.INDEXABLE),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

LASTFM_HISTORY_LEN = HISTORY_LEN
