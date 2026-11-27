# ProjectConfig & columns

[`ProjectConfig`][mars_rec.meta_config.ProjectConfig] is the **central contract** of a project.
It declares which columns exist, how each one is typed and indexed, which Luigi task prepares the
data and which PyTorch `Dataset` wraps it at training time. When you start a new project, write the
`ProjectConfig` first.

```python
from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig, RecommenderType
from examples.movielens.data import BASE_DIR, PrepareMovieLensDataFrame

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
    output_column=Column("label", IOType.NUMBER),
    available_items_column_name=None,
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
```

The training task receives the instance by its import path: `--project examples.movielens.config.movielens`.

## Main fields

| Field | Description |
|---|---|
| `base_dir` | Root directory for the project's data artifacts |
| `prepare_data_frames_task` | A [`BasePrepareDataFrames`][mars_rec.data.task.BasePrepareDataFrames] subclass that writes train/val/test CSVs |
| `dataset_class` | The PyTorch `Dataset` used at training time (see [below](#dataset-classes)) |
| `user_column` | `INDEXABLE` column for the *querying entity*. It can be a session (see `examples/online_retail`) |
| `item_column` | `INDEXABLE` column for the candidate / positive item |
| `other_input_columns` | Extra features passed to `forward()` after user and item, **in declaration order** |
| `output_column` | The label (usually `NUMBER`, 0/1 implicit feedback) |
| `metadata_columns` | Per-item features stored once in a metadata CSV and looked up by item index (e.g. text embeddings) |
| `available_items_column_name` | Per-row candidate slate column; set to `None` to disable slate sampling |
| `recommender_type` | Informational tag ([`RecommenderType`][mars_rec.meta_config.RecommenderType]) |
| `candidate_provider_factory` | Optional dynamic candidate pool for eval / inference (see [`mars_rec.filtering`](../reference/api/filtering.md)) |

For every field, including the less common ones, see the
[API reference](../reference/api/meta_config.md).

## `Column` and `IOType`

A [`Column`][mars_rec.meta_config.Column] is a name plus an [`IOType`][mars_rec.meta_config.IOType]:

| `IOType` | Materialised as | Typical use |
|---|---|---|
| `INDEXABLE` | `int64` index | user id, item id, any categorical value. Feeds an `nn.Embedding` |
| `NUMBER` | `float32` | numeric features, labels |
| `INDEXABLE_ARRAY` | list of `int64` indices | history of item ids (sequence models) |
| `FLOAT_ARRAY` | `float32` vector | precomputed text / image embeddings |
| `INT_ARRAY` | `int64` vector | fixed-size integer features |
| `SEQUENCE` | `int64` vector | alias of `INT_ARRAY` reserved for sequential inputs |

### Sharing a vocabulary: `same_index_as`

A history column must use the **same integer ids** as the item column so both can share one
embedding table:

```python
Column("history_artist_ids", IOType.INDEXABLE_ARRAY, same_index_as="artist_id")
```

## Dataset classes

`dataset_class` controls how training samples (and negatives) are produced. The bundled classes
live in [`mars_rec.data.dataset`](../reference/api/data.md):

| Class | Negatives |
|---|---|
| `InteractionsDataset` | None. Rows are used as-is (explicit labels) |
| `InteractionsWithNegativeItemGenerationDataset` | Uniformly sampled from the full catalogue |
| `InteractionsWithNegativeItemGenerationByAvailableItemsDataset` | Sampled from the per-row slate (`available_items_column_name`) |
| `InteractionsWithPopularityBasedNegativeSamplingDataset` | Sampled proportionally to item popularity |
| `InteractionsWithInBatchNegativesDataset` | Positives of other rows in the same batch |
| `InteractionsWithTimeAwareNegativeSamplingDataset` | Only items that already existed at interaction time |
| `InteractionsWithCurriculumNegativeSamplingDataset` | Progressively harder negatives as training advances |
| `InteractionsWithHardNegativeMiningDataset` | Pre-computed hard negatives stored in a column (random fallback) |
| `InteractionsWithEmbeddingHardNegativeDataset` | Pre-computed top-K nearest-neighbour table |
| `InteractionsWithCandidateProviderNegativeDataset` | The project's `CandidateProvider` eligible pool |
| `InteractionsWithDynamicNegativeMiningDataset` | Dynamic negative sampling (DNS): highest-scoring of K candidates under the current model |
| `InteractionsWithDynamicNegativeMiningCandidateProviderDataset` | DNS over candidates drawn from a `CandidateProvider` pool |
| `InteractionsWithMixedDnsUniformDataset` | Per-slot mix of uniform and DNS negatives |
| `InteractionsWithMixedHardNegativeDataset` | Mixture of three negative sources with per-class proportions |

Class-specific constructor arguments (for example `hard_negative_column` or `uniform_fraction`)
go in `ProjectConfig.dataset_extra_params`. The DNS variants need the current model's scores, so
the training task attaches `DynamicNegativeMiningCallback` for them automatically. For each class's
options, see the [`mars_rec.data.dataset` reference](../reference/api/data.md#mars_rec.data.dataset).

The amount of negative sampling is set at training time with `--negative-proportion`.
