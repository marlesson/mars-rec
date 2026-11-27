# Training parameters

Every parameter below is a Luigi parameter of
[`SupervisedModelTraining`][mars_rec.simulation.training.SupervisedModelTraining]. You can set it
as a keyword argument in Python, or as a kebab-case flag on the CLI (`learning_rate` →
`--learning-rate`). Dict and list values are passed as JSON:

```bash
mars-rec run supervised \
  --project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --recommender-extra-params '{"n_factors": 32}' \
  --metrics '["loss", "binary_accuracy"]' \
  --learning-rate 0.0005
```

!!! info "Parameters define the task id"
    The run's `task_id` is a hash of these values. Changing any of them, `observation` included,
    produces a new run directory. See [Outputs & caching](../concepts/outputs.md).

## Project and model

| Parameter | Default | Description |
|---|---|---|
| `project` | *required* | Dotted path to a `ProjectConfig` instance (e.g. `examples.movielens.config.movielens`) |
| `recommender_module_class` | *required* | Dotted path to the model **class** |
| `recommender_extra_params` | `{}` | Keyword arguments for the model constructor |
| `observation` | `""` | Free-form note. It is part of the hash, so changing it forces a new run |
| `load_index_mapping_path` | `None` | Task directory of a previous run whose `index_mapping.pkl` should be reused |
| `seed` | `42` | Random seed (also sets `PYTHONHASHSEED`) |

## Optimisation

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `100` | Maximum number of epochs |
| `batch_size` | `500` | Batch size |
| `learning_rate` | `0.001` | Learning rate |
| `optimizer` | `adam` | `adam`, `adamw`, `sgd`, `rmsprop`, `radam`, `adadelta`, `adagrad`, `adamax` |
| `optimizer_params` | `{}` | Extra kwargs for the optimizer |
| `lr_scheduler` | `""` | `""` (none), `reduce_on_plateau`, `cosine_annealing`, `cosine_warm_restarts`, `step` |
| `lr_scheduler_params` | `{}` | Extra kwargs for the scheduler. For `reduce_on_plateau`, `monitor` / `interval` / `frequency` are consumed by Lightning |
| `loss_function` | `bce` | See [loss functions](#loss-functions) |
| `loss_function_params` | `{}` | Extra kwargs for the loss |
| `gradient_norm_clipping` | `0.0` | Max gradient norm (`0` disables clipping) |
| `gradient_norm_clipping_type` | `2` | Norm type |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping |
| `early_stopping_min_delta` | `0.001` | Minimum change that counts as an improvement |
| `monitor_metric` | `val_loss` | Metric monitored for early stopping and checkpointing |
| `monitor_mode` | `min` | `min` or `max` |
| `precision` | `""` | Lightning precision (`32-true`, `16-mixed`, `bf16-mixed`). Empty means `16-mixed` on CUDA and `32-true` elsewhere |
| `train_sample_weights_provider` | `""` | `"module.path:func"` that returns per-row sampling weights for the train loader |

### Loss functions

| Name | Class |
|---|---|
| `bce` / `bce_logits` / `bce_logits_smooth` | `BCELoss` / `BCEWithLogitsLoss` / label-smoothed BCE-with-logits |
| `mse`, `nll`, `ce`, `mlm` | `MSELoss`, `NLLLoss`, `CrossEntropyLoss`, `MultiLabelMarginLoss` |
| `focal_loss`, `implicit_feedback_bce` | `FocalLoss`, `ImplicitFeedbackBCELoss` |
| `bpr`, `ranknet`, `lambdarank` | Pairwise / listwise ranking losses |
| `infonce`, `hardneg_infonce`, `cosine_similarity` | Embedding losses (require `EmbeddingRecommenderLightningModule`) |
| `dummy` | `DummyLoss` (for models that compute their own loss) |

For the `forward()` output shape each loss expects, see [Models](../concepts/models.md#forward-output-shape).

## Data split and sampling

| Parameter | Default | Description |
|---|---|---|
| `test_size` | `0.2` | Fraction held out as test |
| `test_split_type` | `random` | `random` or `time` |
| `val_size` | `0.2` | Fraction of the remaining rows used for validation |
| `dataset_split_method` | `time` | `time`, `holdout`, `column`, `k_fold` (see [Pipeline](../concepts/pipeline.md#2-data-preparation)) |
| `n_splits` / `split_index` | `5` / `0` | K-fold configuration |
| `negative_proportion` | `0.0` | Negatives sampled per positive by the dataset class |
| `sampling_strategy` | `none` | `none`, `oversample`, `undersample` |
| `balance_fields` | `[]` | Columns to balance on (defaults to `ProjectConfig.default_balance_fields`) |
| `sampling_proportions` | `{}` | Per-class target proportions |
| `use_sampling_in_validation` | `False` | Also balance the validation split |
| `eq_filters` / `neq_filters` / `isin_filters` | `{}` | Row filters applied before the split |
| `sample_size` | `-1` | Subsample the raw data (`-1` keeps all) |
| `minimum_interactions` | `5` | Minimum interactions per entity (used by data-prep tasks that support it) |
| `data_frames_preparation_extra_params` | `{}` | Extra kwargs for the project's data-prep task (e.g. `column_stratification`) |

## Evaluation

| Parameter | Default | Description |
|---|---|---|
| `metrics` | `["loss"]` | Metrics logged during training (see [`METRIC_REGISTRY`](extending.md#adding-a-metric)) |
| `ranking_metrics` | `False` | Compute NDCG / HitRate / MRR / MAP @{5, 10, 20} and ROC-AUC on the test split |
| `sample_size_eval` | `None` | Cap the number of test rows used for evaluation |

## Hardware and data loading

| Parameter | Default | Description |
|---|---|---|
| `device` | auto | `cpu`, `cuda` or `mps`. Auto-detected in that order of preference: CUDA, MPS, CPU |
| `generator_workers` | `0` | DataLoader `num_workers` |
| `pin_memory` | `False` | DataLoader `pin_memory` |

For the authoritative and complete list, see the
[`mars_rec.simulation.training` API reference](../reference/api/training.md).
