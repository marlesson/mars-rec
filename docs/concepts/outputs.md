# Outputs & caching

## Artifact layout

Every training run writes to `$OUTPUT_PATH/models/SupervisedModelTraining/results/<task_id>/`
(`OUTPUT_PATH` defaults to `output`):

```text
<task_id>/
├── params.json                      # Every parameter of the run
├── metadata.json                    # Run summary: environment, history, test metrics, artifact list
├── model/                           # Everything inference needs — copy this folder to serve the model
│   ├── weights.ckpt                 # PyTorch Lightning checkpoint (best epoch)
│   ├── index_mapping.pkl            # Categorical → index mapping
│   ├── architecture.svg             # Model architecture diagram (traced shapes)
│   └── summary.txt                  # Layer / parameter summary
├── training/                        # How the model was trained
│   ├── history.csv                  # Per-epoch metrics
│   ├── plot_history.jpg             # Training curves
│   ├── lightning_logs/metrics.csv   # Raw per-step metrics
│   └── samples/
│       ├── sample_train.csv         # 100 training rows (after indexing)
│       ├── sample_train_keys.csv    # Raw key columns of those rows (when the project declares them)
│       └── sample_train_scores.csv  # Model logits for those rows
└── eval/                            # One sub-directory per evaluation
    ├── test/                        # Post-training evaluation on the original test split
    │   ├── metrics.json             # Point-wise, ranking and ROC-AUC metrics
    │   ├── predictions.csv          # Per-row scores and ranking
    │   └── plots/
    │       ├── scores.png           # Prediction score distribution
    │       ├── confusion_matrix.png # Binary targets only
    │       ├── roc_curve.png        # Binary targets only
    │       └── precision_recall_curve.png
    └── <name>/                      # `mars-rec evaluate --data <file> [--name <name>]`, same layout
```

`mars-rec list` and `mars-rec results` read `params.json`, `training/history.csv` and
`metadata.json`. `metadata.json` summarises the canonical `eval/test` evaluation only: evaluating
external data writes to its own `eval/<name>/` directory and never overwrites it (see
[Re-evaluation](../guides/inference-evaluation.md#re-evaluation-mars-rec-evaluate)).

Every path is built by a helper in [`mars_rec.utils.files`](../reference/api/utils.md), so code that
needs an artifact should call those helpers (`get_history_path`, `find_checkpoint`,
`get_eval_dir`, …) rather than joining file names onto the task directory.

TensorBoard logs (with `mars-rec[tensorboard]`) go to `$OUTPUT_PATH/tensorboard_logs/<task_id>/`,
outside the task directory, so a single `tensorboard --logdir $OUTPUT_PATH/tensorboard_logs`
compares every run.

### Runs from older versions

Runs trained before this layout keep every file flat in `<task_id>/` (`weights.ckpt`,
`history.csv`, `test_set_predictions.csv`, …). `predict`, `evaluate`, `visualize`, `list` and
`results` still read them: each helper falls back to the flat file when the new one does not exist.
Re-evaluating an old run writes its new results to `eval/test/`.

## Task id

The `<task_id>` (e.g. `SupervisedModelTraining____256____1ab7b0e50e`) is derived from a **hash of
the task parameters**. Two runs with identical parameters share the same id and directory.

## Caching and `--no-cache`

Luigi considers a task **done** when its output exists. Re-running an identical command is
therefore a no-op, which also applies to a run that crashed after creating its output directory.

To force a re-run, use one of these:

- `mars-rec run supervised --no-cache ...` invalidates the data-prep and training outputs
  (`task.invalidate_cache()`) before scheduling;
- change any parameter, for example `--observation "retry"` (free text that becomes part of the hash);
- delete the task directory manually.
