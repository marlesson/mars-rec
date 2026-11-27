# Outputs & caching

## Artifact layout

Every training run writes to `$OUTPUT_PATH/models/SupervisedModelTraining/results/<task_id>/`
(`OUTPUT_PATH` defaults to `output`):

```text
<task_id>/
├── params.json                  # Every parameter of the run
├── metadata.json                # Run summary + ranking / ROC-AUC metrics (when computed)
├── weights.ckpt                 # PyTorch Lightning checkpoint (best epoch)
├── index_mapping.pkl            # Categorical → index mapping (needed for inference)
├── architecture.svg             # Model architecture diagram (traced shapes)
├── summary.txt                  # Layer / parameter summary
├── history.csv                  # Per-epoch metrics
├── plot_history.jpg             # Training curves
├── lightning_logs/
│   └── metrics.csv              # Raw per-step metrics
├── sample_train.csv             # 100 training rows (after indexing)
├── sample_train_keys.csv        # Raw key columns of those rows (when the project declares them)
├── sample_train_scores.csv      # Model logits for those rows
├── test_set_predictions.csv     # Test-set scores and ranking
├── plot_scores.png              # Prediction score distribution
├── plot_confusion_matrix.png    # Binary targets only
├── plot_roc_curve.png           # Binary targets only
├── plot_precision_recall_curve.png
└── eval_results.json            # Written by `mars-rec evaluate`
```

Every path is built by a helper in [`mars_rec.utils.files`](../reference/api/utils.md), so code that
needs an artifact should call those helpers (`get_params_path`, `find_checkpoint`, …) rather than
joining file names onto the task directory.

TensorBoard logs (with `mars-rec[tensorboard]`) go to `$OUTPUT_PATH/tensorboard_logs/<task_id>/`,
outside the task directory, so a single `tensorboard --logdir $OUTPUT_PATH/tensorboard_logs`
compares every run.

`mars-rec list` and `mars-rec results` read `params.json`, `history.csv` and `metadata.json` from
these directories.

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
