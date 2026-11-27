# Outputs & caching

## Artifact layout

Every training run writes to `$OUTPUT_PATH/models/SupervisedModelTraining/results/<task_id>/`
(`OUTPUT_PATH` defaults to `output`):

```text
<task_id>/
├── params.json              # Every parameter of the run
├── weights.ckpt             # PyTorch Lightning checkpoint
├── history.csv              # Per-epoch metrics
├── metadata.json            # Ranking + ROC-AUC metrics (when computed)
├── index_mapping.pkl        # Categorical → index mapping (needed for inference)
├── test_set_predictions.csv # Test-set scores and ranking
├── architecture.svg         # Model architecture diagram (traced shapes)
├── summary.txt              # Layer / parameter summary
├── plot_history.jpg         # Training curves
├── plot_scores.png          # Prediction score distribution
├── sample_train.csv         # A sample of the training data
└── lightning_logs/
    └── metrics.csv          # Raw per-step metrics
```

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
