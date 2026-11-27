# Inference & evaluation

After training, every command below works from the artifacts in the run directory. Identify a run
with either option:

- `--task-id <TASK_ID>` resolves to `$OUTPUT_PATH/models/SupervisedModelTraining/results/<TASK_ID>`;
- `--model-dir <PATH>` takes the full path to a task directory (for example, one downloaded from Vertex AI).

## Listing and comparing runs

```bash
mars-rec list                                  # every run, newest first
mars-rec list --project movielens --limit 10

mars-rec results --project lastfm --sort-by val_loss
mars-rec results --metric val_loss --metric val_binary_accuracy
mars-rec results --params dropout --params n_factors    # also searches recommender_extra_params
```

```text
               date  project              model              epochs  lr     batch_size  loss_fn     val_binary_accuracy  val_loss
1  2026-05-16 13:33  movielens100k_slate  ML100KSlateRanker  8       0.001  256         bce_logits  0.799278             0.406891
2  2026-05-16 12:55  lastfm               LastFMSASRec       5       0.001  256         bce_logits  0.582535             0.675994
```

## Batch prediction: `mars-rec predict`

[`run_inference`][mars_rec.inference.run_inference] loads `params.json`, `model/weights.ckpt` and
`model/index_mapping.pkl`. It preprocesses the input **the same way training does** and writes the top-k
items per user:

```bash
mars-rec predict \
  --task-id SupervisedModelTraining____64____a0cdf4af5e \
  --data data/test.parquet \
  --output result.json \
  --top-k 10 --batch-size 128 --device cpu
```

```json
{
  "top10s": [[101, 57, 9], ["…"]],
  "user":   [1, 2],
  "time":   12.34
}
```

`--data` accepts `.parquet` or `.csv`. Keep in mind:

- the key is always `top10s`, whatever the value of `--top-k`;
- the result never contains [reserved index slots](../concepts/index-mapping.md#reserved-slots):
  unknown / `NaN` / pad logits are masked before the top-k is taken, so every user gets `k` real items
  (when the catalogue has at least `k` items).

## Re-evaluation: `mars-rec evaluate`

[`run_evaluate`][mars_rec.simulation.training.run_evaluate] rebuilds the training task from
`params.json` and runs the same `evaluate()` method as the original training. It has two flows:

=== "Original splits"

    ```bash
    mars-rec evaluate \
      --task-id <TASK_ID> \
      --ranking-metrics \
      --metric loss --metric f1_score --metric binary_accuracy \
      --split test            # val | test | all (default)
    ```

=== "External dataset"

    ```bash
    mars-rec evaluate \
      --task-id <TASK_ID> \
      --data new_test.parquet \
      --ranking-metrics
    ```

    The file must contain the target column. It is preprocessed with the training pipeline.

Each evaluation writes `metrics.json`, `predictions.csv` and `plots/` to its own
`<task_id>/eval/<name>/` directory, so evaluations never overwrite one another:

| Flow | Default `<name>` | Refreshes `metadata.json`? |
|---|---|---|
| Original splits | `test` | Yes — `list` and `results` pick the new metrics up |
| External dataset | file name of `--data` without extension (`new_test`) | No |

Pass `--name <name>` to choose the directory yourself (`test` is reserved for the original splits),
or `--output <dir>` to write the same files somewhere else. Metrics include ranking metrics
(NDCG, HitRate, MRR, MAP @{5, 10, 20}) and ROC-AUC; ROC-AUC is omitted when the evaluated split has
a single class, since it is undefined there.

`--prefilter` applies the project's
[`CandidateProvider`][mars_rec.filtering.CandidateProvider] when it builds the candidate pool for
ranking metrics.

## Architecture diagram: `mars-rec visualize`

Every training run already writes `model/architecture.svg`. To regenerate or inspect it:

```bash
mars-rec visualize --task-id <TASK_ID>                  # SVG, traced tensor shapes
mars-rec visualize --task-id <TASK_ID> --no-trace       # static introspection only
mars-rec visualize --task-id <TASK_ID> --output /tmp/arch.svg
mars-rec visualize --task-id <TASK_ID> --mode summary   # text table
```

Every option of every command is listed in the [CLI reference](../reference/cli.md).
