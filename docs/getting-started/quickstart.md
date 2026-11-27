# Quickstart

This page trains the **MovieLens 100K slate** example, the smallest and fastest of the bundled
projects, then inspects, compares and reuses the trained model.

## 1. Train

=== "Script"

    Each example ships a `run_training.py` with sensible defaults:

    ```bash
    PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.movielens100k_slate.config.movielens100k_slate \
      --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
      --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
      --epochs 8 \
      --batch-size 256 \
      --loss-function bce_logits \
      --device cpu
    ```

=== "Python"

    ```python
    import luigi
    from mars_rec.simulation.training import SupervisedModelTraining

    task = SupervisedModelTraining(
        project="examples.movielens100k_slate.config.movielens100k_slate",
        recommender_module_class="examples.movielens100k_slate.model.ML100KSlateRanker",
        recommender_extra_params={"n_factors": 32, "genre_dim": 19, "hidden_size": 64},
        epochs=8,
        batch_size=256,
        loss_function="bce_logits",
        device="cpu",
        metrics=["loss", "binary_accuracy"],
    )

    luigi.build([task], local_scheduler=True, log_level="INFO")
    ```

On the first run the dataset is downloaded and cached under `output/<example>/dataset/`.
The training artifacts are written to
`output/models/SupervisedModelTraining/results/<task_id>/`. For the full layout, see
[Outputs & caching](../concepts/outputs.md).

!!! tip "Re-running the same command does nothing"
    Luigi treats a task as *done* when its output directory exists. Pass `--no-cache` to
    force a fresh run, or change any parameter (for example `--observation "try-2"`).

## 2. Inspect and compare runs

```bash
mars-rec list
mars-rec results --project movielens100k_slate --sort-by val_loss
```

```text
#    Date               Project     Model               Epochs   Task ID
----------------------------------------------------------------------------------
1    2026-05-16 13:33   ml100k_slt  ML100KSlateRanker   8        SupervisedModelTraining____256____1ab7b0e50e
```

## 3. Predict and re-evaluate

```bash
# Top-10 items per user → result.json
mars-rec predict --task-id <TASK_ID> --data data/test.parquet --top-k 10

# Ranking metrics (NDCG / HitRate / MRR / MAP) on the original test split
mars-rec evaluate --task-id <TASK_ID> --ranking-metrics --split test

# Architecture diagram (also generated automatically after training)
mars-rec visualize --task-id <TASK_ID>
```

## 4. Try the other examples

```bash
PYTHONPATH=src:. python examples/movielens/run_training.py      # MovieLens 1M: NCF
PYTHONPATH=src:. python examples/lastfm/run_training.py         # Last.fm: SASRec-lite
PYTHONPATH=src:. python examples/online_retail/run_training.py  # Online Retail: GRU4Rec
PYTHONPATH=src:. python examples/amazon_beauty/run_training.py  # Amazon Beauty: two-tower
```

For a walkthrough of each example, see [Tutorials](../tutorials/index.md).
