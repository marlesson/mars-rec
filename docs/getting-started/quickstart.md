# Quickstart

This page trains the **synthetic** example, which generates its data locally (no download) and
trains in a few seconds on CPU. It then inspects, compares and reuses the trained model.

## 1. Train

=== "Script"

    Each example ships a `run_training.py` with sensible defaults. This one also re-evaluates the
    saved model at the end:

    ```bash
    PYTHONPATH=src:. python examples/synthetic/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.synthetic.config.synthetic \
      --recommender-module-class examples.synthetic.model.SyntheticMF \
      --recommender-extra-params '{"n_factors": 16}' \
      --epochs 5 \
      --batch-size 256 \
      --learning-rate 0.01 \
      --loss-function bce_logits \
      --negative-proportion 0.5 \
      --ranking-metrics \
      --device cpu
    ```

=== "Python"

    ```python
    import luigi
    from mars_rec.simulation.training import SupervisedModelTraining

    task = SupervisedModelTraining(
        project="examples.synthetic.config.synthetic",
        recommender_module_class="examples.synthetic.model.SyntheticMF",
        recommender_extra_params={"n_factors": 16},
        epochs=5,
        batch_size=256,
        learning_rate=0.01,
        loss_function="bce_logits",
        negative_proportion=0.5,
        ranking_metrics=True,
        device="cpu",
        metrics=["loss", "binary_accuracy"],
    )

    luigi.build([task], local_scheduler=True, log_level="INFO")
    ```

The dataset is generated (or, for the other examples, downloaded) on the first run and cached under
`output/<example>/dataset/`.
The training artifacts are written to
`output/models/SupervisedModelTraining/results/<task_id>/`. For the full layout, see
[Outputs & caching](../concepts/outputs.md).

!!! tip "Re-running the same command does nothing"
    Luigi treats a task as *done* when its output directory exists. Pass `--no-cache` to
    force a fresh run, or change any parameter (for example `--observation "try-2"`).

## 2. Inspect and compare runs

```bash
mars-rec list
mars-rec results --project synthetic --sort-by val_loss
```

```text
#    Date               Project     Model               Epochs   Task ID
----------------------------------------------------------------------------------
1    2026-11-27 21:06   synthetic   SyntheticMF         5        SupervisedModelTraining____256____5750a3a85b
```

## 3. Re-evaluate and predict

```bash
# Ranking evaluation (full catalog, popularity baseline) on the original test split
mars-rec evaluate --task-id <TASK_ID> --ranking-metrics --split test

# Architecture diagram (also generated automatically after training)
mars-rec visualize --task-id <TASK_ID>
```

`mars-rec predict` writes the top-k items per row. It needs a model that scores the whole catalogue in
one forward pass (`[batch, n_items]` logits, trained with `ce`), so try it on the
[Last.fm next-item](../tutorials/lastfm-next-item.md) example:

```bash
mars-rec predict --task-id <TASK_ID> --data <test.csv> --top-k 10
```

## 4. Try the other examples

```bash
PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py      # MovieLens 100K: slate ranking
PYTHONPATH=src:. python examples/movielens100k_retrieval/run_training.py  # MovieLens 100K: two-tower InfoNCE
PYTHONPATH=src:. python examples/movielens/run_training.py                # MovieLens 1M: NCF
PYTHONPATH=src:. python examples/lastfm/run_training.py                   # Last.fm: SASRec-lite (BCE)
PYTHONPATH=src:. python examples/lastfm_next_item/run_training.py         # Last.fm: SASRec next-item (CE)
PYTHONPATH=src:. python examples/online_retail/run_training.py            # Online Retail: GRU4Rec
PYTHONPATH=src:. python examples/amazon_beauty/run_training.py            # Amazon Beauty: two-tower
```

For a walkthrough of each example, see [Tutorials](../tutorials/index.md).
