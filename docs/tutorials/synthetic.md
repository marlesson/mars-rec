# Synthetic: offline quickstart (matrix factorisation)

**Demonstrates:** The whole lifecycle with **no download**, in a few seconds on CPU: data generation as a Luigi task, a temporal split, training, the test-set [ranking evaluation](../concepts/evaluation.md) against a most-popular baseline, and re-evaluation from the saved artifacts with `run_evaluate`.

## Dataset

Generated locally by `GenerateSyntheticInteractions`. By default it has 500 users, 200 items and 30 events per user (15K positives). Users and items get random latent vectors, and each user's events are distinct items drawn from a softmax over `user · item + popularity`. The data therefore has a planted, learnable signal on top of popularity: a working model must clearly beat the most-popular baseline. Items are not repeated per user because the evaluation removes items already seen in train/val from the candidates. The generation settings are Luigi parameters, so changing one gives a new task id instead of reusing a stale CSV.

This is also the project behind `tests/test_examples_e2e.py`, the one example the CI runs end to end.

## Model

`SyntheticMF` is matrix factorisation with biases, `<p_u, q_i> + b_u + b_i + b`. It returns one logit per row (`[batch]`) and is trained with `bce_logits` and uniform negative sampling.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/synthetic/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.synthetic.config.synthetic \
      --recommender-module-class examples.synthetic.model.SyntheticMF \
      --recommender-extra-params '{"n_factors": 16}' \
      --epochs 5 --batch-size 256 --learning-rate 0.01 --loss-function bce_logits \
      --negative-proportion 0.5 --dataset-split-method time --test-split-type time \
      --ranking-metrics --device cpu

    mars-rec evaluate --task-id <TASK_ID> --ranking-metrics --split test
    ```

The script ends with a comparison against the most-popular baseline, scored on the same queries over the full 200-item catalog (seen items excluded):

```text
test metric      model  most-popular
ndcg@10         0.3053        0.1523
recall@10       0.3950        0.2034
hit_rate@10     0.7269        0.4769
coverage@10     0.8550        0.1250
```

## Source

??? example "`examples/synthetic/data.py`"

    ```python
    --8<-- "examples/synthetic/data.py"
    ```

??? example "`examples/synthetic/config.py`"

    ```python
    --8<-- "examples/synthetic/config.py"
    ```

??? example "`examples/synthetic/model.py`"

    ```python
    --8<-- "examples/synthetic/model.py"
    ```

??? example "`examples/synthetic/run_training.py`"

    ```python
    --8<-- "examples/synthetic/run_training.py"
    ```

## Things to notice

- `PrepareSyntheticDataFrame` implements only `requires`, `dataset_dir`, `read_data_frame_path` and `timestamp_property`. That is the minimum a project needs.
- `run_evaluate(task_dir, ...)` rebuilds the task from `params.json` and gives the same metrics as the training run. It is the Python form of `mars-rec evaluate`.
- `mars-rec predict` is not shown here. It returns a top-k list only for models that score the whole catalogue in one forward pass (`[batch, n_items]`), such as [Last.fm next-item](lastfm-next-item.md).
