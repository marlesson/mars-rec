# MovieLens 100K: slate ranking

**Demonstrates:** **Slate-based negatives**: every row has a candidate slate (`available_items`), and `InteractionsWithNegativeItemGenerationByAvailableItemsDataset` samples negatives from it instead of the full catalogue. It also uses a 19-dim genre vector as a `metadata_columns` feature.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) with synthetic per-row slates made of 1 positive and 49 random items. This mirrors the "leave-one-out + N random negatives" benchmark protocol.

## Model

`ML100KSlateRanker` is a two-tower model: a user-id embedding on one side and (item-id embedding ⊕ genre vector) on the other, scored by dot product. Items seen rarely can rely on their genre, and frequent items on their id embedding.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.movielens100k_slate.config.movielens100k_slate \
      --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
      --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
      --epochs 8 --batch-size 256 --loss-function bce_logits --device cpu
    ```

## Source

The snippets below are included from the repository, so they always match the code.

??? example "`examples/movielens100k_slate/config.py`"

    ```python
    --8<-- "examples/movielens100k_slate/config.py"
    ```

??? example "`examples/movielens100k_slate/model.py`"

    ```python
    --8<-- "examples/movielens100k_slate/model.py"
    ```

??? example "`examples/movielens100k_slate/run_training.py`"

    ```python
    --8<-- "examples/movielens100k_slate/run_training.py"
    ```

## Things to notice

- This is the smallest and fastest example, and the one used in the [Quickstart](../getting-started/quickstart.md) and the end-to-end smoke test.
- The slate logic lives entirely in the dataset class. The model scores positives and negatives the same way.
