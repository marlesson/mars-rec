# Last.fm: sequential recommendation (SASRec-lite)

**Demonstrates:** A **sequence column**: the listening history is declared as `IOType.INDEXABLE_ARRAY` with `same_index_as` the item column, so history and target share one item embedding.

## Dataset

[HetRec 2011 Last.fm 2K](https://grouplens.org/datasets/hetrec-2011/): about 76K tagging events after filtering, 1.2K users and 6.8K artists. It is used as an ordered click-stream per user with a fixed-length history window.

## Model

`LastFMSASRec` adds a shared item embedding to a positional embedding, runs a **1-layer causal Transformer** over `history_artist_ids` (PAD positions are masked), and takes the last non-PAD hidden state. The score is its dot product with the target item embedding.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/lastfm/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.lastfm.config.lastfm \
      --recommender-module-class examples.lastfm.model.LastFMSASRec \
      --recommender-extra-params '{"n_factors": 64, "n_heads": 2, "ff_dim": 128, "history_len": 10}' \
      --epochs 5 --batch-size 256 --loss-function bce_logits \
      --negative-proportion 0.5 --dataset-split-method time --device cpu
    ```

## Source

The snippets below are included from the repository, so they always match the code.

??? example "`examples/lastfm/config.py`"

    ```python
    --8<-- "examples/lastfm/config.py"
    ```

??? example "`examples/lastfm/model.py`"

    ```python
    --8<-- "examples/lastfm/model.py"
    ```

??? example "`examples/lastfm/run_training.py`"

    ```python
    --8<-- "examples/lastfm/run_training.py"
    ```

## Things to notice

- `PAD_INDEX = 2` matches the [reserved pad slot](../concepts/index-mapping.md#reserved-slots) and is used as `padding_idx` of the item embedding.
- The model scores one target item per row (`[batch_size]`). To predict over the full vocabulary instead, return `[batch_size, n_items]` logits and train with `--loss-function ce`.
