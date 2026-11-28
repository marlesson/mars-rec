# Last.fm next-item: full-softmax CE (SASRec)

**Demonstrates:** Next-item prediction with **`loss_function="ce"`**: the model returns logits over the whole item vocabulary (`[batch, n_items]`), the target is the item's index, and there is no negative sampling. This is also the output shape that `mars-rec predict` turns into a top-k list per row.

## Dataset

The same [HetRec 2011 Last.fm 2K](https://grouplens.org/datasets/hetrec-2011/) stream as the [Last.fm](lastfm.md) example: about 76K events, 1.2K users, 6.8K artists and a 10-artist history per event. The config reuses `PrepareLastFMDataFrame`, so both examples share the downloaded and split CSVs.

## Model

`LastFMNextItemSASRec` encodes the history with the same causal Transformer as `LastFMSASRec`. Instead of scoring one candidate, it multiplies the last hidden state by the whole item embedding table (`h_u @ item_emb.weight.T`), so the output head is tied to the input embeddings.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/lastfm_next_item/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.lastfm_next_item.config.lastfm_next_item \
      --recommender-module-class examples.lastfm_next_item.model.LastFMNextItemSASRec \
      --recommender-extra-params '{"n_factors": 64, "n_heads": 2, "ff_dim": 128, "history_len": 10}' \
      --epochs 30 --learning-rate 0.002 --batch-size 256 --loss-function ce \
      --dataset-split-method time --test-split-type time \
      --ranking-metrics --sample-size-eval 2000 --metrics '["loss"]' --device cpu
    ```

On the full-catalog test evaluation (2,000 sampled queries, seen artists excluded), this reaches `ndcg@10` ≈ 0.106 and `recall@10` ≈ 0.146, against 0.025 / 0.055 for the most-popular baseline.

Then get the top-10 artists for each row of any CSV with the same columns:

```bash
mars-rec predict --task-id <TASK_ID> --data <test.csv> --top-k 10
```

## Source

??? example "`examples/lastfm_next_item/config.py`"

    ```python
    --8<-- "examples/lastfm_next_item/config.py"
    ```

??? example "`examples/lastfm_next_item/model.py`"

    ```python
    --8<-- "examples/lastfm_next_item/model.py"
    ```

??? example "`examples/lastfm_next_item/run_training.py`"

    ```python
    --8<-- "examples/lastfm_next_item/run_training.py"
    ```

## Things to notice

- `item_is_input=False` removes the target artist from `forward()`'s arguments, so `forward(user_ids, history_ids)` cannot peek at the label.
- `output_column=Column("artist_id", IOType.INDEXABLE)` makes the batch target the artist's **index**, which is what `CrossEntropyLoss` expects for `[batch, n_items]` logits.
- `InteractionsDataset` is used on purpose. The softmax already treats every other artist as a negative, so a negative-sampling dataset would only add noise.
- The reserved index slots (unknown / NaN / PAD) are part of the vocabulary and stay in the logits. Inference masks them before taking the top-k.
- For the [ranking evaluation](../concepts/evaluation.md), every row is a query (it carries its own history) and the logits are read directly for the whole catalog, with no per-candidate forward pass. `sample_size_eval=2000` caps the number of queries and also bounds the one-vs-rest ROC-AUC, whose cost grows with the number of classes.
