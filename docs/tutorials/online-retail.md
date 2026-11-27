# Online Retail: session-based recommendation (GRU4Rec)

**Demonstrates:** `user_column` bound to a **session id**. The data has no stable customer identity, which shows that the "user" can be any indexable entity.

## Dataset

[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail): about 510K purchase line-items, 18K invoices and 3.5K stock codes. Each invoice (`InvoiceNo`) is treated as a session.

## Model

`OnlineRetailGRU4Rec` embeds the per-session history with a shared item embedding, runs a **1-layer GRU** over it, and scores the candidate item by dot product with the final hidden state. It uses the same recipe as SASRec-lite, with recurrence instead of self-attention.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/online_retail/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.online_retail.config.online_retail \
      --recommender-module-class examples.online_retail.model.OnlineRetailGRU4Rec \
      --recommender-extra-params '{"n_factors": 64, "hidden_size": 64, "dropout": 0.2}' \
      --epochs 3 --batch-size 512 --loss-function bce_logits \
      --negative-proportion 0.5 --dataset-split-method time --test-split-type time \
      --ranking-metrics --device cpu
    ```

## Source

The snippets below are included from the repository, so they always match the code.

??? example "`examples/online_retail/config.py`"

    ```python
    --8<-- "examples/online_retail/config.py"
    ```

??? example "`examples/online_retail/model.py`"

    ```python
    --8<-- "examples/online_retail/model.py"
    ```

??? example "`examples/online_retail/run_training.py`"

    ```python
    --8<-- "examples/online_retail/run_training.py"
    ```

## Things to notice

- `user_column=Column("session_id", IOType.INDEXABLE)`: nothing in the framework assumes a real user.
- To keep each session entirely on one side of the split, use `--dataset-split-method column --data-frames-preparation-extra-params '{"column_stratification": "session_id"}'` (see [Pipeline](../concepts/pipeline.md#2-data-preparation)).
