# Amazon Beauty: content-based two-tower

**Demonstrates:** `metadata_columns` with a dense `FLOAT_ARRAY` **text embedding** that is stored once per item in a metadata CSV and attached to each interaction by item index at sample time.

## Dataset

[Amazon Beauty 5-core reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): about 198K reviews, 22K users and 12K items. Each item's review text is embedded with TF-IDF + TruncatedSVD into a 64-d vector.

## Model

`AmazonBeautyTwoTower` has a **user tower** (user-id embedding followed by an MLP) and an **item tower** (precomputed text embedding followed by an MLP). The score is their dot product. The item tower has **no per-item-id parameter**, so the model generalises to cold items.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/amazon_beauty/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.amazon_beauty.config.amazon_beauty \
      --recommender-module-class examples.amazon_beauty.model.AmazonBeautyTwoTower \
      --recommender-extra-params '{"n_factors": 64, "text_emb_dim": 64, "hidden_size": 64, "dropout": 0.1}' \
      --epochs 5 --batch-size 512 --loss-function bce_logits \
      --negative-proportion 0.5 --dataset-split-method time --test-split-type time \
      --ranking-metrics --device cpu
    ```

## Source

The snippets below are included from the repository, so they always match the code.

??? example "`examples/amazon_beauty/config.py`"

    ```python
    --8<-- "examples/amazon_beauty/config.py"
    ```

??? example "`examples/amazon_beauty/model.py`"

    ```python
    --8<-- "examples/amazon_beauty/model.py"
    ```

??? example "`examples/amazon_beauty/run_training.py`"

    ```python
    --8<-- "examples/amazon_beauty/run_training.py"
    ```

## Things to notice

- The data-prep task exposes `metadata_data_frame_path`. Metadata columns are appended **after** the other inputs in `forward()`.
- Use this pattern for any dense per-item feature (text, image or graph embeddings). It would otherwise inflate the interaction CSV.
