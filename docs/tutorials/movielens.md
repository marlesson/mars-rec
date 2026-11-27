# MovieLens 1M: Neural Collaborative Filtering

**Demonstrates:** Implicit-feedback collaborative filtering with **side features**: user demographics as per-row `NUMBER` inputs and an 18-dim multi-hot genre vector as per-item `metadata_columns`.

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) from GroupLens: about 1M ratings from 6,040 users on 3,706 movies. A rating ≥ 4 counts as a positive. The dataset is downloaded automatically from `files.grouplens.org`.

## Model

`MovieLensNCF` combines a **GMF** path (element-wise product of the user and item embeddings) with an **MLP** path over the embeddings, user features and genres. Both paths are concatenated and passed to a linear layer. It is trained with `bce_logits`, uniform negative sampling (`negative_proportion=0.5`) and a temporal split.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/movielens/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.movielens.config.movielens \
      --recommender-module-class examples.movielens.model.MovieLensNCF \
      --recommender-extra-params '{"n_factors": 32, "hidden_size": 128, "n_layers": 3, "dropout": 0.2, "genre_dim": 18}' \
      --epochs 10 --batch-size 512 --loss-function bce_logits \
      --negative-proportion 0.5 --dataset-split-method time --test-split-type time \
      --ranking-metrics --device cpu
    ```

## Source

The snippets below are included from the repository, so they always match the code.

??? example "`examples/movielens/config.py`"

    ```python
    --8<-- "examples/movielens/config.py"
    ```

??? example "`examples/movielens/model.py`"

    ```python
    --8<-- "examples/movielens/model.py"
    ```

??? example "`examples/movielens/run_training.py`"

    ```python
    --8<-- "examples/movielens/run_training.py"
    ```

## Things to notice

- Genres are declared in `metadata_columns`, not `other_input_columns`. Negative sampling swaps the item id and re-looks-up metadata for the sampled item, but it copies `other_input_columns` from the positive row unchanged. Item features stored per row would therefore leak the label: every negative would carry the positive movie's genres. Keep **item** features in `metadata_columns` and only **row/user/context** features in `other_input_columns`.
- `forward(*args)` receives the user features first, then the genre vector, because `input_columns` puts `metadata_columns` after `other_input_columns`.
- `available_items_column_name=None` turns off slate sampling. Negatives come from the whole catalogue through `InteractionsWithNegativeItemGenerationDataset`.
