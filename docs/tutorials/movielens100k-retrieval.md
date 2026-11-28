# MovieLens 100K retrieval: two-tower + InfoNCE

**Demonstrates:** **Embedding losses** (`infonce`). The model returns an L2-normalised query embedding and exposes `get_target_emb(item_ids)` for the item side. The framework then switches to `EmbeddingRecommenderLightningModule`, and the other targets in each batch act as negatives.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): each positive rating (≥ 4) becomes one row with the user, the target movie and the user's previous 20 positive movies (about 54K rows, 942 users, 1.4K movies). The dataset is downloaded automatically from `files.grouplens.org`.

## Model

`ML100KTwoTowerRetrieval` has two towers:

- the **query tower** concatenates the user embedding with the mean-pooled history embedding, then applies an MLP;
- the **item tower** applies an MLP to the item embedding.

The history and the targets share one item table. Both outputs are L2-normalised, so the score is the cosine similarity. The item tower depends only on the item id, so every item vector can be precomputed and served from an approximate-nearest-neighbour index.

### logQ correction

With in-batch negatives, a popular movie is a negative far more often than a rare one, so plain InfoNCE learns to push popular movies down. The model applies the standard **logQ correction** ([Yi et al., RecSys 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)). It trains on `sim(q, j) / τ − log p̂(j)`, where `p̂(j)` is a streaming estimate of how often movie `j` is a training target. The correction is carried in one extra embedding dimension (`[q, 1]` · `[v_j, −τ·log p̂(j)]`), so `InfoNCELoss` stays unchanged. The extra dimension is zeroed in eval mode, so the ranking uses the plain `q · v_j`. Set the model's `temperature` to the loss `temperature`.

On the full-catalog test evaluation (seen movies excluded):

| | ndcg@10 | recall@10 |
|---|---|---|
| two-tower, `logq_correction=True` | **0.047** | **0.093** |
| most-popular baseline | 0.026 | 0.055 |
| two-tower, `logq_correction=False` | 0.005 | 0.013 |

Without the correction the model scores 5× below the popularity baseline.

## Run it

=== "Script"

    ```bash
    PYTHONPATH=src:. python examples/movielens100k_retrieval/run_training.py
    ```

=== "CLI"

    ```bash
    mars-rec run supervised \
      --project examples.movielens100k_retrieval.config.movielens100k_retrieval \
      --recommender-module-class examples.movielens100k_retrieval.model.ML100KTwoTowerRetrieval \
      --recommender-extra-params '{"n_factors": 64, "hidden_size": 128, "dropout": 0.1, "logq_correction": true, "temperature": 0.1}' \
      --epochs 20 --batch-size 512 --loss-function infonce \
      --loss-function-params '{"temperature": 0.1}' \
      --dataset-split-method time --test-split-type time \
      --ranking-metrics --metrics '["loss"]' --device cpu
    ```

Training logs `avg_cos_sim` / `val_avg_cos_sim` (mean query–target similarity) next to the loss. The [ranking evaluation](../concepts/evaluation.md) embeds the whole catalog once with `get_target_emb` and scores each query as `query · item` for every candidate.

## Source

??? example "`examples/movielens100k_retrieval/data.py`"

    ```python
    --8<-- "examples/movielens100k_retrieval/data.py"
    ```

??? example "`examples/movielens100k_retrieval/config.py`"

    ```python
    --8<-- "examples/movielens100k_retrieval/config.py"
    ```

??? example "`examples/movielens100k_retrieval/model.py`"

    ```python
    --8<-- "examples/movielens100k_retrieval/model.py"
    ```

??? example "`examples/movielens100k_retrieval/run_training.py`"

    ```python
    --8<-- "examples/movielens100k_retrieval/run_training.py"
    ```

## Things to notice

- The `loss_function` alone selects the Lightning module: `infonce`, `cosine_similarity` and `hardneg_infonce` route to `EmbeddingRecommenderLightningModule`.
- As in the CE example, `item_is_input=False` and `output_column` is the item column. Here the target ids are embedded with `get_target_emb`, not used as class labels.
- There is no negative sampling (`InteractionsDataset`). A larger `batch_size` means more in-batch negatives per query, and `temperature` controls how sharply the loss separates them.
- In-batch negatives follow item popularity, which is why the logQ correction above matters. `hardneg_infonce` adds harder negatives per query on top.
- Validation runs in eval mode, so `val_loss` is the uncorrected InfoNCE loss. Use it for early stopping, not as a ranking metric.
- ROC-AUC is not defined for embedding outputs and is skipped.
