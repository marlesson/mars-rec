# Tutorials

Start with **[Your first project](custom-project.md)** to build a project from scratch. Then read
the reference examples. Each one is a self-contained package in `examples/<name>/` (`data.py`,
`config.py`, `model.py`, `run_training.py`). All but `synthetic` download a public dataset on first run.

| Example | Dataset | Framework feature it exercises |
|---|---|---|
| [Synthetic](synthetic.md) | Generated offline | Full lifecycle in seconds with no download (matrix factorisation, `run_evaluate`) |
| [MovieLens 1M](movielens.md) | MovieLens 1M | Implicit-feedback CF with side features (per-row demographics + per-item genre `metadata_columns`) |
| [Last.fm](lastfm.md) | HetRec Last.fm 2K | Sequence column via `INDEXABLE_ARRAY` + `same_index_as` (SASRec-lite, one sampled candidate per row with `bce_logits`) |
| [Last.fm next-item](lastfm-next-item.md) | HetRec Last.fm 2K | Full-softmax `ce` over the item vocabulary (`[batch, n_items]`), `item_is_input=False`, `mars-rec predict` |
| [Online Retail](online-retail.md) | UCI Online Retail | `user_column` bound to `session_id`: session-based GRU4Rec |
| [Amazon Beauty](amazon-beauty.md) | Amazon Beauty 5-core | `metadata_columns` with a `FLOAT_ARRAY` text embedding, content-based two-tower |
| [MovieLens 100K slate](movielens100k-slate.md) | MovieLens 100K + synthetic slates | `available_items` slate negatives + `metadata_columns` |
| [MovieLens 100K retrieval](movielens100k-retrieval.md) | MovieLens 100K | Embedding loss `infonce` with in-batch negatives, `get_target_emb`, `EmbeddingRecommenderLightningModule` |
