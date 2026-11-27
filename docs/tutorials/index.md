# Tutorials

Start with **[Your first project](custom-project.md)** to build a project from scratch. Then read
the reference examples. Each one is a self-contained package in `examples/<name>/` (`data.py`,
`config.py`, `model.py`, `run_training.py`) and downloads a public dataset on first run.

| Example | Dataset | Framework feature it exercises |
|---|---|---|
| [MovieLens 1M](movielens.md) | MovieLens 1M | Implicit-feedback CF with side features (demographics + genre multi-hot) |
| [Last.fm](lastfm.md) | HetRec Last.fm 2K | Sequence column via `INDEXABLE_ARRAY` + `same_index_as` (SASRec-lite) |
| [Online Retail](online-retail.md) | UCI Online Retail | `user_column` bound to `session_id`: session-based GRU4Rec |
| [Amazon Beauty](amazon-beauty.md) | Amazon Beauty 5-core | `metadata_columns` with a `FLOAT_ARRAY` text embedding, content-based two-tower |
| [MovieLens 100K slate](movielens100k-slate.md) | MovieLens 100K + synthetic slates | `available_items` slate negatives + `metadata_columns` |
