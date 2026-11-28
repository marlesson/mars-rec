# MARS-Rec

[![CI](https://github.com/marlesson/mars-rec/actions/workflows/ci.yml/badge.svg)](https://github.com/marlesson/mars-rec/actions/workflows/ci.yml)
[![Docs](https://github.com/marlesson/mars-rec/actions/workflows/docs.yml/badge.svg)](https://marlesson.github.io/mars-rec/)

**MARS-Rec** is a framework to model, train, and evaluate deep-learning recommendation systems.
It is the supervised-only successor of [MARS-Gym](https://github.com/deeplearningbrasil/mars-gym)
(formerly *MARS-Gym Lite*). All Reinforcement Learning, bandit, gym-environment and
policy-evaluation layers were removed, keeping a lean **data preparation → supervised training →
evaluation** pipeline built on Luigi, PyTorch and PyTorch Lightning.

📖 **Documentation: <https://marlesson.github.io/mars-rec/>**

## Features

- **Declarative projects**: a single `ProjectConfig` declares the columns, their types, the
  data-prep task and the dataset class.
- **Reproducible pipelines**: Luigi caches every step by a hash of its parameters.
- **PyTorch Lightning training loop**: early stopping, checkpointing, CSV logging (TensorBoard optional),
  LR schedulers and mixed precision.
- **Ranking evaluation**: full-catalog top-k ranking with seen items excluded. Reports NDCG, HitRate,
  Precision, Recall, MRR and MAP, plus coverage, Gini, popularity and novelty, all @{5, 10, 20}.
  Includes a most-popular baseline and ROC-AUC.
- **One CLI** (`mars-rec`): `run`, `list`, `results`, `predict`, `evaluate` and `visualize`.
- **Docker & Vertex AI**: the same image runs locally and on Google Cloud.

## Installation

Python 3.11–3.13.

```bash
pip install mars-rec
```

From source (to run the bundled `examples/` or contribute):

```bash
git clone https://github.com/marlesson/mars-rec.git
cd mars-rec
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: make install-locked (pinned stack, as CI)
```

## Quickstart

```bash
# Train the smallest bundled example (downloads MovieLens 100K on first run)
mars-rec run supervised \
  --project examples.movielens100k_slate.config.movielens100k_slate \
  --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
  --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
  --loss-function bce_logits --epochs 8 --device cpu

# Compare runs, predict, re-evaluate
mars-rec results --project movielens100k_slate --sort-by val_loss
mars-rec predict  --task-id <TASK_ID> --data data/test.parquet --top-k 10
mars-rec evaluate --task-id <TASK_ID> --ranking-metrics --split test
```

A project has three parts: a data-prep task, a `ProjectConfig` and a model.

```python
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.abstract import RecommenderModule

my_project = ProjectConfig(
    base_dir="output/my_project",
    prepare_data_frames_task=PrepareMyDataFrame,          # a BasePrepareDataFrames subclass
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[Column("price", IOType.NUMBER)],
    output_column=Column("purchased", IOType.NUMBER),
)

class MyModel(RecommenderModule):
    def forward(self, user_ids, item_ids, price):  # same order as the input columns
        ...                                        # → one score per row: [batch_size]
```

For a step-by-step walkthrough, see
[Your first project](https://marlesson.github.io/mars-rec/tutorials/custom-project/).

## Examples

| Example | Dataset | Feature it exercises |
|---|---|---|
| `movielens` | MovieLens 1M | NCF with user/genre side features |
| `lastfm` | HetRec Last.fm 2K | Sequential SASRec-lite (`INDEXABLE_ARRAY` + `same_index_as`) |
| `online_retail` | UCI Online Retail | Session-based GRU4Rec (`session_id` as user) |
| `amazon_beauty` | Amazon Beauty 5-core | Content-based two-tower (`metadata_columns` text embedding) |
| `movielens100k_slate` | MovieLens 100K | Slate ranking (`available_items` negatives) |

```bash
PYTHONPATH=src:. python examples/<name>/run_training.py
```

## Documentation

| Section | Content |
|---|---|
| [Getting started](https://marlesson.github.io/mars-rec/getting-started/) | Installation and quickstart |
| [Concepts](https://marlesson.github.io/mars-rec/concepts/) | `ProjectConfig`, models, pipeline, index mapping, caching |
| [Tutorials](https://marlesson.github.io/mars-rec/tutorials/) | Building a project, and the five examples |
| [Guides](https://marlesson.github.io/mars-rec/guides/) | Training parameters, inference, extending, Docker & Vertex AI |
| [Reference](https://marlesson.github.io/mars-rec/reference/) | CLI and Python API |

To preview the docs locally: `pip install -e ".[docs]" && make docs-serve`.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/marlesson/mars-rec/blob/main/CONTRIBUTING.md) for the branch, lint and pull request workflow.
AI agents should also read [`AGENTS.md`](https://github.com/marlesson/mars-rec/blob/main/AGENTS.md).

## License

MIT
