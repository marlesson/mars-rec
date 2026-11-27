---
hide:
  - navigation
---

# MARS-Rec

**MARS-Rec** is a framework to model, train, and evaluate deep-learning recommendation systems.
It is the supervised-only successor of [MARS-Gym](https://github.com/deeplearningbrasil/mars-gym)
(formerly *MARS-Gym Lite*): all Reinforcement Learning, bandit, gym-environment and policy-evaluation
layers were removed, keeping a lean **data preparation → supervised training → evaluation** pipeline.

- :material-file-cog: **Declarative projects.** A single [`ProjectConfig`](concepts/project-config.md)
  describes columns, types, the data-prep task and the dataset class.
- :material-pipe: **Reproducible pipelines.** [Luigi](https://luigi.readthedocs.io/) caches every step by a
  hash of its parameters, so re-running a finished experiment is free.
- :material-lightning-bolt: **PyTorch Lightning training loop.** Includes early stopping, checkpointing,
  CSV logging (TensorBoard via the `tensorboard` extra), gradient clipping and LR schedulers.
- :material-chart-line: **Ranking evaluation.** Ranks the full catalog for every test query and reports
  NDCG, HitRate, Precision, Recall, MRR and MAP, plus coverage, Gini, popularity and novelty, all
  @{5, 10, 20}. Each run is compared with a most-popular baseline and ROC-AUC is included
  ([protocol](concepts/evaluation.md)).
- :material-console: **One CLI.** `mars-rec` handles training, comparing runs, batch inference,
  re-evaluation and architecture diagrams.
- :material-cloud-upload: **Docker & Vertex AI.** The same image runs locally and as a Vertex AI
  Custom Training job.

## At a glance

```bash
pip install -e .

mars-rec run supervised \
  --project examples.movielens100k_slate.config.movielens100k_slate \
  --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
  --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
  --loss-function bce_logits --epochs 8 --device cpu

mars-rec results --project movielens100k_slate --sort-by val_loss
```

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started/index.md)**

    Install the package and train a bundled example in minutes.

- :material-lightbulb-on: **[Concepts](concepts/index.md)**

    Learn about `ProjectConfig`, the `forward()` contract, index mapping and Luigi caching.

- :material-school: **[Tutorials](tutorials/index.md)**

    Build your own project, or walk through the five reference examples.

- :material-api: **[Reference](reference/index.md)**

    Reference for every CLI option and the full Python API.

</div>

## MARS-Rec vs. MARS-Gym

| Aspect | MARS-Gym | MARS-Rec |
|---|---|---|
| Data processing | PySpark + Pandas | Pandas only |
| Training | Supervised + Bandit/RL | Supervised only |
| Evaluation | Bandit policies, counterfactual | Score-based ranking |
| Prediction | `BanditAgent` with policies (ε-greedy, …) | Direct ranking by model score |
| Orchestration | Luigi | Luigi |
| Training loop | Torchbearer | PyTorch Lightning |
| CLI | `run supervised` / `interaction` / `evaluate` / `viz` | `run supervised` / `run data` / `list` / `results` / `predict` / `evaluate` / `visualize` |
