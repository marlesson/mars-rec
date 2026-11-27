# Extending the framework

!!! note "Framework vs. application code"
    Contributions to `src/mars_rec/` must stay **domain-agnostic**: no hard-coded column names,
    dataset names or competition-specific logic. Custom models and data tasks live in your own
    project, which you reference by dotted path.

## Adding a model or dataset

You do not need to change the framework. Write a `RecommenderModule` subclass and a
`BasePrepareDataFrames` subclass in your project. See [Your first project](../tutorials/custom-project.md).

## Adding a loss function

1. Implement it in [`mars_rec/torch/loss.py`](../reference/api/torch.md#mars_rec.torch.loss). It must
   accept `(input, target)` and return a scalar.
2. Register it in `TORCH_LOSS_FUNCTIONS` in `mars_rec/simulation/training.py`.
3. If it consumes **logits**, also add it to `_LOGIT_LOSSES` in `mars_rec/torch/lightning_module.py`.
   Metrics that expect probabilities then receive `sigmoid(score)`.

```python
class MyLoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...
```

## Adding a metric

1. Implement `fn(y_pred, y_true) -> torch.Tensor` in
   [`mars_rec/torch/metrics.py`](../reference/api/torch.md#mars_rec.torch.metrics).
2. Add it to `METRIC_REGISTRY` in the same file. The registry key is the name you pass in
   `metrics` (for example `"ndcg@10"`).

## Adding an optimizer or LR scheduler

1. Implement or import it (custom optimizers live in `mars_rec/torch/optimizer.py`).
2. Register it in `TORCH_OPTIMIZERS` (or `TORCH_LR_SCHEDULERS`) in `mars_rec/simulation/training.py`.

## Adding a dataset class

Subclass [`InteractionsDataset`][mars_rec.data.dataset.InteractionsDataset] to change how samples
and negatives are produced, then reference it from `ProjectConfig.dataset_class`. A generic class
can go in `mars_rec/data/dataset.py`. A project-specific one belongs in your project.

## Before opening a PR

```bash
make format && make lint && make test
make docs          # strict docs build, which also catches broken API references
```

See [Contributing](../contributing.md) for the full workflow.
