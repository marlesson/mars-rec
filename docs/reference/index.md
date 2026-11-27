# Reference

- **[CLI](cli.md)**: every `mars-rec` command and option, generated from the Click definitions.
- **API**: the Python API, generated from the docstrings in `src/mars_rec/`:

| Module | Contents |
|---|---|
| [`mars_rec.meta_config`](api/meta_config.md) | `ProjectConfig`, `Column`, `IOType`, `RecommenderType` |
| [`mars_rec.simulation.training`](api/training.md) | `SupervisedModelTraining`, `run_evaluate`, the loss / optimizer registries |
| [`mars_rec.inference`](api/inference.md) | `run_inference`: standalone batch top-k prediction |
| [`mars_rec.filtering`](api/filtering.md) | `CandidateProvider` / `CandidateFilter`: inference-time candidate pools |
| [`mars_rec.visualization`](api/visualization.md) | Architecture diagrams (SVG) and text summaries |
| [`mars_rec.data`](api/data.md) | `BasePrepareDataFrames`, dataset classes, download helpers |
| [`mars_rec.model`](api/model.md) | `RecommenderModule` base class and the `LogisticRegression` baseline |
| [`mars_rec.torch`](api/torch.md) | Lightning module, losses, metrics, optimizers, callbacks |
| [`mars_rec.utils`](api/utils.md) | Artifact paths, index mapping, reflection, plotting |
