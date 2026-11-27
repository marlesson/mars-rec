# Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `RuntimeError: target size != input size` | `forward()` returns the wrong shape for the loss | BCE / MSE-style losses expect `[batch]` (add `.squeeze(-1)`). `ce` expects `[batch, n_items]`. See [Models](../concepts/models.md#forward-output-shape) |
| `ModuleNotFoundError: No module named 'examples'` (or your package) | `PYTHONPATH` does not include the project root | Run from the root, or prefix the command with `PYTHONPATH=.` |
| The run finishes instantly / *task already DONE* | The output directory exists from a previous run | Use `--no-cache`, change a parameter (e.g. `--observation`), or delete the directory. See [Outputs & caching](../concepts/outputs.md#caching-and-no-cache) |
| `KeyError` in index_mapping | An `INDEXABLE` column is missing from the DataFrame | Make the `ProjectConfig` column names match the CSV |
| `issubclass() arg 1 must be a class` | `recommender_module_class` points to an instance | Point it to the **class** |
| `PermissionError` on cleanup | A previous process still holds a file open | Delete the output directory manually |
| `FileNotFoundError: weights*.ckpt` in `predict` / `evaluate` | The task directory has no checkpoint | Re-train, or pass the correct `--task-id` / `--model-dir` |
| Ranking metrics missing from `results` | The run was trained without `--ranking-metrics` | Run `mars-rec evaluate --task-id <ID> --ranking-metrics` |
| `no match for platform in manifest` on Vertex AI | The image was built for arm64 (Apple Silicon) | Push with `make docker-push`, which forces `linux/amd64` |
