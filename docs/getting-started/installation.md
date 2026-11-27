# Installation

MARS-Rec supports **Python 3.11 – 3.13**.

## From PyPI

```bash
pip install mars-rec                  # the package + the `mars-rec` CLI
pip install "mars-rec[tensorboard]"   # with TensorBoard logging
```

This is the way to go when you use MARS-Rec as a library in your own project. The bundled
`examples/` are not part of the wheel; clone the repository to run them.

## From source

```bash
git clone https://github.com/marlesson/mars-rec.git
cd mars-rec

python -m venv .venv
source .venv/bin/activate

pip install -e .          # the package + the `mars-rec` CLI
# or
make install-dev          # package + dev tools (pytest, ruff)
```

Optional extras:

```bash
pip install -e ".[tensorboard]"   # TensorBoard logging during training (CSV logging is always on)
```

Check that the installation works:

```bash
mars-rec --help
python -c "from mars_rec.simulation.training import SupervisedModelTraining; print('OK')"
```

## Reproducible install (pinned stack)

`pip install -e .` resolves the loose version floors from `pyproject.toml`. To install
**exactly** what CI and the Docker image use, install from the lockfile:

```bash
make install-locked

# CPU-only torch wheels (smaller download), as CI does:
make install-locked TORCH_INDEX=https://download.pytorch.org/whl/cpu
```

For more on the lockfile and how to upgrade it, see [Dependencies](../guides/dependencies.md).

## Main dependencies

| Package | Role |
|---|---|
| `torch >= 2.4` | Models |
| `pytorch-lightning >= 2.4, < 3` | Training loop, callbacks, logging |
| `luigi >= 3.5, < 4` | Pipeline orchestration and caching |
| `pandas >= 2.2` | Data manipulation (pandas 3.x is supported) |
| `scikit-learn >= 1.5` | Splits and metrics |
| `matplotlib >= 3.8` | Training plots (history, scores, ROC/PR, confusion matrix) |
| `pyarrow >= 15` | Parquet I/O for `predict` / `evaluate` |
| `click`, `tqdm`, `certifi` | CLI, progress bars, CA bundle for dataset downloads |

TensorBoard logging is optional: install the `tensorboard` extra to enable it.

## `PYTHONPATH`

Projects, models and configs are passed to the framework as **dotted import paths**
(for example `examples.movielens.config.movielens`). The directory that contains your project
package must be on `PYTHONPATH`. From the repository root:

```bash
PYTHONPATH=src:. python examples/movielens/run_training.py
```

When you run `mars-rec` from the repository root, the current directory is usually importable already.
If you see `ModuleNotFoundError: No module named 'examples'`, add `PYTHONPATH=.`.
