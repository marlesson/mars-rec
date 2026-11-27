# Installation

MARS-Rec supports **Python 3.11 – 3.13**.

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
| `imbalanced-learn >= 0.13` | Over/undersampling |
| `pyarrow >= 15` | Parquet I/O for `predict` / `evaluate` |

## `PYTHONPATH`

Projects, models and configs are passed to the framework as **dotted import paths**
(for example `examples.movielens.config.movielens`). The directory that contains your project
package must be on `PYTHONPATH`. From the repository root:

```bash
PYTHONPATH=src:. python examples/movielens/run_training.py
```

When you run `mars-rec` from the repository root, the current directory is usually importable already.
If you see `ModuleNotFoundError: No module named 'examples'`, add `PYTHONPATH=.`.
