# Dependencies

`pyproject.toml` declares **loose floors** so that `pip install -e .` stays quick. The full pinned
tree lives in `requirements.lock`. It is resolved for Python 3.11, the oldest supported version,
so it installs on every supported version. The Dockerfile and CI install from the lock.
Lock-only pins, such as keeping torch on a CUDA 12.x release, live in `lock-constraints.txt`.

```bash
make lock            # re-resolve requirements.lock after editing pyproject.toml
make lock-upgrade    # bump every package to the newest allowed version
make install-locked  # install the pinned stack + dev tools (what CI tests)
make audit           # check the lockfile against the OSV vulnerability database
```

CI also runs a **non-blocking** job against the newest versions allowed by `pyproject.toml`. It
gives early warning before the next upgrade.

## Upgrading the pinned stack

1. Run `make lock-upgrade`, then `make install-locked`.
2. `make test` and `make audit` must pass. `tests/test_dependency_regression.py` checks that
   checkpoints produced by the previous stack still load and score the same. It also checks that
   a seeded training run stays close to its golden history.
3. Run an example end to end:
   `PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py`.
4. If the golden history drifted for an expected reason, review the drift, then refresh it with
   `PYTHONPATH=. python scripts/make_regression_fixtures.py --golden-only`.
   **Do not** regenerate `tests/_fixtures/legacy_artifacts/` to silence a failure.

## Documentation dependencies

The docs toolchain (MkDocs Material, mkdocstrings, mkdocs-click) is an optional extra:

```bash
pip install -e ".[docs]"
make docs-serve
```
