# Contributing to MARS-Rec

Thanks for your interest in contributing! This guide keeps the workflow short and predictable.

## Setup

```bash
git clone https://github.com/marlesson/mars-rec.git
cd mars-rec
python -m venv .venv && source .venv/bin/activate
make install-dev
```

Optional: run lint automatically on every commit with `pip install pre-commit && pre-commit install`.

## Workflow

1. `main` is protected; every change goes through a pull request with green CI.
2. Create a short-lived branch from `main` using a type prefix:
   `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`, `chore/…`, `ci/…`.
   AI agents must do this in a dedicated git worktree — see *Work in a git worktree* in [`AGENTS.md`](https://github.com/marlesson/mars-rec/blob/main/AGENTS.md).
3. Before pushing, run:
   ```bash
   make format   # autofix + format with ruff
   make lint     # what CI checks
   make test
   ```
4. Open a PR against `main` and fill in the template.
5. PRs are **squash-merged**. The PR title becomes the commit message on `main`, so write it in
   [Conventional Commits](https://www.conventionalcommits.org/) style, e.g. `feat: add MAP@k metric`
   or `fix(inference): handle empty candidate pool`.

## Guidelines

- All code, comments, docs, commit messages and PR descriptions are written in **English**.
- `src/mars_rec/` must stay domain-agnostic: no hard-coded column names, dataset names or
  competition-specific logic. Application code belongs in a separate project that depends on `mars-rec`.
- Add or update tests under `tests/` for any behavior change. Mark heavyweight end-to-end tests with
  `@pytest.mark.slow`.
- If you change dependencies in `pyproject.toml`, regenerate the lockfile with `make lock` and commit it.
  To upgrade the pinned stack, follow [*Upgrading the pinned stack*](https://marlesson.github.io/mars-rec/guides/dependencies/#upgrading-the-pinned-stack).
- See [`AGENTS.md`](https://github.com/marlesson/mars-rec/blob/main/AGENTS.md) for architecture details and the critical files to modify with care.

## CI

Every PR and every push to `main` runs:

| Job | What it does |
|---|---|
| `lint` | `ruff check` + `ruff format --check` on `src`, `tests`, `examples`, `scripts` |
| `test` | `pytest` on Python 3.11 and 3.13 against the pinned `requirements.lock` (CPU-only torch) |
| `test-latest` | `pytest` on the newest versions allowed by `pyproject.toml`; non-blocking early warning |
| `audit` | `pip-audit` on `requirements.lock` (also runs weekly) |

## Releasing

Releases are published to [PyPI](https://pypi.org/project/mars-rec/) by the `Release` workflow
(`.github/workflows/release.yml`) using PyPI Trusted Publishing, so no API token is stored in the repository.

1. Open a PR that bumps `__version__` in `src/mars_rec/__init__.py` (the only place the version lives;
   `pyproject.toml` reads it). Follow [Semantic Versioning](https://semver.org/).
2. After it is merged, tag the merge commit on `main` and push the tag:
   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```
3. The workflow checks that the tag matches `__version__`, builds the sdist + wheel, smoke-tests the wheel,
   publishes to PyPI (job environment `pypi`) and creates a GitHub Release with generated notes.

A published version can never be re-uploaded to PyPI: if something goes wrong after publishing, bump the
patch version and release again.
