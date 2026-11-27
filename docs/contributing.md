--8<-- "CONTRIBUTING.md"

## Documentation

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). The API
reference comes from the docstrings through [mkdocstrings](https://mkdocstrings.github.io/), and the
CLI reference from the Click definitions.

```bash
pip install -e ".[docs]"
make docs-serve   # live-reload preview at http://127.0.0.1:8000
make docs         # strict build into site/, the same check CI runs
```

- Pages live in `docs/` and the navigation is defined in `mkdocs.yml`.
- Example tutorials include source files with `--8<-- "examples/<name>/model.py"`, so they stay in sync with the code.
- Docstrings may use Google (`Args:`) or Sphinx (`:param x:`) sections. Sphinx roles such as
  ``:class:`mars_rec.meta_config.ProjectConfig` `` are turned into links by `docs/_ext/griffe_sphinx_roles.py`.
- CI runs the strict build on every PR that touches the docs or the code (`.github/workflows/docs.yml`).
  Publishing to GitHub Pages is opt-in through the `DOCS_DEPLOY` repository variable.
