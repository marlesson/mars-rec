# CLI

The `mars-rec` command is installed with the package (`pip install -e .`). This page is
generated from the Click definitions in `mars_rec/cli.py`.

!!! note "`run supervised` forwards Luigi parameters"
    Apart from `--no-cache` and `--task-class`, every flag of `mars-rec run supervised` is passed
    verbatim to the Luigi task. For the list, see [Training parameters](../guides/training-parameters.md).

::: mkdocs-click
    :module: mars_rec.cli
    :command: cli
    :prog_name: mars-rec
    :depth: 1
    :style: table
    :list_subcommands: true
