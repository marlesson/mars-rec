"""Click-based CLI for mars-rec.

Exposes the framework's main entry points as ``mars-rec`` shell
commands installed by ``pip install -e .``:

- ``mars-rec run supervised`` — dispatch a
  :class:`SupervisedModelTraining` (or any task fully-qualified via
  ``--task-class``) through Luigi.
- ``mars-rec run data <module> <TaskClass>`` — run an arbitrary
  data-preparation Luigi task.
- ``mars-rec list`` — list previously executed runs.
- ``mars-rec results`` — compare per-run metrics / hyperparameters.
- ``mars-rec predict`` / ``evaluate`` — wrappers around
  :mod:`mars_rec.inference` and
  :func:`mars_rec.simulation.training.run_evaluate`.
- ``mars-rec visualize`` — render the architecture diagram of a
  trained run (see :mod:`mars_rec.visualization`).

Each command surfaces its CLI options via ``--help``; the function
docstrings below mirror those summaries so pydoc reflects the same
contract.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Tuple

import click

import mars_rec

os.environ["LUIGI_CONFIG_PATH"] = os.path.join(os.path.split(mars_rec.__file__)[0], "luigi.cfg")


def _run_luigi(luigi_args) -> None:
    """Dispatch ``luigi`` in a subprocess and exit with its return code.

    Uses ``sys.executable -m luigi`` (not whatever ``luigi`` is first on
    ``PATH``) so the task runs in the same interpreter/venv as this CLI,
    and *prepends* the cwd to ``PYTHONPATH`` instead of overwriting it so
    user-provided paths (e.g. ``PYTHONPATH=src:.``) keep working. Args are
    passed as a list, so no shell quoting is involved.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([os.getcwd()] + ([existing] if existing else []))
    cmd = [sys.executable, "-m", "luigi", "--module", *luigi_args, "--local-scheduler"]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _invalidate_supervised_cache(
    args: Tuple[str],
    module_path: str = "mars_rec.simulation.training",
    class_name: str = "SupervisedModelTraining",
):
    """Build the dispatched training task from CLI args and delete its cached outputs."""
    import importlib
    import sys

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    task_cls = getattr(module, class_name)

    params = {}
    it = iter(args)
    for arg in it:
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            try:
                value = next(it)
                if value.startswith("--"):
                    params[key] = "True"
                    key = value.lstrip("-").replace("-", "_")
                    continue
                params[key] = value
            except StopIteration:
                params[key] = "True"

    task = task_cls.from_str_params(params)
    task.invalidate_cache()
    click.echo("Cache invalidated. Re-running entire pipeline...")


def _get_results_dir():
    from mars_rec.utils.files import OUTPUT_PATH

    return os.path.join(OUTPUT_PATH, "models", "SupervisedModelTraining", "results")


def _load_run(run_dir):
    """Load metadata for a single training run."""
    from mars_rec.utils.files import get_history_path, get_metadata_path, get_params_path

    params_path = get_params_path(run_dir)
    history_path = get_history_path(run_dir)

    if not os.path.exists(params_path):
        return None

    with open(params_path) as f:
        params = json.load(f)

    # Get timestamp from params.json modification time
    mtime = os.path.getmtime(params_path)
    timestamp = datetime.fromtimestamp(mtime)

    # Load best metrics from last epoch of history
    best_metrics = {}
    if os.path.exists(history_path):
        import pandas as pd

        history = pd.read_csv(history_path)
        if len(history) > 0:
            last = history.iloc[-1]
            for col in history.columns:
                if col != "epoch":
                    best_metrics[col] = last[col]
            best_metrics["total_epochs"] = int(last["epoch"]) + 1

    # Load ranking metrics from metadata.json (computed post-training)
    metadata_path = get_metadata_path(run_dir)
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        ranking = metadata.get("ranking_metrics", {})
        for k, v in ranking.items():
            best_metrics[k] = v
        roc_auc = metadata.get("roc_auc_metrics", {})
        for k, v in roc_auc.items():
            best_metrics[k] = v

    return {
        "task_id": os.path.basename(run_dir),
        "path": run_dir,
        "timestamp": timestamp,
        "params": params,
        "best_metrics": best_metrics,
    }


def _load_all_runs(project_filter=None):
    """Load all training runs, optionally filtered by project."""
    results_dir = _get_results_dir()
    if not os.path.isdir(results_dir):
        return []

    runs = []
    for name in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        run = _load_run(run_dir)
        if run is None:
            continue
        if project_filter and project_filter not in run["params"].get("project", ""):
            continue
        runs.append(run)

    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


@click.group()
def cli():
    """``mars-rec`` — supervised training framework for recommender systems."""


@cli.group()
def run():
    """Run training tasks."""
    pass


@run.command(
    context_settings=dict(
        ignore_unknown_options=True,
    ),
    add_help_option=False,
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Force re-execution of the entire pipeline, ignoring cached results.",
)
@click.option(
    "--task-class",
    default="mars_rec.simulation.training.SupervisedModelTraining",
    help=(
        "Fully-qualified path of the Luigi training task to dispatch "
        "(e.g. a `SupervisedModelTraining` subclass in your project). "
        "Default is `mars_rec.simulation.training.SupervisedModelTraining`."
    ),
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def supervised(no_cache: bool, task_class: str, args: Tuple[str]):
    """Run a supervised training task (default ``SupervisedModelTraining``).

    Dispatches the configured task through Luigi with ``--local-scheduler``.
    Extra positional arguments are forwarded verbatim as Luigi parameters
    (e.g. ``--project examples.movielens.config.movielens
    --recommender-module-class ... --epochs 5``).
    """
    if "." not in task_class:
        raise click.BadParameter(
            f"--task-class must be a fully-qualified Python path (`module.submodule.ClassName`), got {task_class!r}"
        )
    module_path, class_name = task_class.rsplit(".", 1)
    if no_cache:
        _invalidate_supervised_cache(args, module_path=module_path, class_name=class_name)
    _run_luigi([module_path, class_name, *args])


@run.command(
    context_settings=dict(
        ignore_unknown_options=True,
    ),
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def data(args: Tuple[str]):
    """Run an arbitrary Luigi data-preparation task by ``module Task``.

    Useful to materialise the train/val/test CSVs without launching a
    training (e.g. ``mars-rec run data examples.movielens.data
    PrepareMovieLensDataFrame``).
    """
    _run_luigi(list(args))


@cli.command("list")
@click.option("--project", default=None, help="Filter by project name (substring match).")
@click.option("--limit", default=20, help="Max number of runs to show.")
def list_runs(project, limit):
    """List all executed training runs."""
    runs = _load_all_runs(project_filter=project)

    if not runs:
        click.echo("No training runs found.")
        return

    runs = runs[:limit]

    # Table header
    click.echo(f"{'#':<4} {'Date':<18} {'Project':<40} {'Model':<40} {'Epochs':<8} {'Task ID'}")
    click.echo("-" * 160)

    for i, run in enumerate(runs, 1):
        p = run["params"]
        project_name = p.get("project", "?").split(".")[-1]
        model_name = p.get("recommender_module_class", "?").split(".")[-1]
        date_str = run["timestamp"].strftime("%Y-%m-%d %H:%M")
        epochs = run["best_metrics"].get("total_epochs", p.get("epochs", "?"))
        task_id = run["task_id"]

        click.echo(f"{i:<4} {date_str:<18} {project_name:<40} {model_name:<40} {epochs:<8} {task_id}")

    click.echo(f"\nTotal: {len(runs)} run(s)")


@cli.command("results")
@click.option("--project", default=None, help="Filter by project name (substring match).")
@click.option(
    "--metric",
    multiple=True,
    help="Metrics to display (e.g. --metric val_loss --metric val_binary_accuracy). Shows all if omitted.",
)
@click.option("--sort-by", default=None, help="Sort results by this metric (ascending).")
@click.option("--limit", default=20, help="Max number of runs to show.")
@click.option(
    "--params", multiple=True, help="Extra params to display as columns (e.g. --params learning_rate --params epochs)."
)
def results(project, metric, sort_by, limit, params):
    """Compare results across training runs."""
    import pandas as pd

    runs = _load_all_runs(project_filter=project)

    if not runs:
        click.echo("No training runs found.")
        return

    # Build comparison table
    rows = []
    for run in runs:
        p = run["params"]
        row = {
            "date": run["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "project": p.get("project", "?").split(".")[-1],
            "model": p.get("recommender_module_class", "?").split(".")[-1],
            "epochs": run["best_metrics"].get("total_epochs", p.get("epochs", "?")),
            "lr": p.get("learning_rate"),
            "batch_size": p.get("batch_size"),
            "loss_fn": p.get("loss_function"),
            "optimizer": p.get("optimizer"),
        }

        # Add extra params (also search inside recommender_extra_params)
        extra = p.get("recommender_extra_params", {})
        for param in params:
            if param in p:
                row[param] = p[param]
            elif param in extra:
                row[param] = extra[param]
            else:
                row[param] = "?"

        # Add metrics
        for k, v in run["best_metrics"].items():
            if k == "total_epochs":
                continue
            if metric and k not in metric:
                continue
            if isinstance(v, float):
                row[k] = round(v, 6)
            else:
                row[k] = v

        row["task_id"] = run["task_id"]
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)

    df = df.head(limit)
    df.index = range(1, len(df) + 1)
    df.index.name = "#"

    # Separate into config cols and metric cols for display
    config_cols = ["date", "project", "model", "epochs", "lr", "batch_size", "loss_fn", "optimizer"]
    config_cols += list(params)
    metric_cols = [c for c in df.columns if c not in config_cols and c != "task_id"]

    display_cols = [c for c in config_cols if c in df.columns] + metric_cols + ["task_id"]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)

    click.echo(df[display_cols].to_string())
    click.echo(f"\nTotal: {len(df)} run(s)")


@cli.command("predict")
@click.option("--model-dir", required=False, default=None, help="Full path to the model task directory.")
@click.option(
    "--task-id", required=False, default=None, help="Task ID (looks up inside the standard results directory)."
)
@click.option("--data", required=True, help="Path to input data file (parquet or CSV).")
@click.option("--output", default="result.json", help="Output file path.")
@click.option("--batch-size", default=128, type=int, help="Batch size for inference.")
@click.option("--top-k", default=10, type=int, help="Number of top items per user.")
@click.option("--device", default="cpu", help="Device for inference (cpu, cuda, cuda:0, ...).")
def predict(model_dir, task_id, data, output, batch_size, top_k, device):
    """Run inference on a trained model and produce top-k predictions."""
    import sys

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if model_dir is None and task_id is None:
        click.echo("Error: Provide either --model-dir or --task-id.", err=True)
        raise SystemExit(1)

    if model_dir is None:
        model_dir = os.path.join(_get_results_dir(), task_id)

    if not os.path.isdir(model_dir):
        click.echo(f"Error: Directory not found: {model_dir}", err=True)
        raise SystemExit(1)

    from mars_rec.inference import run_inference

    run_inference(
        task_dir=model_dir,
        data_path=data,
        output_path=output,
        batch_size=batch_size,
        k=top_k,
        device=device,
    )


@cli.command("evaluate")
@click.option("--model-dir", required=False, default=None, help="Full path to the model task directory.")
@click.option(
    "--task-id", required=False, default=None, help="Task ID (looks up inside the standard results directory)."
)
@click.option(
    "--data",
    default=None,
    help="Path to external data file (CSV or parquet) for evaluation. Must contain the target column. If omitted, uses original training splits.",
)
@click.option(
    "--metric",
    multiple=True,
    help="Override validation metrics (e.g. --metric loss --metric f1_score --metric binary_accuracy). Uses original if omitted.",
)
@click.option(
    "--ranking-metrics/--no-ranking-metrics",
    default=None,
    help="Enable/disable ranking metrics (NDCG, HitRate, MRR, MAP).",
)
@click.option("--sample-size-eval", default=None, type=int, help="Override test-set sample size for evaluation.")
@click.option("--batch-size", default=None, type=int, help="Override batch size for evaluation.")
@click.option("--device", default=None, help="Override device (cpu, cuda, mps).")
@click.option(
    "--split",
    type=click.Choice(["val", "test", "all"]),
    default="all",
    help="Which split to evaluate: val, test, or all (default). Ignored when --data is provided.",
)
@click.option(
    "--output",
    default=None,
    help="Directory to save evaluation artifacts (plots, predictions, eval_results.json). Defaults to the model task directory.",
)
@click.option(
    "--prefilter/--no-prefilter",
    default=False,
    help="Apply the project's CandidateProvider (eligibility filter) when generating ranking-metrics candidates. Default off: ranking metrics are computed against the legacy random-sample candidate pool. Configs without a provider declared are unaffected.",
)
def evaluate_cmd(
    model_dir, task_id, data, metric, ranking_metrics, sample_size_eval, batch_size, device, split, output, prefilter
):
    """Re-run evaluation on training splits or on a new external dataset.

    \b
    Flow 1 (original data):
      mars-rec evaluate --task-id <ID> --ranking-metrics --metric f1_score

    \b
    Flow 2 (external data):
      mars-rec evaluate --task-id <ID> --data new_test.parquet --ranking-metrics
    """
    import sys

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if model_dir is None and task_id is None:
        click.echo("Error: Provide either --model-dir or --task-id.", err=True)
        raise SystemExit(1)

    if model_dir is None:
        model_dir = os.path.join(_get_results_dir(), task_id)

    if not os.path.isdir(model_dir):
        click.echo(f"Error: Directory not found: {model_dir}", err=True)
        raise SystemExit(1)

    if data is not None and not os.path.isfile(data):
        click.echo(f"Error: Data file not found: {data}", err=True)
        raise SystemExit(1)

    from mars_rec.simulation.training import run_evaluate

    metrics_list = list(metric) if metric else None

    results = run_evaluate(
        task_dir=model_dir,
        metrics=metrics_list,
        ranking_metrics=ranking_metrics,
        sample_size_eval=sample_size_eval,
        batch_size=batch_size,
        device=device,
        eval_split=split,
        data_path=data,
        output_dir=output,
        enable_prefilter=prefilter,
    )

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION SUMMARY")
    click.echo("=" * 60)

    if "val_metrics" in results and results["val_metrics"]:
        click.echo("\nPoint-wise metrics:")
        for k, v in results["val_metrics"].items():
            click.echo(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if results.get("ranking_metrics"):
        click.echo("\nRanking metrics:")
        for k, v in results["ranking_metrics"].items():
            click.echo(f"  {k}: {v:.6f}")

    if results.get("roc_auc_metrics"):
        click.echo("\nROC-AUC metrics:")
        for k, v in results["roc_auc_metrics"].items():
            click.echo(f"  {k}: {v:.6f}")


@cli.command("visualize")
@click.option("--model-dir", required=False, default=None, help="Full path to the model task directory.")
@click.option(
    "--task-id", required=False, default=None, help="Task ID (looks up inside the standard results directory)."
)
@click.option(
    "--mode",
    type=click.Choice(["svg", "summary"]),
    default="svg",
    help="svg: writes architecture.svg (default).  summary: prints a text table.",
)
@click.option(
    "--output", "-o", default=None, help="Output .svg path (svg mode only). Defaults to <task_dir>/architecture.svg."
)
@click.option(
    "--trace/--no-trace", default=True, help="Trace tensor shapes via a forward pass (slightly slower, more accurate)."
)
@click.option("--device", default="cpu", help="Device for model instantiation.")
def visualize(model_dir, task_id, mode, output, trace, device):
    """Generate the architecture diagram for a trained model.

    \b
    Modes:
      svg     — Hand-rolled SVG flow diagram with feature pills, role
                colors, and tensor shapes (default).  No external deps.
      summary — Enhanced text table with feature groups + layer list.
    """
    import sys

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if model_dir is None and task_id is None:
        click.echo("Error: Provide either --model-dir or --task-id.", err=True)
        raise SystemExit(1)

    if model_dir is None:
        model_dir = os.path.join(_get_results_dir(), task_id)

    if not os.path.isdir(model_dir):
        click.echo(f"Error: Directory not found: {model_dir}", err=True)
        raise SystemExit(1)

    from mars_rec.visualization import (
        generate_architecture_diagram,
        load_model_from_task,
        print_text_summary,
    )

    click.echo(f"Loading model from: {model_dir}")
    model, project_config, index_mapping, params = load_model_from_task(
        model_dir,
        device=device,
    )
    model_cls = params.get("recommender_module_class", "?").split(".")[-1]
    click.echo(f"Model: {model_cls} | Input columns: {len(project_config.input_columns)}")

    if mode == "summary":
        print_text_summary(model, project_config, params)
        return

    if output is None:
        from mars_rec.utils.files import get_architecture_path

        output = get_architecture_path(model_dir)

    rendered = generate_architecture_diagram(
        model,
        project_config,
        params,
        output,
        index_mapping=index_mapping,
        trace=trace,
        device=device,
    )
    click.echo(f"SVG diagram saved to: {rendered}" + ("  (with forward-pass shape trace)" if trace else ""))


if __name__ == "__main__":
    cli()
