"""Generate the fixtures used by ``tests/test_dependency_regression.py``.

The dependency-regression tests guard against silent breakage when the
pinned stack (``requirements.lock``) is upgraded:

- ``tests/_fixtures/legacy_artifacts/``: a ``params.json`` +
  ``index_mapping.pkl`` + ``weights.ckpt`` trio produced by a real
  ``SupervisedModelTraining`` run, plus ``expected_scores.json`` with the
  model's scores on a fixed input. Newer stacks must still load these
  artifacts and reproduce the scores, which is what happens when a model
  trained on an older image is served or re-evaluated with a newer one.
- ``tests/_fixtures/golden_history.json``: the per-epoch ``history.csv``
  values of a seeded run, so numerical drift in training is visible.

Run it with the stack you want to treat as the reference, from the repo root:

    PYTHONPATH=. python scripts/make_regression_fixtures.py               # both
    PYTHONPATH=. python scripts/make_regression_fixtures.py --golden-only

Only regenerate ``legacy_artifacts`` on purpose. Its whole point is to
represent models produced by an *older* stack. ``golden_history.json`` can
be refreshed when a numerical change is expected and has been reviewed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "_fixtures"
LEGACY_DIR = FIXTURES_DIR / "legacy_artifacts"
GOLDEN_HISTORY_PATH = FIXTURES_DIR / "golden_history.json"

sys.path.insert(0, str(REPO_ROOT))

from tests._fixtures.regression import (  # noqa: E402
    SCORING_INPUT,
    build_training_task,
    score_fixed_input,
    stack_versions,
)
from mars_rec.utils import files  # noqa: E402


def main() -> None:
    import argparse

    import luigi
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="refresh golden_history.json and leave legacy_artifacts untouched",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="mars_rec_fixtures_"))
    os.environ["MARS_TEST_DATASET_DIR"] = str(workdir / "ds")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        luigi.task_register.Register.clear_instance_cache()
        task = build_training_task()
        task.requires().run()
        task.run()
        out = str(Path(task.task_dir).resolve())

        history = pd.read_csv(files.get_history_path(out))
    finally:
        os.chdir(cwd)

    # --- legacy artifacts -------------------------------------------------
    if not args.golden_only:
        if LEGACY_DIR.exists():
            shutil.rmtree(LEGACY_DIR)
        LEGACY_DIR.mkdir(parents=True)
        # The fixture keeps the flat, pre-reorganisation file names on purpose:
        # it is what artifacts trained by older versions look like.
        sources = {
            "params.json": files.get_params_path(out),
            "index_mapping.pkl": files.get_index_mapping_path(out),
            "weights.ckpt": files.find_checkpoint(out),
        }
        for name, src in sources.items():
            shutil.copy2(src, LEGACY_DIR / name)

        scores = score_fixed_input(str(LEGACY_DIR))
        with open(LEGACY_DIR / "expected_scores.json", "w") as f:
            json.dump(
                {"generated_with": stack_versions(), "input": SCORING_INPUT, "scores": scores},
                f,
                indent=2,
            )

    # --- golden history ---------------------------------------------------
    with open(GOLDEN_HISTORY_PATH, "w") as f:
        json.dump(
            {"generated_with": stack_versions(), "history": history.to_dict(orient="list")},
            f,
            indent=2,
        )

    shutil.rmtree(workdir, ignore_errors=True)
    written = [GOLDEN_HISTORY_PATH] if args.golden_only else [LEGACY_DIR, GOLDEN_HISTORY_PATH]
    print("Wrote " + ", ".join(str(p) for p in written))
    print(json.dumps(stack_versions(), indent=2))


if __name__ == "__main__":
    main()
