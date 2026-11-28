"""End-to-end run of ``examples/synthetic`` — the one example that needs no download.

Complements ``test_examples_smoke.py`` (which never touches data): here the
real script runs as a subprocess against a temporary ``OUTPUT_PATH``, through
data generation, the split, training, the test-set ranking evaluation and
``run_evaluate``. A learnable signal is planted in the synthetic data, so the
model must clearly beat the most-popular baseline.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_example_trains_and_ranks(tmp_path):
    env = {**os.environ, "OUTPUT_PATH": str(tmp_path), "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "synthetic" / "run_training.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    (metadata_path,) = glob.glob(
        str(tmp_path / "models" / "SupervisedModelTraining" / "results" / "*" / "metadata.json")
    )
    metadata = json.loads(Path(metadata_path).read_text())
    model, baseline = metadata["ranking_metrics"], metadata["ranking_baseline"]
    # Full-catalog ranking with seen items excluded: the planted user-item
    # signal must clearly beat the most-popular baseline on the same queries.
    assert model["ndcg@10"] > 1.5 * baseline["ndcg@10"], (model, baseline)
    assert metadata["ranking_protocol"]["candidates"] == "full_catalog", metadata["ranking_protocol"]
    assert "Re-evaluated test ndcg@10" in proc.stdout
