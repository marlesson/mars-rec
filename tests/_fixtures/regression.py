"""Shared helpers for the dependency-regression fixtures.

Used both by ``scripts/make_regression_fixtures.py`` (to generate the
reference files) and by ``tests/test_dependency_regression.py`` (to check
them), so both sides build the training task and score inputs the same way.
"""

from __future__ import annotations

from importlib import metadata
from typing import Dict, List

# Fixed rows scored against the legacy checkpoint. They include an unseen
# user and an unseen item to also cover the OOV (index 0) path.
SCORING_INPUT: Dict[str, list] = {
    "user_id": ["u0", "u1", "u2", "u3", "u4", "u_unseen", "u2"],
    "item_id": ["i0", "i5", "i11", "i3", "i7", "i1", "i_unseen"],
    "label": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
}

TRAINING_PARAMS = dict(
    project="tests._fixtures.dummy_project.dummy_project_config",
    recommender_module_class="mars_rec.model.base_model.LogisticRegression",
    recommender_extra_params={"n_factors": 4},
    epochs=3,
    batch_size=8,
    loss_function="mse",
    optimizer="adam",
    learning_rate=1e-2,
    metrics=["loss"],
    ranking_metrics=False,
    device="cpu",
    generator_workers=0,
    seed=42,
)

_TRACKED_PACKAGES = ("torch", "pytorch-lightning", "numpy", "pandas", "scikit-learn", "scipy", "luigi")


def build_training_task(**overrides):
    from mars_rec.simulation.training import SupervisedModelTraining

    params = dict(TRAINING_PARAMS)
    params.update(overrides)
    return SupervisedModelTraining(**params)


def score_fixed_input(task_dir: str) -> List[float]:
    """Load the artifacts in ``task_dir`` and score ``SCORING_INPUT``.

    Goes through the same public helpers ``mars-rec predict`` / ``evaluate``
    use: ``load_trained_model`` + ``preprocess_raw_data``.
    """
    import pandas as pd

    from mars_rec.inference import _collect_logits_and_targets, load_trained_model, preprocess_raw_data

    model, project_config, index_mapping, _ = load_trained_model(task_dir, device="cpu")
    _, dataset = preprocess_raw_data(pd.DataFrame(SCORING_INPUT), project_config, index_mapping)
    logits, _ = _collect_logits_and_targets(model, dataset, batch_size=4, device="cpu")
    return [float(x) for x in logits.reshape(-1).tolist()]


def stack_versions() -> Dict[str, str]:
    import platform

    versions = {"python": platform.python_version()}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:  # pragma: no cover
            versions[pkg] = "not-installed"
    return versions
