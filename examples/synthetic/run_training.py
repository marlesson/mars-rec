#!/usr/bin/env python3
"""
Train matrix factorisation on the synthetic offline dataset, then re-evaluate it.

No download, a few seconds on CPU: the quickest way to see the whole
mars-rec lifecycle (data generation → split → training → ranking metrics →
standalone re-evaluation from the saved artifacts).

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/synthetic/run_training.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.synthetic.config.synthetic",
        recommender_module_class="examples.synthetic.model.SyntheticMF",
        recommender_extra_params={"n_factors": 16},
        epochs=5,
        batch_size=256,
        learning_rate=0.01,
        loss_function="bce_logits",
        negative_proportion=0.5,
        test_size=0.1,
        val_size=0.1,
        dataset_split_method="time",
        test_split_type="time",
        device="cpu",
        early_stopping_patience=2,
        metrics=["loss", "binary_accuracy"],
        ranking_metrics=True,
    )

    if not luigi.build([task], local_scheduler=True, log_level="INFO"):
        print("\nTraining FAILED!")
        sys.exit(1)

    task_dir = task.task_dir
    print("\n" + "=" * 60)
    print("Synthetic MF training completed!")
    print(f"Output dir: {task_dir}")
    print("=" * 60)

    # metadata.json holds the test-set ranking evaluation (ranking_metrics=True):
    # full catalog, items seen in train/val excluded, plus a most-popular
    # baseline scored on the same queries. `mars-rec results` reads this file.
    with open(os.path.join(task_dir, "metadata.json")) as f:
        metadata = json.load(f)
    model_metrics = metadata.get("ranking_metrics", {})
    baseline = metadata.get("ranking_baseline", {})
    print(f"\n{'test metric':<14}{'model':>8}{'most-popular':>14}")
    for name in ("ndcg@10", "recall@10", "hit_rate@10", "coverage@10"):
        if name in model_metrics:
            print(f"{name:<14}{model_metrics[name]:>8.4f}{baseline.get(name, float('nan')):>14.4f}")

    # Re-evaluate later from the saved artifacts alone (params.json +
    # weights.ckpt + index_mapping.pkl) — the Python equivalent of
    # `mars-rec evaluate --task-id <id> --ranking-metrics --split test`.
    from mars_rec.simulation.training import run_evaluate

    results = run_evaluate(task_dir, ranking_metrics=True, eval_split="test")
    print(f"\nRe-evaluated test ndcg@10: {results['ranking_metrics'].get('ndcg@10', float('nan')):.4f}")
