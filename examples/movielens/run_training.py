#!/usr/bin/env python3
"""
Train a MovieLens 1M recommendation model using mars-rec.

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/movielens/run_training.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.movielens.config.movielens",
        recommender_module_class="examples.movielens.model.MovieLensNCF",
        recommender_extra_params={
            "n_factors": 32,
            "hidden_size": 128,
            "n_layers": 3,
            "dropout": 0.2,
        },
        epochs=10,
        batch_size=512,
        learning_rate=0.001,
        loss_function="bce_logits",
        negative_proportion=0.5,
        test_size=0.1,
        val_size=0.1,
        dataset_split_method="time",
        test_split_type="time",
        device="cpu",
        early_stopping_patience=3,
        metrics=["loss", "binary_accuracy"],
        ranking_metrics=True,
    )

    success = luigi.build([task], local_scheduler=True, log_level="INFO")

    if success:
        print("\n" + "=" * 60)
        print("MovieLens 1M training completed successfully!")
        print(f"Output dir: {task.output().path}")
        print("=" * 60)

        import pandas as pd
        from mars_rec.utils.files import get_history_path

        history_path = get_history_path(task.output().path)
        if os.path.exists(history_path):
            history = pd.read_csv(history_path)
            print("\nTraining History:")
            print(history.to_string(index=False))
    else:
        print("\nTraining FAILED!")
        sys.exit(1)
