#!/usr/bin/env python3
"""
Train the two-tower retrieval model on MovieLens 100K with InfoNCE.

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/movielens100k_retrieval/run_training.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining

# InfoNCE temperature. The model needs the same value for its logQ correction.
TEMPERATURE = 0.1

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.movielens100k_retrieval.config.movielens100k_retrieval",
        recommender_module_class="examples.movielens100k_retrieval.model.ML100KTwoTowerRetrieval",
        recommender_extra_params={
            "n_factors": 64,
            "hidden_size": 128,
            "dropout": 0.1,
            "logq_correction": True,
            "temperature": TEMPERATURE,
        },
        epochs=20,
        # Larger batches = more in-batch negatives per query for InfoNCE.
        batch_size=512,
        learning_rate=0.001,
        loss_function="infonce",
        loss_function_params={"temperature": TEMPERATURE},
        test_size=0.1,
        val_size=0.1,
        dataset_split_method="time",
        test_split_type="time",
        device="cpu",
        early_stopping_patience=3,
        metrics=["loss"],
        ranking_metrics=True,
    )

    success = luigi.build([task], local_scheduler=True, log_level="INFO")

    if success:
        print("\n" + "=" * 60)
        print("MovieLens 100K two-tower retrieval training completed!")
        print(f"Output dir: {task.task_dir}")
        print("=" * 60)

        import pandas as pd
        from mars_rec.utils.files import get_history_path

        history_path = get_history_path(task.task_dir)
        if os.path.exists(history_path):
            history = pd.read_csv(history_path)
            print("\nTraining History:")
            print(history.to_string(index=False))
    else:
        print("\nTraining FAILED!")
        sys.exit(1)
