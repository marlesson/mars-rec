#!/usr/bin/env python3
"""
Train GRU4Rec on UCI Online Retail (session-based next-item).

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/online_retail/run_training.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.online_retail.config.online_retail",
        recommender_module_class="examples.online_retail.model.OnlineRetailGRU4Rec",
        recommender_extra_params={
            "n_factors": 64,
            "hidden_size": 64,
            "dropout": 0.2,
        },
        epochs=3,
        batch_size=512,
        learning_rate=0.001,
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

    success = luigi.build([task], local_scheduler=True, log_level="INFO")

    if success:
        print("\n" + "=" * 60)
        print("Online Retail GRU4Rec training completed!")
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
