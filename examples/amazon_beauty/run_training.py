#!/usr/bin/env python3
"""
Train the two-tower content-based ranker on Amazon Beauty 5-core.

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/amazon_beauty/run_training.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining
from examples.amazon_beauty.config import AMAZON_BEAUTY_TEXT_EMB_DIM

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.amazon_beauty.config.amazon_beauty",
        recommender_module_class="examples.amazon_beauty.model.AmazonBeautyTwoTower",
        recommender_extra_params={
            "n_factors": 64,
            "text_emb_dim": AMAZON_BEAUTY_TEXT_EMB_DIM,
            "hidden_size": 64,
            "dropout": 0.1,
        },
        epochs=5,
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
        print("Amazon Beauty two-tower training completed!")
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
