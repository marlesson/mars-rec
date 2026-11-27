#!/usr/bin/env python3
"""
Train the SASRec-lite next-item model on Last.fm HetRec 2K.

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/lastfm/run_training.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining
from examples.lastfm.config import LASTFM_HISTORY_LEN

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.lastfm.config.lastfm",
        recommender_module_class="examples.lastfm.model.LastFMSASRec",
        recommender_extra_params={
            "n_factors": 64,
            "n_heads": 2,
            "ff_dim": 128,
            "dropout": 0.2,
            "history_len": LASTFM_HISTORY_LEN,
        },
        epochs=5,
        batch_size=256,
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
        print("Last.fm SASRec-lite training completed!")
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
