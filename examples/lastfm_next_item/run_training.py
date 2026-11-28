#!/usr/bin/env python3
"""
Train the full-softmax SASRec next-item model on Last.fm HetRec 2K.

Reuses the dataset prepared by ``examples/lastfm`` (downloaded on first run).

Usage:
    cd <project-root>
    PYTHONPATH=src:. python examples/lastfm_next_item/run_training.py

Afterwards, top-k recommendations for any CSV with the same columns:
    mars-rec predict --task-id <TASK_ID> --data <test.csv> --top-k 10
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import luigi
from mars_rec.simulation.training import SupervisedModelTraining
from examples.lastfm_next_item.config import LASTFM_HISTORY_LEN

if __name__ == "__main__":
    task = SupervisedModelTraining(
        project="examples.lastfm_next_item.config.lastfm_next_item",
        recommender_module_class="examples.lastfm_next_item.model.LastFMNextItemSASRec",
        recommender_extra_params={
            "n_factors": 64,
            "n_heads": 2,
            "ff_dim": 128,
            "dropout": 0.2,
            "history_len": LASTFM_HISTORY_LEN,
        },
        epochs=30,
        batch_size=256,
        learning_rate=0.002,
        loss_function="ce",
        test_size=0.1,
        val_size=0.1,
        dataset_split_method="time",
        test_split_type="time",
        device="cpu",
        early_stopping_patience=2,
        metrics=["loss"],
        ranking_metrics=True,
        # Rank a 2K-query sample of the test split against the full catalog
        # (~7K artists); it also caps the one-vs-rest ROC-AUC, which scales
        # with the number of classes.
        sample_size_eval=2000,
    )

    success = luigi.build([task], local_scheduler=True, log_level="INFO")

    if success:
        print("\n" + "=" * 60)
        print("Last.fm next-item (CE) training completed!")
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
