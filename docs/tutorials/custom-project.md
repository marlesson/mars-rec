# Your first project

This tutorial builds a project from scratch: a purchase-prediction model over
`(user_id, item_id, price)` interactions.

!!! note "Where your code lives"
    Application code (your data tasks, configs and models) belongs in **your own repository**,
    which depends on `mars-rec` as a library. Only the framework (`src/mars_rec/`) and the
    reference `examples/` live in the MARS-Rec repository.

```text
my_project/
├── __init__.py
├── data.py      # Luigi tasks: raw data → train/val/test CSVs
├── config.py    # ProjectConfig
└── model.py     # RecommenderModule subclass
```

## 1. Prepare the data

Subclass [`BasePrepareDataFrames`][mars_rec.data.task.BasePrepareDataFrames]. It needs an
upstream task that produces the raw file, a directory for the outputs, and the path to read from.

```python title="my_project/data.py"
import luigi
import pandas as pd
from mars_rec.data.task import BasePrepareDataFrames


class RawInteractions(luigi.ExternalTask):
    """An existing CSV with columns: user_id, item_id, price, purchased, timestamp."""

    def output(self):
        return luigi.LocalTarget("data/interactions.csv")


class PrepareMyDataFrame(BasePrepareDataFrames):
    def requires(self):
        return RawInteractions()

    @property
    def dataset_dir(self) -> str:
        return "output/my_project/dataset"

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def timestamp_property(self) -> str:
        return "timestamp"  # required for time-based splits

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        df["price"] = df["price"].astype(float)
        return df
```

Optional hooks:

- `metadata_data_frame_path` returns the path to a per-item metadata CSV (for `metadata_columns`);
- `stratification_property` sets the column to stratify on for random test splits;
- `create_available_items(df)` lets you override or skip the creation of the `available_items` slate column.

## 2. Declare the `ProjectConfig`

```python title="my_project/config.py"
from mars_rec.data.dataset import InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig
from my_project.data import PrepareMyDataFrame

my_project = ProjectConfig(
    base_dir="output/my_project",
    prepare_data_frames_task=PrepareMyDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[Column("price", IOType.NUMBER)],
    output_column=Column("purchased", IOType.NUMBER),
    available_items_column_name=None,
)
```

## 3. Implement the model

`forward()` receives one tensor per input column, in the order `user_id, item_id, price`, and
returns one score per row.

```python title="my_project/model.py"
import torch
import torch.nn as nn
from mars_rec.model.abstract import RecommenderModule


class MyModel(RecommenderModule):
    def __init__(self, project_config, index_mapping, n_factors: int = 32):
        super().__init__(project_config, index_mapping)
        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.item_emb = nn.Embedding(self._n_items, n_factors)
        self.fc = nn.Linear(n_factors * 2 + 1, 1)

    def forward(self, user_ids, item_ids, price):
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids), price.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(-1)  # [batch_size] logits → use bce_logits
```

## 4. Train

```bash
PYTHONPATH=. mars-rec run supervised \
  --project my_project.config.my_project \
  --recommender-module-class my_project.model.MyModel \
  --recommender-extra-params '{"n_factors": 32}' \
  --loss-function bce_logits \
  --negative-proportion 0.5 \
  --epochs 20 \
  --ranking-metrics
```

## 5. Iterate

```bash
mars-rec results --project my_project --sort-by val_loss --params n_factors
mars-rec visualize --task-id <TASK_ID>
```

Next steps:

- [Training parameters](../guides/training-parameters.md) lists every option;
- [Extending the framework](../guides/extending.md) shows how to add losses, metrics and optimizers;
- [Troubleshooting](../guides/troubleshooting.md) covers common errors.
