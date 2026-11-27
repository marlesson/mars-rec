# Models

Every model inherits from [`RecommenderModule`][mars_rec.model.abstract.RecommenderModule], which
is an `nn.Module` constructed with the project config and the index mapping:

```python
import torch
import torch.nn as nn
from mars_rec.model.abstract import RecommenderModule


class MyModel(RecommenderModule):
    def __init__(self, project_config, index_mapping, n_factors: int = 32):
        super().__init__(project_config, index_mapping)
        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.item_emb = nn.Embedding(self._n_items, n_factors)
        self.fc = nn.Linear(n_factors * 2 + 1, 1)  # +1 for "price"

    def forward(self, user_ids, item_ids, price):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i, price.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(-1)  # [batch_size]
```

Keyword arguments after `index_mapping` (here `n_factors`) come from `--recommender-extra-params`
as JSON: `--recommender-extra-params '{"n_factors": 64}'`.

## What the base class gives you

| Attribute | Description |
|---|---|
| `self._n_users` / `self._n_items` | Vocabulary sizes, **including the reserved slots**. Use them directly as `num_embeddings` |
| `self._index_mapping` | `{column: {raw_value: index}}` |
| `self._project_config` | The project's `ProjectConfig` |
| `recommendation_score(*args)` | Score used for ranking. Defaults to `forward()`; override it for multi-task models |

## `forward()` argument order

`forward()` receives **one positional tensor per input column**, in the order of
[`ProjectConfig.input_columns`][mars_rec.meta_config.ProjectConfig.input_columns]:

1. `user_column` (unless `user_is_input=False`);
2. `item_column` (unless `item_is_input=False`);
3. every entry of `other_input_columns`;
4. every entry of `metadata_columns` (only when the item is an input).

```python
# ProjectConfig:
#   user_column         = Column("user_id", IOType.INDEXABLE)
#   item_column         = Column("item_id", IOType.INDEXABLE)
#   other_input_columns = [Column("price", IOType.NUMBER)]
def forward(self, user_ids, item_ids, price): ...
```

Models that must handle a variable number of columns can accept `*features`.

## `forward()` output shape

The expected output depends on the loss function:

| Loss | Output shape | Notes |
|---|---|---|
| `bce`, `implicit_feedback_bce` | `[batch_size]` | Probabilities: apply `sigmoid` in the model |
| `bce_logits`, `bce_logits_smooth`, `focal_loss` | `[batch_size]` | Raw logits (metrics receive `sigmoid(score)`) |
| `mse`, `dummy` | `[batch_size]` | One score per row |
| `bpr`, `ranknet`, `lambdarank` | `[batch_size]` | Raw logits. Pairwise / listwise ranking on per-row scores |
| `ce`, `nll` | `[batch_size, n_items]` | Scores over the item vocabulary (next-item models) |
| `infonce`, `cosine_similarity`, `hardneg_infonce` | embeddings | Use with `EmbeddingRecommenderLightningModule` |

!!! warning "`RuntimeError: target size != input size`"
    This is the most common error and is almost always a shape mismatch. If your last layer
    returns `[batch_size, 1]`, add `.squeeze(-1)`. For `ce`, return the `[batch_size, n_items]`
    logits **without** squeezing.

## Sequence models

For next-item prediction:

1. Declare the history as `Column(..., IOType.INDEXABLE_ARRAY, same_index_as=<item column>)`.
2. Accept `*features` in `forward()` and embed the history with the item embedding.
3. Either score the target item (`[batch_size]`, BCE-style), or return logits over the vocabulary
   (`[batch_size, n_items]`) and train with `--loss-function ce`.

See [Last.fm (sequential)](../tutorials/lastfm.md) and
[Online Retail (session-based)](../tutorials/online-retail.md).
