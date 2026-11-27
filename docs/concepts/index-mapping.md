# Index mapping

Columns of type `INDEXABLE` and `INDEXABLE_ARRAY` are stored as raw values (strings, ids) on disk
and converted to integers before they reach the model. `SupervisedModelTraining` builds the
mapping from the data and saves it as `index_mapping.pkl`, which is reused by `predict` and
`evaluate`.

## Reserved slots

The first three indices of every mapping are reserved:

| Index | Meaning |
|---|---|
| `0` | **Unknown**: any value not seen during training (the mapping is a `defaultdict`) |
| `1` | `None` / `NaN` |
| `2` | `-1` / `"-1"`: the pad token used by sequence columns |

**Real values start at `3`.** `RecommenderModule._n_users` and `_n_items` already account for the
reserved slots (`max(index) + 1`), so pass them straight to `nn.Embedding`:

```python
self.item_emb = nn.Embedding(self._n_items, n_factors, padding_idx=2)  # 2 = pad
```

## Shared vocabularies

A column declared with `same_index_as="item_id"` reuses the item column's mapping. This is how a
history of items and the target item end up in the same embedding space.

## Reusing a mapping across runs

To keep ids stable between experiments (for example, to compare or ensemble models), point a new
run at the **task directory** of a previous run. Its `index_mapping.pkl` is loaded instead of
building a new one:

```bash
mars-rec run supervised ... \
  --load-index-mapping-path output/models/SupervisedModelTraining/results/<task_id>
```

!!! warning "`KeyError` in index_mapping"
    The column names in `ProjectConfig` must match the CSV produced by the data-prep task exactly.
