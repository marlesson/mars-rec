# Ranking evaluation

With `ranking_metrics=True` (or `mars-rec evaluate --ranking-metrics`), the test split is evaluated as
a **top-k recommendation** task. Training, `mars-rec evaluate` and
[`inference.run_evaluate`][mars_rec.inference.run_evaluate] all share one implementation,
[`mars_rec.evaluation`](../reference/api/evaluation.md), so their numbers are comparable.

## Protocol

1. **Ground truth.** Only *positive* test rows count: rows whose `output_column` is `> 0` (a
   click, a `label` of 1, a positive rating). Negative rows, such as ratings below the positive
   threshold, are never treated as relevant. When the output column *is* the item (next-item / CE
   models), every row is positive.
2. **Queries.** Positive rows that give the model the same input apart from the item form one
   **query**. The key is the user column plus every other input column, except the item features that
   `candidate_feature_lookup` rewrites per candidate. A collaborative-filtering user with five
   held-out items is one query with five relevant items. A sequential model gets one query per event,
   because each row carries its own history.
3. **Candidates.** Each query is ranked against the **whole catalog**. When the project declares a
   real per-row slate (`available_items_column_name`), the slate is ranked instead. When
   `--prefilter` is set, the project's
   [`CandidateProvider`][mars_rec.filtering.CandidateProvider] pool is ranked. There is no negative
   sampling: metrics computed on a sampled pool (for example 100 random items plus the positive) can
   rank models differently from the exact metrics ([Krichene & Rendle, KDD 2020][sampled]).
4. **Seen items.** Items the user already interacted with in train or validation are removed from
   the candidates. A held-out item that the user also saw in training is removed from the relevant
   set, since it can no longer be recommended. A query left without relevant items is dropped.
5. **Ties** are broken uniformly at random with the run's `seed`. A model that outputs constant scores
   therefore cannot gain hits from the order in which candidates are listed, and re-evaluating a run
   gives identical numbers.
6. **Aggregation.** Each metric is computed per query, averaged within each user, then averaged across
   users. Heavy users do not dominate the result.

`sample_size_eval` caps the number of queries, drawn at random with the run's `seed`. Full-catalog
scoring costs `queries × catalog` forward passes for point-wise models, and the log warns when that
exceeds 50M pairs. The example projects use `sample_size_eval=2000`.

## Metrics

All metrics are reported at `k ∈ {5, 10, 20}`. For a query, `R` is its relevant set and `hits` are
the relevant items in its top-k.

| Metric | Definition |
|---|---|
| `hit_rate@k` | 1 when at least one relevant item is in the top-k |
| `precision@k` | `|hits| / k` |
| `recall@k` | `|hits| / |R|` |
| `ndcg@k` | DCG with binary relevance divided by the ideal DCG of `min(|R|, k)` hits |
| `mrr@k` | `1 / rank` of the first relevant item (0 when there is none in the top-k) |
| `map@k` | Average precision truncated at k: `Σ precision@i · rel_i / min(|R|, k)` |
| `coverage@k` | Share of the catalog that appears in at least one top-k list |
| `gini@k` | Gini index of how often each catalog item is recommended (0 = even exposure, 1 = a single item gets every slot) |
| `avg_popularity@k` | Mean training popularity of the recommended items, measured as users with a positive interaction (ARP) |
| `novelty@k` | Mean self-information `-log2(popularity / n_users)` of the recommended items. Higher means less popular items |

With one relevant item per query, `recall@k == hit_rate@k` and `map@k == mrr@k`. These metrics
differ only when queries have several relevant items.

The accuracy metrics measure whether the model finds what the user consumed. The beyond-accuracy
metrics measure whether it recommends the same few popular items to everyone, which accuracy alone
hides ([Kaminskas & Bridge, 2016][beyond]).

## Popularity baseline

The evaluation also ranks the same queries and candidates by training popularity, with the same
seen-item exclusion. A model that does not beat this most-popular baseline has not learned
personalisation. The training log prints the model and baseline side by side:

```text
  metric                    model popularity
  ndcg@10                  0.3126     0.3976
  coverage@10              0.4569     0.4478
```

## Where the results go

| File | Keys |
|---|---|
| `metadata.json`, `eval/<name>/metrics.json` | `ranking_metrics` (model), `ranking_baseline` (most popular), `ranking_protocol`, `roc_auc_metrics` |
| `eval/<name>/predictions.csv` | One row per query: user, `relevant_items`, `sorted_actions` (top-20 items), `action_scores`, `n_candidates` |

`ranking_protocol` records how the numbers were produced. It holds the candidate source
(`full_catalog`, `available_items` or the provider name), whether seen items were excluded, the
number of test rows, positive rows, queries, users and relevant items, how many relevant items were
dropped as already seen, the average number of candidates and the catalog size. Compare runs only
when their protocols match.

[`inference.run_evaluate`][mars_rec.inference.run_evaluate] works from a task directory alone, with
no training data. It therefore does not exclude seen items and computes neither the popularity
metrics nor the baseline. Use `mars-rec evaluate` for the full protocol.

## Training-time metrics

The `metrics` list used during training (`ndcg@10`, `top_10_accuracy`, … from
[`METRIC_REGISTRY`][mars_rec.torch.metrics]) is computed per batch on the validation split, for
monitoring and early stopping. It is a different measurement from the offline ranking evaluation
above and is not comparable to it.

[sampled]: https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
[beyond]: https://research.ucc.ie/en/publications/diversity-serendipity-novelty-and-coverage-a-survey-and-empirical/
