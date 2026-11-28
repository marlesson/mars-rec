"""ROC-AUC is not defined for embedding losses: ``forward()`` returns ``[N, D]``
query embeddings, which the multiclass branch used to read as class logits
(an ``IndexError``, logged and swallowed)."""

import types

from mars_rec.simulation.training import SupervisedModelTraining


def test_roc_auc_skipped_for_embedding_losses():
    ns = types.SimpleNamespace(loss_function="infonce")
    assert SupervisedModelTraining._compute_roc_auc(ns) == {}
