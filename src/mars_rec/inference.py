"""Standalone inference module for trained mars-rec models.

Loads a trained model from its task directory, processes raw input data
(e.g. a parquet file), runs batch inference, and produces a result.json
with top-k predictions per user.

Usage (programmatic)::

    from mars_rec.inference import run_inference
    run_inference(
        task_dir="output/models/SupervisedModelTraining/results/SupervisedModelTraining____32____xxx",
        data_path="data/test.parquet",
        output_path="result.json",
    )

Usage (CLI)::

    mars-rec predict \\
        --model-dir output/models/.../SupervisedModelTraining____32____xxx \\
        --data data/test.parquet --output result.json
"""

import json
import logging
import os
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mars_rec.data.candidate_features import (
    FN_PARAM as CANDIDATE_FN_PARAM,
    LOOKUP_PARAM as CANDIDATE_LOOKUP_PARAM,
    apply_candidate_features_to_frame,
    resolve_candidate_lookup,
    validate_candidate_feature_fn,
)
from mars_rec.data.dataset import (
    InteractionsDataset,
    preprocess_interactions_data_frame,
)
from mars_rec.filtering import CandidateProvider
from mars_rec.meta_config import ProjectConfig, IOType
from mars_rec.torch.data import NoAutoCollationDataLoader, FasterBatchSampler
from mars_rec.utils.files import (
    find_checkpoint,
    get_params,
    get_index_mapping_path,
)
from mars_rec.utils.index_mapping import is_reserved_key, transform_with_indexing
from mars_rec.utils.logging import banner, setup_logging
from mars_rec.utils.reflection import load_attr

setup_logging()
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a runtime convenience

    def tqdm(it, **_):  # type: ignore[no-redef]
        return it


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_trained_model(
    task_dir: str,
    device: str = "cpu",
) -> Tuple[nn.Module, ProjectConfig, Dict[str, Dict[Any, int]], dict]:
    """Load a trained model and its artefacts from a task directory.

    Returns:
        (model, project_config, index_mapping, params)
    """
    # --- params.json ---
    params = get_params(task_dir)
    project_path = params["project"]
    module_class_path = params["recommender_module_class"]
    extra_params = params.get("recommender_extra_params", {})

    # --- project config ---
    project_config: ProjectConfig = deepcopy(load_attr(project_path, ProjectConfig))

    # --- index mapping ---
    idx_path = get_index_mapping_path(task_dir)
    with open(idx_path, "rb") as f:
        index_mapping: Dict[str, Dict[Any, int]] = pickle.load(f)

    # --- model ---
    module_class = load_attr(module_class_path, nn.Module)
    torch_device = torch.device(device)
    model = module_class(
        project_config=project_config,
        index_mapping=index_mapping,
        **extra_params,
    ).to(torch_device)

    # Load weights from Lightning checkpoint
    ckpt_path = find_checkpoint(task_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {task_dir}")
    checkpoint = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, project_config, index_mapping, params


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------


def preprocess_raw_data(
    df_raw: pd.DataFrame,
    project_config: ProjectConfig,
    index_mapping: Dict[str, Dict[Any, int]],
) -> Tuple[pd.DataFrame, Dataset]:
    """Transform a raw DataFrame into a Dataset ready for inference.

    Applies the same transform pipeline that training uses:
    1. ``transform_data_frame`` from the project's data-preparation task
    2. ``preprocess_interactions_data_frame`` (literal_eval, type conversions)
    3. ``transform_with_indexing`` (categorical → integer indices)
    4. Creates an ``InteractionsDataset``

    Returns:
        (df_processed, dataset)
    """
    df = df_raw.copy()

    # Step 1: project-specific feature transforms (pad arrays, build seqs, etc.)
    # Must run BEFORE adding output column — some tasks have early-exit checks
    task_cls = project_config.prepare_data_frames_task
    task_instance = task_cls.__new__(task_cls)
    df = task_instance.transform_data_frame(df, data_key="predict")

    # Ensure output column exists (may be absent in blind test sets)
    output_col = project_config.output_column.name
    if output_col not in df.columns:
        if project_config.output_column.type == IOType.INDEXABLE:
            # Use item_id as placeholder
            if project_config.item_column.name in df.columns:
                df[output_col] = df[project_config.item_column.name]
            else:
                df[output_col] = 0
        else:
            df[output_col] = 0

    # Fill remaining NaN in array columns with zero-filled lists
    all_columns = (
        [project_config.user_column, project_config.item_column]
        + list(project_config.other_input_columns)
        + [project_config.output_column]
    )
    for col in all_columns:
        if col.type in (IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY):
            mask = df[col.name].isna()
            if mask.any():
                # Infer length from first non-null value
                first_valid = df[col.name].dropna().iloc[0] if (~mask).any() else []
                pad_len = len(first_valid) if hasattr(first_valid, "__len__") else 1
                df.loc[mask, col.name] = [([0] * pad_len)] * mask.sum()
        elif col.type == IOType.FLOAT_ARRAY:
            mask = df[col.name].isna()
            if mask.any():
                first_valid = df[col.name].dropna().iloc[0] if (~mask).any() else []
                pad_len = len(first_valid) if hasattr(first_valid, "__len__") else 1
                df.loc[mask, col.name] = [([0.0] * pad_len)] * mask.sum()

    # Step 2: literal_eval + type conversions
    df = preprocess_interactions_data_frame(df, project_config)

    # Step 3: categorical → index
    transform_with_indexing(df, index_mapping, project_config)

    # Step 4: build dataset
    dataset = InteractionsDataset(
        data_frame=df,
        embeddings_for_metadata=None,
        project_config=project_config,
        index_mapping=index_mapping,
    )

    return df, dataset


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def predict_top_k(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 128,
    k: int = 10,
    device: str = "cpu",
    candidate_provider: Optional[CandidateProvider] = None,
    full_catalog: Optional[np.ndarray] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    exclude_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batch forward pass and extract top-k predictions.

    Assumes the model outputs ``[B, num_classes]`` logits.

    When ``candidate_provider`` is supplied, logits at non-eligible item
    indices are masked to ``-inf`` before the top-k extraction. Caller must
    also pass ``full_catalog`` (the int idx of every catalog item) and
    ``rows`` (per-observation context dicts, length == len(dataset)).

    ``exclude_indices`` are logit indices that are never returned (masked
    to ``-inf``) — pass :func:`reserved_logit_indices` so the reserved
    OOV / NaN / PAD slots cannot take a top-k position.

    Returns:
        (top_k_indices, top_k_scores)  — both shape ``[N, k]``.
    """
    torch_device = torch.device(device)
    batch_sampler = FasterBatchSampler(dataset, batch_size, shuffle=False)
    loader = NoAutoCollationDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
    )

    use_filter = candidate_provider is not None
    if use_filter and (full_catalog is None or rows is None):
        raise ValueError("predict_top_k: candidate_provider requires full_catalog and rows.")

    all_indices: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    row_cursor = 0

    with torch.no_grad():
        for x, _ in loader:
            inputs = x if isinstance(x, (list, tuple)) else [x]
            inputs = [t.to(torch_device) if isinstance(t, torch.Tensor) else t for t in inputs]
            logits = model(*inputs)  # [B, C] or [B]

            if logits.dim() == 2 and logits.size(1) > 1:
                if use_filter:
                    n_classes = logits.size(1)
                    batch_size_actual = logits.size(0)
                    mask = torch.zeros(
                        (batch_size_actual, n_classes),
                        dtype=torch.bool,
                        device=logits.device,
                    )
                    for j in range(batch_size_actual):
                        eligible = candidate_provider.get_candidates(
                            rows[row_cursor + j],
                            full_catalog,
                        )
                        # Clip to vocab range to be safe.
                        elig = np.asarray(eligible, dtype=np.int64)
                        elig = elig[(elig >= 0) & (elig < n_classes)]
                        mask[j, torch.as_tensor(elig, device=logits.device)] = True
                    neg_inf = torch.tensor(
                        float("-inf"),
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    logits = torch.where(mask, logits, neg_inf)
                logits = _mask_indices(logits, exclude_indices)
                actual_k = min(k, logits.size(1))
                scores, indices = logits.topk(actual_k, dim=1)
            else:
                # Binary / 1-class: return the single prediction
                scores = logits.unsqueeze(1) if logits.dim() == 1 else logits
                indices = torch.zeros_like(scores, dtype=torch.long)

            all_indices.append(indices.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            row_cursor += int(logits.size(0)) if logits.dim() >= 1 else 0

    return np.concatenate(all_indices, axis=0), np.concatenate(all_scores, axis=0)


# ---------------------------------------------------------------------------
# Reverse mapping
# ---------------------------------------------------------------------------


def build_reverse_item_mapping(
    index_mapping: Dict[str, Dict[Any, int]],
    output_column: str,
) -> Dict[int, str]:
    """Build ``{integer_index → original_item_id}`` for the output column.

    Only real catalog values are kept: the OOV slot (``0``) and the
    sentinel keys ``create_index_mapping`` reserves (``None`` / NaN → ``1``,
    ``-1`` / ``"-1"`` → ``2``) are dropped, so the PAD token can never be
    returned as a recommendation.
    """
    fwd = index_mapping.get(output_column, {})
    rev = {}
    for raw_val, idx in fwd.items():
        if idx == 0 or is_reserved_key(raw_val):
            continue
        rev[int(idx)] = raw_val
    return rev


def reserved_logit_indices(rev_map: Dict[int, Any], n_classes: int) -> List[int]:
    """Logit indices in ``[0, n_classes)`` that are not a real catalog item."""
    return [i for i in range(n_classes) if i not in rev_map]


def _mask_indices(logits: torch.Tensor, exclude_indices: Optional[Sequence[int]]) -> torch.Tensor:
    """Set ``logits[:, exclude_indices]`` to ``-inf`` (no-op for 1-D / empty)."""
    if not exclude_indices or logits.dim() != 2:
        return logits
    idx = torch.as_tensor(
        [i for i in exclude_indices if 0 <= i < logits.size(1)],
        dtype=torch.long,
        device=logits.device,
    )
    if idx.numel() == 0:
        return logits
    logits = logits.clone()
    logits[:, idx] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_inference(
    task_dir: str,
    data_path: str,
    output_path: str = "result.json",
    batch_size: int = 128,
    k: int = 10,
    device: str = "cpu",
) -> dict:
    """End-to-end inference pipeline.

    1. Load trained model from *task_dir*.
    2. Read raw data from *data_path* (parquet or CSV).
    3. Preprocess with the same transforms used during training.
    4. Run batch forward pass and extract top-k items.
    5. Reverse-map indices to original item IDs.
    6. Write ``result.json`` with ``{top10s, user, time}``.

    Returns:
        The result dict that was written.
    """
    t0 = time.time()

    # --- 1. Load model ---
    banner("Inference", logger=logger)
    logger.info("task_dir: %s", task_dir)
    model, project_config, index_mapping, params = load_trained_model(task_dir, device)
    logger.info(
        "Model: %s",
        params.get("recommender_module_class", "?").split(".")[-1],
    )

    # --- 2. Read raw data ---
    logger.info("Reading data: %s", data_path)
    if data_path.endswith(".parquet"):
        df_raw = pd.read_parquet(data_path)
    else:
        df_raw = pd.read_csv(data_path)
    logger.info("  rows=%d", len(df_raw))

    # Preserve original user IDs before indexing
    user_col = project_config.user_column.name
    original_user_ids = df_raw[user_col].tolist()

    # --- 3. Preprocess ---
    logger.info("Preprocessing ...")
    df_processed, dataset = preprocess_raw_data(
        df_raw,
        project_config,
        index_mapping,
    )
    logger.info("  dataset size=%d", len(dataset))

    # --- 4. Predict ---
    logger.info(
        "Running inference (batch_size=%d, k=%d, device=%s)",
        batch_size,
        k,
        device,
    )
    output_col = project_config.output_column.name
    rev_map = build_reverse_item_mapping(index_mapping, output_col)
    top_k_indices, top_k_scores = predict_top_k(
        model,
        dataset,
        batch_size=batch_size,
        k=k,
        device=device,
        exclude_indices=reserved_logit_indices(rev_map, _n_output_classes(index_mapping, output_col)),
    )

    # --- 5. Reverse map ---
    logger.info("Mapping indices to item IDs ...")

    top10s: List[List[Any]] = []
    for row_indices in top_k_indices:
        items = []
        for idx in row_indices:
            raw_id = rev_map.get(int(idx), None)
            if raw_id is None:
                # Index 0 (unknown) / 1 (padding) or unseen — skip
                continue
            # Convert stringified ints back to int if possible
            try:
                raw_id = int(raw_id)
            except (ValueError, TypeError):
                pass
            items.append(raw_id)
        top10s.append(items)

    elapsed = time.time() - t0

    # --- 6. Write result ---
    result = {
        "top10s": top10s,
        "user": original_user_ids,
        "time": round(elapsed, 4),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    banner("Inference summary", logger=logger)
    logger.info("Result saved → %s", output_path)
    logger.info("  users:           %d", len(result["user"]))
    logger.info(
        "  top-%d per user: %d items",
        k,
        len(result["top10s"][0]) if result["top10s"] else 0,
    )
    logger.info("  elapsed:         %.2fs", elapsed)

    return result


# ---------------------------------------------------------------------------
# Batch inference — collect full logits + targets
# ---------------------------------------------------------------------------


def _collect_logits_and_targets(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run batch forward and collect *all* logits and targets.

    Returns:
        (logits, targets) — ``logits`` shape ``[N, C]`` or ``[N]``,
        ``targets`` shape ``[N]``.
    """
    torch_device = torch.device(device)
    batch_sampler = FasterBatchSampler(dataset, batch_size, shuffle=False)
    loader = NoAutoCollationDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
    )

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            inputs = x if isinstance(x, (list, tuple)) else [x]
            inputs = [t.to(torch_device) if isinstance(t, torch.Tensor) else t for t in inputs]
            logits = model(*inputs).cpu()
            all_logits.append(logits)

            target = y[0] if isinstance(y, (list, tuple)) else y
            all_targets.append(target.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------


def compute_ranking_metrics(
    ground_truth_items: List[str],
    sorted_actions_list: List[List[Any]],
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Compute ranking metrics (NDCG, HitRate, MRR, MAP).

    Args:
        ground_truth_items: The true item id per observation (as strings).
        sorted_actions_list: Ranked predicted items per observation
            (descending score, original IDs).
        ks: List of cutoffs.

    Returns:
        Dict like ``{"ndcg@5": 0.xx, "hit_rate@5": ..., ...}``.
    """
    results: Dict[str, float] = {}

    for k in sorted(ks):
        ndcg_scores: List[float] = []
        hit_scores: List[float] = []
        rr_scores: List[float] = []
        ap_scores: List[float] = []

        for gt_item, ranked_items in zip(ground_truth_items, sorted_actions_list):
            ranked_k = [str(x) for x in ranked_items[:k]]

            if gt_item in ranked_k:
                rank_pos = ranked_k.index(gt_item) + 1
                ndcg_scores.append(1.0 / np.log2(rank_pos + 1))
                hit_scores.append(1.0)
                rr_scores.append(1.0 / rank_pos)
                ap_scores.append(1.0 / rank_pos)
            else:
                ndcg_scores.append(0.0)
                hit_scores.append(0.0)
                rr_scores.append(0.0)
                ap_scores.append(0.0)

        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
        results[f"hit_rate@{k}"] = float(np.mean(hit_scores))
        results[f"mrr@{k}"] = float(np.mean(rr_scores))
        results[f"map@{k}"] = float(np.mean(ap_scores))

    return results


def compute_roc_auc(
    all_logits: torch.Tensor,
    all_targets: torch.Tensor,
    loss_function: str = "ce",
) -> Dict[str, float]:
    """Compute ROC-AUC from raw model logits and integer targets.

    Args:
        all_logits: ``[N, C]`` for multi-class or ``[N]`` for binary.
        all_targets: ``[N]`` integer class labels.
        loss_function: The loss used during training (determines
            whether sigmoid is applied for binary).

    Returns:
        ``{"roc_auc_ovr": ...}`` for multi-class or
        ``{"roc_auc": ...}`` for binary.
    """
    from sklearn.metrics import roc_auc_score

    y_true = all_targets.numpy().astype(int)
    is_multiclass = all_logits.dim() == 2 and all_logits.size(1) > 1
    results: Dict[str, float] = {}

    try:
        if is_multiclass:
            y_probs = torch.softmax(all_logits, dim=1).numpy()
            n_classes = y_probs.shape[1]
            present_classes = np.unique(y_true)

            if len(present_classes) < 2:
                logger.warning("ROC-AUC skipped: fewer than 2 classes in dataset")
                return {}

            y_true_onehot = np.zeros((len(y_true), n_classes), dtype=int)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1

            results["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true_onehot[:, present_classes],
                    y_probs[:, present_classes],
                    multi_class="ovr",
                    average="macro",
                )
            )
        else:
            # Same guard as SupervisedModelTraining._compute_roc_auc: sklearn
            # returns NaN here, which json.dump writes as invalid ``NaN``.
            if len(np.unique(y_true)) < 2:
                logger.warning("ROC-AUC skipped: only one class present in y_true (undefined)")
                return {}

            logit_losses = {
                "bce_logits",
                "bce_logits_smooth",
                "ce",
                "focal_loss",
                "bpr",
                "ranknet",
                "lambdarank",
            }
            if loss_function in logit_losses:
                y_scores = torch.sigmoid(all_logits).numpy()
            else:
                y_scores = all_logits.numpy()
            y_scores = np.clip(y_scores, 0.0, 1.0)

            results["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        logger.exception("ROC-AUC computation failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# Evaluate orchestrator
# ---------------------------------------------------------------------------


def run_evaluate(
    task_dir: str,
    data_path: Optional[str] = None,
    ranking_k: List[int] = [5, 10, 20],
    batch_size: int = 128,
    device: str = "cpu",
    output_path: str = "eval_results.json",
) -> dict:
    """Unified evaluation pipeline.

    Two modes:

    * **Re-eval** (``data_path=None``): re-evaluates the test split that
      was used during training.  Locates the test CSV via the data
      preparation task.
    * **New data** (``data_path`` given): evaluates an arbitrary
      CSV/parquet that contains the ground-truth target column.

    In both modes the function:

    1. Loads the trained model and artefacts.
    2. Obtains or preprocesses the evaluation DataFrame + Dataset.
    3. Runs batch forward to collect full logits and top-k ranked items.
    4. Computes ranking metrics (NDCG, HitRate, MRR, MAP) for each *k*.
    5. Computes ROC-AUC (binary or multi-class OvR).
    6. Writes a JSON report.

    Returns:
        The result dict.
    """
    t0 = time.time()

    # --- 1. Load model ---
    banner("Evaluation", logger=logger)
    logger.info("task_dir: %s", task_dir)
    model, project_config, index_mapping, params = load_trained_model(task_dir, device)
    model_name = params.get("recommender_module_class", "?").split(".")[-1]
    loss_function = params.get("loss_function", "ce")
    logger.info("Model: %s | loss=%s", model_name, loss_function)

    # --- 2. Obtain evaluation data ---
    if data_path is not None:
        # New dataset
        data_source = data_path
        logger.info("Reading new data: %s", data_path)
        if data_path.endswith(".parquet"):
            df_raw = pd.read_parquet(data_path)
        else:
            df_raw = pd.read_csv(data_path)
        logger.info("  rows=%d", len(df_raw))
        logger.info("Preprocessing ...")
        df_processed, dataset = preprocess_raw_data(
            df_raw,
            project_config,
            index_mapping,
        )
    else:
        # Re-eval: find the test CSV from the data preparation task
        data_source = "test_split"
        logger.info("Locating test split from training run ...")
        df_processed, dataset = _load_test_split(
            task_dir,
            params,
            project_config,
            index_mapping,
        )

    n_samples = len(dataset)
    logger.info("Dataset size: %d", n_samples)

    # --- 3. Collect logits + top-k ---
    max_k = max(ranking_k)
    logger.info(
        "Running inference (batch_size=%d, k=%d, device=%s)",
        batch_size,
        max_k,
        device,
    )

    all_logits, all_targets = _collect_logits_and_targets(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    output_col = project_config.output_column.name
    rev_map = build_reverse_item_mapping(index_mapping, output_col)
    n_classes = all_logits.size(1) if all_logits.dim() == 2 else 1
    top_k_indices, top_k_scores = _topk_from_logits(
        all_logits, max_k, exclude_indices=reserved_logit_indices(rev_map, n_classes)
    )

    # --- 4. Ranking metrics ---
    logger.info("Computing ranking metrics ...")

    # Build sorted_actions (original IDs) per observation
    sorted_actions_list: List[List[Any]] = []
    for row_indices in top_k_indices:
        items = []
        for idx in row_indices:
            raw_id = rev_map.get(int(idx), None)
            if raw_id is not None:
                items.append(raw_id)
        sorted_actions_list.append(items)

    # Ground truth: the target lives in ``output_col`` (the vocabulary the
    # logits are over) and ``df_processed`` is already indexed, so map each
    # index back to its original id — the same space as ``sorted_actions_list``.
    # Unknown / reserved targets become "None" and never match.
    gt_items = [str(rev_map.get(_safe_int(v))) for v in df_processed[output_col]]

    ranking_metrics = compute_ranking_metrics(
        gt_items,
        sorted_actions_list,
        ks=ranking_k,
    )

    if ranking_metrics:
        banner("Ranking metrics", logger=logger)
        for metric_name, value in ranking_metrics.items():
            logger.info("  %-20s %.4f", metric_name, value)

    # --- 5. ROC-AUC ---
    logger.info("Computing ROC-AUC ...")
    roc_auc_metrics = compute_roc_auc(all_logits, all_targets, loss_function)

    if roc_auc_metrics:
        banner("ROC-AUC", logger=logger)
        for metric_name, value in roc_auc_metrics.items():
            logger.info("  %-20s %.4f", metric_name, value)

    elapsed = time.time() - t0

    # --- 6. Write result ---
    result = {
        "task_id": os.path.basename(task_dir),
        "model": model_name,
        "data_source": data_source,
        "n_samples": n_samples,
        "ranking_k": ranking_k,
        "ranking_metrics": ranking_metrics,
        "roc_auc_metrics": roc_auc_metrics,
        "time": round(elapsed, 4),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Evaluation saved → %s", output_path)
    logger.info("  samples: %d", n_samples)
    logger.info("  elapsed: %.2fs", elapsed)

    return result


# ---------------------------------------------------------------------------
# Helpers for re-eval mode
# ---------------------------------------------------------------------------


def _load_test_split(
    task_dir: str,
    params: dict,
    project_config: ProjectConfig,
    index_mapping: Dict[str, Dict[Any, int]],
) -> Tuple[pd.DataFrame, Dataset]:
    """Load the test CSV that was used during training.

    The file lives under the dataset output directory with a naming
    convention produced by the data-preparation Luigi task.
    """

    # Resolve the dataset directory from the data preparation task
    task_cls = project_config.prepare_data_frames_task
    # dataset_dir is a class property — get it from an uninitialised instance
    task_instance = task_cls.__new__(task_cls)
    dataset_dir = task_instance.dataset_dir

    # Build the file-name pattern used by BasePrepareDataFrames
    test_size = params.get("test_size", 0.2)
    test_split_type = params.get("test_split_type", "random")
    seed = params.get("seed", 42)

    # Must match ``BasePrepareDataFrames.output()``, which formats test_size
    # with ``%.2f`` (0.2 -> "0.20").
    prefix = f"test_{float(test_size):.2f}_test={test_split_type}_{int(seed)}_"

    # Scan the dataset directory for matching test CSV
    test_csv = None
    if os.path.isdir(dataset_dir):
        for fname in os.listdir(dataset_dir):
            if fname.startswith(prefix) and fname.endswith(".csv"):
                test_csv = os.path.join(dataset_dir, fname)
                break

    if test_csv is None:
        raise FileNotFoundError(
            f"Could not find test split CSV in {dataset_dir} "
            f"with prefix '{prefix}'. "
            f"Run the training pipeline first or provide --data explicitly."
        )

    logger.info("Test CSV: %s", test_csv)

    df = pd.read_csv(test_csv)
    logger.info("  rows=%d", len(df))

    # Apply same preprocessing as training
    df = preprocess_interactions_data_frame(df, project_config)
    transform_with_indexing(df, index_mapping, project_config)

    dataset = InteractionsDataset(
        data_frame=df,
        embeddings_for_metadata=None,
        project_config=project_config,
        index_mapping=index_mapping,
    )

    return df, dataset


def _n_output_classes(index_mapping: Dict[str, Dict[Any, int]], output_col: str) -> int:
    """Logit width implied by the output column's index mapping."""
    values = list(index_mapping.get(output_col, {}).values())
    return int(max(values)) + 1 if values else 0


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _topk_from_logits(
    logits: torch.Tensor,
    k: int,
    exclude_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-k indices and scores from logits tensor."""
    if logits.dim() == 2 and logits.size(1) > 1:
        logits = _mask_indices(logits, exclude_indices)
        actual_k = min(k, logits.size(1))
        scores, indices = logits.topk(actual_k, dim=1)
    else:
        scores = logits.unsqueeze(1) if logits.dim() == 1 else logits
        indices = torch.zeros_like(scores, dtype=torch.long)
    return indices.numpy(), scores.numpy()


# ---------------------------------------------------------------------------
# BCE catalog scoring (per-row × candidate expansion)
# ---------------------------------------------------------------------------


def score_candidates_emb(
    model: nn.Module,
    df_processed: pd.DataFrame,
    project_config: ProjectConfig,
    index_mapping: Dict[str, Dict[Any, int]],
    full_catalog: np.ndarray,
    candidate_provider: Optional[CandidateProvider] = None,
    batch_size: int = 1024,
    device: str = "cpu",
    show_progress: bool = True,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """Score per-row candidates via embedding similarity.

    For embedding models that implement ``get_target_emb()`` and return
    ``[B, emb_dim]`` from ``forward()`` (L2-normalised).  Unlike
    ``score_candidates_bce``, there is **no per-candidate row expansion** —
    each context row produces a single ``pred_emb``, which is then compared
    against the candidate embeddings via dot product (= cosine sim, both sides
    are L2-normalised).

    Algorithm::

        for batch of rows:
            pred_emb = model(*context_inputs)      # [B, D]
        for each row i:
            cand_emb = item_target_emb[eligible[i]]  # [n_cands, D]
            scores[i] = pred_emb[i] @ cand_emb.T    # [n_cands]

    Returns ``(turn_scores, eligible_per_row)`` — same format as
    ``score_candidates_bce``.
    """

    if not hasattr(model, "get_target_emb"):
        raise AttributeError(
            "score_candidates_emb requires model.get_target_emb(track_ids). Use score_candidates_bce for BCE/CE models."
        )

    torch_device = torch.device(device)
    n_turns = len(df_processed)
    full_catalog = np.asarray(full_catalog, dtype=np.int64)

    # Precompute and cache all item embeddings via the model's canonical
    # interface — works for both untied (separate item_target_emb) and tied
    # (shared seq_emb_layer.track_emb) variants.  get_target_emb returns
    # L2-normalised vectors.
    with torch.no_grad():
        all_ids = torch.arange(model._n_items, device=torch_device, dtype=torch.long)
        all_item_embs = model.get_target_emb(all_ids).float()  # [n_items, D]

    # Single forward pass per batch of context rows (no candidate expansion).
    ds = InteractionsDataset(df_processed, None, project_config, index_mapping)
    loader = NoAutoCollationDataLoader(
        ds,
        batch_sampler=FasterBatchSampler(ds, batch_size, shuffle=False),
        num_workers=0,
        pin_memory=False,
    )

    pred_embs_list: List[torch.Tensor] = []
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Embedding forward passes")
    with torch.no_grad():
        for x, _ in iterator:
            inputs = x if isinstance(x, (list, tuple)) else [x]
            inputs = [t.to(torch_device) if isinstance(t, torch.Tensor) else t for t in inputs]
            emb = model(*inputs).float()  # [B, D] L2-normalised
            pred_embs_list.append(emb.cpu())

    all_pred_embs = torch.cat(pred_embs_list)  # [n_turns, D]

    # Per-row scoring: dot product vs candidate set.
    out_scores: List[torch.Tensor] = []
    out_eligible: List[np.ndarray] = []

    row_iterator = range(n_turns)
    if show_progress:
        row_iterator = tqdm(row_iterator, total=n_turns, desc="Ranking candidates")

    for i in row_iterator:
        if candidate_provider is None:
            eligible = full_catalog
        else:
            row = df_processed.iloc[i].to_dict()
            eligible = np.asarray(
                candidate_provider.get_candidates(row, full_catalog),
                dtype=np.int64,
            )

        cand_embs = all_item_embs[eligible]  # [n_cands, D]
        scores = (all_pred_embs[i].to(torch_device) @ cand_embs.T).cpu()  # [n_cands]
        out_scores.append(scores)
        out_eligible.append(eligible)

    return out_scores, out_eligible


def score_candidates_ce(
    model: nn.Module,
    df_processed: pd.DataFrame,
    project_config: ProjectConfig,
    index_mapping: Dict[str, Dict[Any, int]],
    full_catalog: np.ndarray,
    candidate_provider: Optional[CandidateProvider] = None,
    batch_size: int = 1024,
    device: str = "cpu",
    show_progress: bool = True,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """Score per-row candidates via a CE model's full-vocab logits head.

    For multi-class CE models that output ``[B, n_items]`` from a single
    forward pass (no ``target_id`` in the inputs, no candidate expansion).

    Algorithm::

        for batch of rows:
            logits = model(*context_inputs)        # [B, n_items]
        for each row i:
            scores[i] = logits[i, eligible[i]]     # [n_cands]

    Returns ``(turn_scores, eligible_per_row)`` — same format as
    ``score_candidates_bce`` / ``score_candidates_emb``.

    Use this when ``project_config.item_is_input is False`` and the model
    classifier emits one logit per catalog item.
    """
    torch_device = torch.device(device)
    n_turns = len(df_processed)
    full_catalog = np.asarray(full_catalog, dtype=np.int64)

    ds = InteractionsDataset(df_processed, None, project_config, index_mapping)
    loader = NoAutoCollationDataLoader(
        ds,
        batch_sampler=FasterBatchSampler(ds, batch_size, shuffle=False),
        num_workers=0,
        pin_memory=False,
    )

    all_logits: List[torch.Tensor] = []
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="CE forward passes")
    with torch.no_grad():
        for x, _ in iterator:
            inputs = x if isinstance(x, (list, tuple)) else [x]
            inputs = [t.to(torch_device) if isinstance(t, torch.Tensor) else t for t in inputs]
            logits = model(*inputs)  # [B, n_items]
            if logits.dim() != 2:
                raise ValueError(f"score_candidates_ce expects [B, n_items] logits; got shape {tuple(logits.shape)}.")
            all_logits.append(logits.float().cpu())

    logits_all = torch.cat(all_logits, dim=0)  # [n_turns, n_items]
    n_items = logits_all.size(1)

    out_scores: List[torch.Tensor] = []
    out_eligible: List[np.ndarray] = []

    row_iterator = range(n_turns)
    if show_progress:
        row_iterator = tqdm(row_iterator, total=n_turns, desc="Ranking candidates")

    for i in row_iterator:
        if candidate_provider is None:
            eligible = full_catalog
        else:
            row = df_processed.iloc[i].to_dict()
            eligible = np.asarray(
                candidate_provider.get_candidates(row, full_catalog),
                dtype=np.int64,
            )
        # Clip to vocab range — CandidateProviders may return ids that
        # post-date this checkpoint's index_mapping. Rare, but mirrors the
        # safety net in predict_top_k.
        eligible = eligible[(eligible >= 0) & (eligible < n_items)]
        scores = logits_all[i, torch.as_tensor(eligible)]
        out_scores.append(scores)
        out_eligible.append(eligible)

    return out_scores, out_eligible


def score_candidates_bce(
    model: nn.Module,
    df_processed: pd.DataFrame,
    project_config: ProjectConfig,
    index_mapping: Dict[str, Dict[Any, int]],
    full_catalog: np.ndarray,
    candidate_provider: Optional[CandidateProvider] = None,
    candidate_feature_overrides: Optional[Callable[[pd.DataFrame, np.ndarray, np.ndarray], pd.DataFrame]] = None,
    batch_size: int = 4096,
    turn_batch_size: int = 32,
    device: str = "cpu",
    show_progress: bool = True,
    embeddings_for_metadata: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """Score per-row candidates via BCE expansion.

    For each row in ``df_processed``, expand it across its eligible
    candidate set, score every (row, candidate) pair via the model, and
    return the per-row score tensor plus the candidate idx vector that
    produced it.

    The eligible set comes from ``candidate_provider`` when supplied; when
    omitted the full catalog is used (the legacy behaviour).

    Item-dependent input columns must describe each candidate, not the
    row's original item. By default the project's ``candidate_feature_lookup``
    / ``candidate_feature_fn`` (from ``dataset_extra_params``) are applied
    through :func:`mars_rec.data.candidate_features.apply_candidate_features_to_frame`,
    exactly as in training. The optional ``candidate_feature_overrides``
    callback replaces that default: it receives the expanded chunk DataFrame
    and the full per-chunk candidate idx vector and should return the chunk
    DataFrame with project-specific columns rewritten in place.

    Args:
        model: trained nn.Module returning a single logit per row.
        df_processed: framework-preprocessed observations, one row per turn.
        project_config: project config (needed for the item column name).
        index_mapping: framework index map (passed to InteractionsDataset).
        full_catalog: int64 idx of every catalog item, sorted, no PAD/OOV.
        candidate_provider: optional eligibility provider.
        candidate_feature_overrides: project-specific per-candidate feature
            rewrite callback; receives ``(expanded_df, cand_array, row_offsets)``
            and must return the modified DataFrame. ``row_offsets`` is a
            length-(chunk_size+1) cumulative array so callbacks can slice
            per-row segments without reconstructing the offsets themselves.
        batch_size: DataLoader batch size for the scoring forward pass.
        turn_batch_size: number of input rows expanded per chunk.
        device: torch device (cpu/cuda/mps).
        show_progress: print a tqdm bar over chunks.

    Returns:
        ``(turn_scores, eligible_per_row)`` — both lists of length
        ``len(df_processed)``. ``turn_scores[i]`` is a 1-D tensor of length
        ``len(eligible_per_row[i])`` with the model's score for each
        eligible candidate.
    """
    item_col = project_config.item_column.name
    torch_device = torch.device(device)
    n_turns = len(df_processed)
    full_catalog = np.asarray(full_catalog, dtype=np.int64)

    # Without an explicit override callback, apply the project's
    # ``candidate_feature_lookup`` / ``candidate_feature_fn`` with the same
    # helper the negative samplers and the training-time evaluation use.
    extra = project_config.dataset_extra_params or {}
    default_lookup = None
    default_fn = None
    if candidate_feature_overrides is None:
        default_lookup = resolve_candidate_lookup(extra.get(CANDIDATE_LOOKUP_PARAM), index_mapping)
        default_fn = validate_candidate_feature_fn(extra.get(CANDIDATE_FN_PARAM))

    out_scores: List[torch.Tensor] = []
    out_eligible: List[np.ndarray] = []

    n_chunks = (n_turns + turn_batch_size - 1) // turn_batch_size
    iterator = range(0, n_turns, turn_batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_chunks, desc="Scoring")

    for chunk_start in iterator:
        chunk_end = min(chunk_start + turn_batch_size, n_turns)
        chunk_size = chunk_end - chunk_start
        chunk_df = df_processed.iloc[chunk_start:chunk_end]

        # Decide eligible candidates per row in the chunk.
        if candidate_provider is None:
            eligible_per_row = [full_catalog] * chunk_size
        else:
            eligible_per_row = []
            for j in range(chunk_size):
                row = chunk_df.iloc[j].to_dict()
                eligible = candidate_provider.get_candidates(row, full_catalog)
                eligible_per_row.append(np.asarray(eligible, dtype=np.int64))

        # Build the cand_array (concatenation of eligible idx vectors) and
        # the per-row offsets that consumers / overrides slice on.
        sizes = np.array([len(e) for e in eligible_per_row], dtype=np.int64)
        row_offsets = np.concatenate(([0], np.cumsum(sizes)))
        total_pairs = int(row_offsets[-1])
        if total_pairs == 0:
            # Pathological: every row had an empty eligible set. Emit empties.
            for _ in range(chunk_size):
                out_scores.append(torch.empty(0))
                out_eligible.append(np.empty(0, dtype=np.int64))
            continue

        cand_array = np.concatenate(eligible_per_row).astype(np.int64)
        # Per-row repeat counts feed into the iloc expansion.
        repeats = np.repeat(np.arange(chunk_size), sizes)
        expanded = chunk_df.iloc[repeats].reset_index(drop=True)
        expanded[item_col] = cand_array

        if candidate_feature_overrides is not None:
            expanded = candidate_feature_overrides(
                expanded,
                cand_array,
                row_offsets,
            )
        elif default_lookup or default_fn is not None:
            expanded = apply_candidate_features_to_frame(
                expanded,
                cand_array,
                project_config.input_columns,
                default_lookup,
                default_fn,
                item_column_name=item_col,
            )

        ds = InteractionsDataset(expanded, embeddings_for_metadata, project_config, index_mapping)
        loader = NoAutoCollationDataLoader(
            ds,
            batch_sampler=FasterBatchSampler(ds, batch_size, shuffle=False),
            num_workers=0,
            pin_memory=False,
        )

        chunk_scores: List[torch.Tensor] = []
        with torch.no_grad():
            for x, _ in loader:
                inputs = x if isinstance(x, (list, tuple)) else [x]
                inputs = [t.to(torch_device) if isinstance(t, torch.Tensor) else t for t in inputs]
                s = model(*inputs)
                if s.dim() > 1:
                    s = s.squeeze(-1)
                chunk_scores.append(s.cpu())

        flat = torch.cat(chunk_scores)  # [total_pairs]
        for j in range(chunk_size):
            start = int(row_offsets[j])
            end = int(row_offsets[j + 1])
            out_scores.append(flat[start:end])
            out_eligible.append(eligible_per_row[j])

    return out_scores, out_eligible
