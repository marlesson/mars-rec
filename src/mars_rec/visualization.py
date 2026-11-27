"""Architecture visualization for mars-rec models.

Produces a hand-rolled SVG flow diagram (no external dependencies) from a
trained or in-memory model. Two consumer surfaces:

- :func:`generate_architecture_diagram` — invoked automatically during
  training to produce ``architecture.svg`` next to each model checkpoint.
- :func:`print_text_summary` — terminal-friendly text table inspection.

Plus :func:`load_model_from_task` for re-hydrating a model from a task
directory.

Opt-in metadata
---------------
Models can override the role/heuristic system by implementing::

    def arch_metadata(self) -> dict:
        return {
            "starred_features": [...],
            "disabled_features": [...],
            "feature_roles": {"col_name": "role", ...},
            "block_roles": {"module_name": "role", ...},
            "gated_residuals": [(target_block, source_feature, gate_module), ...],
            "annotations": [(block_or_feature_name, "prose"), ...],
        }
"""

from __future__ import annotations

import glob
import html
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.utils.files import get_params
from mars_rec.utils.reflection import load_attr

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — roles, colors, labels
# =============================================================================

FEATURE_ROLES = {
    "seq_id",
    "user_cf",
    "user_profile",
    "content",
    "text",
    "target",
    "goal",
    "temporal",
    "scalar",
    "categorical",
    "mood",
    "session",
    "tag",
    "other",
}

BLOCK_ROLES = {
    "embedding",
    "projection",
    "fusion",
    "attention",
    "rnn",
    "conv",
    "mlp",
    "gate",
    "output",
    "norm",
    "other",
}

# Hex palette (no alpha; renderer applies opacity).
ROLE_COLORS = {
    # features
    "seq_id": "#4A90D9",
    "user_cf": "#4A90D9",
    "user_profile": "#3498DB",
    "content": "#2ECC71",
    "text": "#3498DB",
    "target": "#E74C3C",
    "goal": "#2ECC71",
    "temporal": "#F39C12",
    "scalar": "#95A5A6",
    "categorical": "#4A90D9",
    "mood": "#9B59B6",
    "session": "#9B59B6",
    "tag": "#E67E22",
    "other": "#95A5A6",
    # blocks
    "embedding": "#4A90D9",
    "projection": "#7F8C8D",
    "fusion": "#E8D44D",
    "attention": "#4A90D9",
    "rnn": "#4A90D9",
    "conv": "#2ECC71",
    "mlp": "#7F8C8D",
    "gate": "#E8D44D",
    "output": "#17A2B8",
    "norm": "#BDC3C7",
}

# Used by print_text_summary
IOTYPE_LABELS = {
    IOType.INDEXABLE: "Categorical",
    IOType.NUMBER: "Numeric",
    IOType.INDEXABLE_ARRAY: "Categorical Seq",
    IOType.FLOAT_ARRAY: "Float Array",
    IOType.INT_ARRAY: "Int Array",
    IOType.SEQUENCE: "Sequence",
}


# =============================================================================
# Dataclasses (IR)
# =============================================================================


@dataclass
class FeatureGroup:
    """IR node: a labelled bundle of input columns rendered as one box."""

    label: str
    role: str
    columns: List[str]
    shape: str = ""  # e.g. "[B]", "[B, L]", "[B, K]"
    starred: bool = False  # ★ — feeds a gated residual or special path
    disabled: bool = False  # off by default for this run


@dataclass
class Block:
    """IR node: a single layer / module box in the architecture diagram."""

    name: str
    label: str
    role: str
    params: int = 0
    frozen: bool = False
    shape_in: str = ""
    shape_out: str = ""
    details: str = ""


@dataclass
class Edge:
    """IR edge: a connection between two blocks (solid / dashed / gated)."""

    src: str
    dst: str
    shape: str = ""
    kind: str = "solid"  # solid | dashed | gated


@dataclass
class Annotation:
    """IR overlay: free-text annotation attached to a block (``anchor``)."""

    anchor: str
    text: str


@dataclass
class ModelIR:
    """Top-level intermediate representation of a model's architecture.

    Built by :func:`extract_model_ir` and consumed by
    :func:`render_svg_flow` to produce the architecture SVG saved
    alongside every trained run.
    """

    title: str
    subtitle: str
    feature_groups: List[FeatureGroup] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    frozen_params: int = 0


# =============================================================================
# Helpers
# =============================================================================


def _fmt_params(n: int) -> str:
    """Compact param count, e.g. ``"1.5M"`` / ``"3.2K"`` / ``"42"``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n}"


def _esc(s: str) -> str:
    return html.escape(s or "")


def _common_prefix(names: List[str]) -> str:
    """Common prefix truncated to the last ``_`` boundary."""
    if not names:
        return ""
    prefix = os.path.commonprefix(names)
    idx = prefix.rfind("_")
    if idx > 0:
        return prefix[: idx + 1]
    return prefix


def _shape_of(x: Any) -> Any:
    """Return a list of dims, or a list of shapes for tuple/list inputs."""
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [_shape_of(v) for v in x]
    if x is None:
        return None
    return "?"


def _fmt_shape(s: Any) -> str:
    """Format shape (list or nested) compactly: ``[B, 64]`` / ``[B, L] ×2``."""
    if s is None:
        return ""
    if isinstance(s, list) and s and isinstance(s[0], (int, type(None))):
        dims = ["B" if i == 0 else (str(d) if d is not None else "?") for i, d in enumerate(s)]
        return "[" + ", ".join(dims) + "]"
    if isinstance(s, list):
        parts = [_fmt_shape(v) for v in s]
        if len(set(parts)) == 1 and len(parts) > 1:
            return f"{parts[0]} ×{len(parts)}"
        return " / ".join(parts)
    return str(s)


# =============================================================================
# Role inference
# =============================================================================

_FEATURE_ROLE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^prev_.*_id"), "seq_id"),
    (re.compile(r"^prev_"), "seq_id"),
    (re.compile(r"user_cf|user_bpr"), "user_cf"),
    (re.compile(r"^age|gender|country|culture|lang|^age5"), "user_profile"),
    (re.compile(r"text_emb|user_query|assist.*response|goal_text|^text_"), "text"),
    (re.compile(r"goal_|progress|listener_goal"), "goal"),
    (re.compile(r"mood_cluster|mood_seq"), "mood"),
    (re.compile(r"session_year|session_week|session_quarter|holiday|session_progress"), "session"),
    (re.compile(r"^dow|^month|^year|day_of_year|^week"), "temporal"),
    (re.compile(r"^tag"), "tag"),
    (re.compile(r"^target|^track_id$|^album_id$|^artist_id$"), "target"),
]

_BLOCK_ROLE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"sasrec|attn|attention|transformer"), "attention"),
    (re.compile(r"gru|rnn|lstm|bigru"), "rnn"),
    (re.compile(r"conv"), "conv"),
    (re.compile(r"_proj|projection"), "projection"),
    (re.compile(r"_gate$"), "gate"),
    (re.compile(r"fusion|moe|adaptor|contextual"), "fusion"),
    (re.compile(r"classifier|^head$|output_head"), "output"),
    (re.compile(r"layernorm|batchnorm|^ln$|^bn$"), "norm"),
    (re.compile(r"^mlp|^fc|^ffn|^dense"), "mlp"),
    (re.compile(r"emb$|embedding"), "embedding"),
]


def _infer_feature_role(col: Column) -> str:
    """Map a Column to a semantic role."""
    name_lower = col.name.lower()
    for pattern, role in _FEATURE_ROLE_PATTERNS:
        if pattern.search(name_lower):
            return role
    if col.type == IOType.NUMBER:
        return "scalar"
    if col.type in (IOType.INDEXABLE_ARRAY, IOType.SEQUENCE, IOType.INT_ARRAY):
        return "categorical"
    if col.type == IOType.FLOAT_ARRAY:
        return "scalar"
    return "categorical"


def _infer_block_role(name: str, module: nn.Module) -> str:
    """Map a module name + class to a role."""
    lower = (name + module.__class__.__name__).lower()
    for pattern, role in _BLOCK_ROLE_PATTERNS:
        if pattern.search(lower):
            return role
    if isinstance(module, nn.Embedding):
        return "embedding"
    if isinstance(module, nn.MultiheadAttention):
        return "attention"
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return "conv"
    if isinstance(module, nn.Linear):
        return "projection" if "proj" in name.lower() else "mlp"
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        return "norm"
    if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
        return "rnn"
    return "other"


def _infer_shape(
    col: Column,
    dataset_extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Static tensor shape from a Column (no forward pass)."""
    _ = dataset_extra_params  # reserved
    if col.type == IOType.INDEXABLE:
        return "[B]"
    if col.type == IOType.NUMBER:
        return "[B]"
    if col.type == IOType.SEQUENCE:
        return "[B, L]"
    if col.type == IOType.INDEXABLE_ARRAY:
        if col.name.startswith("prev_") or "seq" in col.name.lower():
            return "[B, L]"
        return "[B, K]"
    if col.type == IOType.FLOAT_ARRAY:
        return "[B, K]"
    if col.type == IOType.INT_ARRAY:
        return "[B, K]"
    return "[B, ?]"


# =============================================================================
# Feature grouping
# =============================================================================


def _group_features_by_role(
    columns: List[Column],
    role_overrides: Dict[str, str],
    starred: set,
    disabled: set,
    dataset_extras: Optional[Dict[str, Any]] = None,
) -> List[FeatureGroup]:
    """Group columns by (role, common-prefix) for compact display."""
    annotated = []
    for col in columns:
        role = role_overrides.get(col.name) or _infer_feature_role(col)
        annotated.append((col, role))

    by_key: Dict[Tuple[str, str], List[Column]] = defaultdict(list)
    for col, role in annotated:
        parts = col.name.split("_")
        prefix = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        by_key[(role, prefix)].append(col)

    groups: List[FeatureGroup] = []
    for (role, prefix), cols in by_key.items():
        names = [c.name for c in cols]
        if len(cols) == 1:
            label = cols[0].name
        else:
            label = f"{prefix}_* ({len(cols)})"
        shape = _infer_shape(cols[0], dataset_extras)
        is_starred = any(n in starred for n in names)
        is_disabled = all(n in disabled for n in names) if disabled else False
        groups.append(
            FeatureGroup(
                label=label,
                role=role,
                columns=names,
                shape=shape,
                starred=is_starred,
                disabled=is_disabled,
            )
        )

    role_order = {
        "seq_id": 0,
        "content": 1,
        "user_cf": 2,
        "user_profile": 3,
        "text": 4,
        "goal": 5,
        "target": 6,
        "mood": 7,
        "tag": 8,
        "session": 9,
        "temporal": 10,
        "categorical": 11,
        "scalar": 12,
        "other": 99,
    }
    groups.sort(key=lambda g: (role_order.get(g.role, 99), not g.starred, g.label))
    return groups


def _group_features_legacy(columns: List[Column]) -> List[dict]:
    """IOType-based grouping for the text summary table.

    Returns a list of dicts ``{"label", "columns", "type"}``. Kept separate
    from the IR-style ``_group_features_by_role`` because the text summary
    cares about IOType categories, not semantic roles.
    """
    by_type: Dict[IOType, List[Column]] = defaultdict(list)
    for col in columns:
        by_type[col.type].append(col)

    groups: List[dict] = []
    for io_type, cols in by_type.items():
        if len(cols) <= 3:
            for col in cols:
                groups.append({"label": col.name, "columns": [col], "type": io_type})
            continue

        prefix_groups: Dict[str, List[Column]] = defaultdict(list)
        for col in cols:
            parts = col.name.rsplit("_", 1)
            if len(parts) > 1 and len(cols) > 5:
                key = parts[0].rsplit("_", 1)[0] if "_" in parts[0] else parts[0]
            else:
                key = col.name
            prefix_groups[key].append(col)

        merged_singles: List[Column] = []
        for _, pcols in prefix_groups.items():
            if len(pcols) >= 2:
                names = [c.name for c in pcols]
                common = _common_prefix(names)
                if common and len(common) > 2:
                    label = f"{common}* ({len(pcols)})"
                else:
                    label = f"{IOTYPE_LABELS.get(io_type, str(io_type))} features ({len(pcols)})"
                groups.append({"label": label, "columns": pcols, "type": io_type})
            else:
                merged_singles.extend(pcols)

        if merged_singles:
            if len(merged_singles) >= 3:
                groups.append(
                    {
                        "label": f"{IOTYPE_LABELS.get(io_type, str(io_type))} ({len(merged_singles)})",
                        "columns": merged_singles,
                        "type": io_type,
                    }
                )
            else:
                for col in merged_singles:
                    groups.append({"label": col.name, "columns": [col], "type": io_type})

    type_order = {
        IOType.SEQUENCE: 0,
        IOType.INDEXABLE_ARRAY: 1,
        IOType.FLOAT_ARRAY: 2,
        IOType.INT_ARRAY: 3,
        IOType.INDEXABLE: 4,
        IOType.NUMBER: 5,
    }
    groups.sort(key=lambda g: (type_order.get(g["type"], 99), g["label"]))
    return groups


# =============================================================================
# Block detection
# =============================================================================


def _detect_blocks(
    model: nn.Module,
    block_role_overrides: Dict[str, str],
) -> List[Block]:
    """Walk top-level children and emit one Block per architectural unit.

    Groups bare Embeddings by output dimension to keep grouped vocab tables
    visually together.
    """
    blocks: List[Block] = []
    raw_embeddings: List[Tuple[str, nn.Embedding]] = []

    for name, module in model.named_children():
        if isinstance(module, nn.Embedding):
            raw_embeddings.append((name, module))
            continue

        params = sum(p.numel() for p in module.parameters())
        frozen = params > 0 and not any(p.requires_grad for p in module.parameters())
        role = block_role_overrides.get(name) or _infer_block_role(name, module)
        cls = module.__class__.__name__
        details = ""

        if isinstance(module, nn.Linear):
            details = f"{module.in_features} → {module.out_features}"
        elif isinstance(module, nn.MultiheadAttention):
            details = f"d={module.embed_dim} h={module.num_heads}"
        elif isinstance(module, nn.ModuleList):
            n = sum(1 for _ in module.children())
            child_classes = {c.__class__.__name__ for c in module.children()}
            cls = f"{'/'.join(child_classes)} × {n}"
            details = f"{n} blocks"
        elif isinstance(module, nn.Sequential):
            linear_dims: List[int] = []
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    if not linear_dims:
                        linear_dims.append(m.in_features)
                    linear_dims.append(m.out_features)
            if linear_dims:
                details = " → ".join(str(d) for d in linear_dims)
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            details = f"hidden={module.hidden_size}, layers={module.num_layers}"

        blocks.append(
            Block(
                name=name,
                label=name,
                role=role,
                params=params,
                frozen=frozen,
                details=f"{cls} {details}".strip(),
            )
        )

    by_dim: Dict[int, List[Tuple[str, nn.Embedding]]] = defaultdict(list)
    for n, e in raw_embeddings:
        by_dim[e.embedding_dim].append((n, e))

    emb_blocks: List[Block] = []
    for dim, members in sorted(by_dim.items(), reverse=True):
        names = [n for n, _ in members]
        total_p = sum(e.num_embeddings * e.embedding_dim for _, e in members)
        frozen_count = sum(1 for _, e in members if not e.weight.requires_grad)
        all_frozen = frozen_count == len(members)
        member_labels = [
            f"{n}: {e.num_embeddings}×{e.embedding_dim}" + (" ❄" if not e.weight.requires_grad else "")
            for n, e in members[:5]
        ]
        if len(members) > 5:
            member_labels.append(f"+{len(members) - 5} more")
        if len(members) == 1:
            label = names[0]
        else:
            label = f"Emb d={dim} (×{len(members)})"
            if frozen_count > 0 and not all_frozen:
                label += f" [{frozen_count} frozen]"
            elif all_frozen:
                label += " [all frozen]"
        emb_blocks.append(
            Block(
                name=f"_emb_d{dim}",
                label=label,
                role="embedding",
                params=total_p,
                frozen=all_frozen,
                shape_out=f"[B, ..., {dim}]",
                details=" · ".join(member_labels),
            )
        )

    return emb_blocks + blocks


# =============================================================================
# Forward-pass tracing
# =============================================================================


def _build_synthetic_batch(
    project_config: ProjectConfig,
    index_mapping: Optional[Dict[str, Any]],
    model: Optional[nn.Module] = None,
    batch_size: int = 2,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Build a positional list of tensors matching ``project_config.input_columns``.

    Uses index 0 (typically the OOV/padding slot) for all indexable inputs:
    safe regardless of true vocab size, and shape capture is value-invariant.
    """
    _ = index_mapping  # reserved for future use
    seq_len = int(getattr(model, "seq_max_len", 7) or 7)
    array_k = 5

    inputs: List[torch.Tensor] = []
    for col in project_config.input_columns:
        if col.type == IOType.INDEXABLE:
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif col.type == IOType.NUMBER:
            t = torch.zeros(batch_size, dtype=torch.float32, device=device)
        elif col.type == IOType.SEQUENCE:
            t = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        elif col.type == IOType.INDEXABLE_ARRAY:
            k = seq_len if (col.name.startswith("prev_") or "seq" in col.name.lower()) else array_k
            t = torch.zeros(batch_size, k, dtype=torch.long, device=device)
        elif col.type == IOType.FLOAT_ARRAY:
            t = torch.zeros(batch_size, array_k, dtype=torch.float32, device=device)
        elif col.type == IOType.INT_ARRAY:
            t = torch.zeros(batch_size, array_k, dtype=torch.long, device=device)
        else:
            t = torch.zeros(batch_size, dtype=torch.float32, device=device)
        inputs.append(t)
    return inputs


def trace_model_shapes(
    model: nn.Module,
    project_config: ProjectConfig,
    index_mapping: Optional[Dict[str, Any]],
    batch_size: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run a forward pass and capture per-module input/output shapes.

    Returns ``{"shapes": {name: {"in": [...], "out": ...}}, "order": [...],
    "error": str_or_None}``. On error, returns whatever was captured up to
    the failure point.
    """
    inputs = _build_synthetic_batch(
        project_config,
        index_mapping,
        model=model,
        batch_size=batch_size,
        device=device,
    )

    shapes: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    handles: List[Any] = []

    def make_hook(name: str):
        def hook(module: nn.Module, args: tuple, out: Any) -> None:
            shapes[name] = {
                "in": [_shape_of(a) for a in args],
                "out": _shape_of(out),
            }
            order.append(name)

        return hook

    for name, m in model.named_modules():
        if name == "":
            continue
        handles.append(m.register_forward_hook(make_hook(name)))

    error: Optional[str] = None
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning("Forward-pass tracing failed partway: %s", error)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        if was_training:
            model.train()

    return {"shapes": shapes, "order": order, "error": error}


def _apply_trace_to_blocks(
    blocks: List[Block],
    model: nn.Module,
    trace: Dict[str, Any],
) -> None:
    """Populate ``shape_in``/``shape_out`` on blocks from a trace dict.

    For grouped embedding blocks (``_emb_d<dim>``) aggregates from any member
    embedding with matching dim. Mutates ``blocks`` in place.
    """
    shapes = trace.get("shapes", {})

    emb_dim_members: Dict[int, List[str]] = defaultdict(list)
    for name, module in model.named_children():
        if isinstance(module, nn.Embedding):
            emb_dim_members[module.embedding_dim].append(name)

    for blk in blocks:
        if blk.name.startswith("_emb_d"):
            try:
                dim = int(blk.name.replace("_emb_d", ""))
            except ValueError:
                continue
            members = emb_dim_members.get(dim, [])
            sample = next((shapes[n] for n in members if n in shapes), None)
            if sample is not None:
                blk.shape_in = _fmt_shape(sample["in"][0]) if sample["in"] else ""
                blk.shape_out = _fmt_shape(sample["out"])
            continue
        info = shapes.get(blk.name)
        if info:
            blk.shape_in = _fmt_shape(info["in"][0]) if info["in"] else ""
            blk.shape_out = _fmt_shape(info["out"])


# =============================================================================
# Edge building + IR extraction
# =============================================================================

_BACKBONE_ORDER = [
    "embedding",
    "fusion",
    "conv",
    "attention",
    "rnn",
    "projection",
    "mlp",
    "output",
]


def _build_edges(blocks: List[Block]) -> List[Edge]:
    """Default sequential edge list following role ordering."""
    edges: List[Edge] = []
    by_role: Dict[str, List[Block]] = defaultdict(list)
    for b in blocks:
        by_role[b.role].append(b)

    prev: Optional[Block] = None
    for role in _BACKBONE_ORDER:
        for b in by_role.get(role, []):
            if prev is not None:
                edges.append(Edge(src=prev.name, dst=b.name))
            prev = b
    return edges


def extract_model_ir(
    model: nn.Module,
    project_config: ProjectConfig,
    params: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
) -> ModelIR:
    """Build a :class:`ModelIR` from a model + config.

    Honors ``model.arch_metadata()`` when implemented. When ``trace`` is
    supplied (output of :func:`trace_model_shapes`), block shape_in/out are
    populated with real tensor shapes captured via forward hooks.
    """
    arch_meta: Dict[str, Any] = {}
    if hasattr(model, "arch_metadata") and callable(model.arch_metadata):
        try:
            arch_meta = model.arch_metadata() or {}
        except Exception:
            arch_meta = {}

    starred = set(arch_meta.get("starred_features", []) or [])
    disabled = set(arch_meta.get("disabled_features", []) or [])
    feature_roles: Dict[str, str] = arch_meta.get("feature_roles", {}) or {}
    block_roles: Dict[str, str] = arch_meta.get("block_roles", {}) or {}
    gated_residuals = arch_meta.get("gated_residuals", []) or []
    user_annotations: List[Tuple[str, str]] = arch_meta.get("annotations", []) or []

    dataset_extras = getattr(project_config, "dataset_extra_params", {}) or {}

    feature_groups = _group_features_by_role(
        project_config.input_columns,
        role_overrides=feature_roles,
        starred=starred,
        disabled=disabled,
        dataset_extras=dataset_extras,
    )

    blocks = _detect_blocks(model, block_role_overrides=block_roles)
    if trace is not None:
        _apply_trace_to_blocks(blocks, model, trace)

    edges = _build_edges(blocks)
    for target_block, source_feature, _gate_module in gated_residuals:
        edges.append(Edge(src=source_feature, dst=target_block, kind="gated"))

    annotations = [Annotation(anchor=anchor, text=text) for anchor, text in user_annotations]

    model_cls = params.get("recommender_module_class", "Unknown").split(".")[-1]
    loss = params.get("loss_function", "?")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p = total - trainable

    subtitle = (
        f"Loss: {loss}  ·  "
        f"Total: {_fmt_params(total)}  ·  "
        f"Trainable: {_fmt_params(trainable)}  ·  "
        f"Frozen: {_fmt_params(frozen_p)}"
    )

    return ModelIR(
        title=model_cls,
        subtitle=subtitle,
        feature_groups=feature_groups,
        blocks=blocks,
        edges=edges,
        annotations=annotations,
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen_p,
    )


# =============================================================================
# SVG layout — constants
# =============================================================================

CANVAS_WIDTH = 1100
MARGIN_X = 32
MARGIN_TOP = 24

TITLE_HEIGHT = 56
PILL_HEIGHT = 28
PILL_GAP_X = 8
PILL_GAP_Y = 8
PILL_MAX_PER_ROW = 6
PILL_BLOCK_TITLE_HEIGHT = 22

BLOCK_HEIGHT = 64
BLOCK_GAP = 28
BLOCK_WIDTH_NARROW = 200
BLOCK_WIDTH_WIDE = 320

_ROLE_TITLES = {
    "embedding": "Embeddings",
    "fusion": "Fusion / Mixer",
    "conv": "Conv / FFT",
    "attention": "Self-Attention",
    "rnn": "Recurrent Encoder",
    "projection": "Projection",
    "mlp": "Feed-Forward / MLP",
    "output": "Output Head",
    "norm": "Normalization",
    "gate": "Gates",
}


@dataclass
class _NodePos:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h


# =============================================================================
# SVG primitives
# =============================================================================


def _pill(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    starred: bool = False,
    disabled: bool = False,
    sub: Optional[str] = None,
) -> str:
    """Rounded pill for a feature group."""
    opacity_fill = 0.12 if disabled else 0.22
    opacity_stroke = 0.5 if disabled else 1.0
    text_color = "#888" if disabled else "#222"
    star = " ★" if starred else ""
    sub_text = ""
    if sub:
        sub_text = (
            f'<text x="{x + w / 2:.1f}" y="{y + h - 6:.1f}" text-anchor="middle" '
            f'font-size="8.5" fill="#666" font-style="italic">{_esc(sub)}</text>'
        )
        label_y = y + h / 2 - 1
    else:
        label_y = y + h / 2 + 4
    return (
        f"<g>"
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{color}" fill-opacity="{opacity_fill}" '
        f'stroke="{color}" stroke-width="1.2" stroke-opacity="{opacity_stroke}"/>'
        f'<text x="{x + w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" '
        f'font-size="10" font-weight="600" fill="{text_color}">'
        f"{_esc(label)}{star}</text>"
        f"{sub_text}"
        f"</g>"
    )


def _block_box(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    role: str,
    sub_lines: List[str],
    color: str,
    frozen: bool = False,
    fill_strength: float = 0.18,
) -> str:
    """Backbone block (embedding, attention, fusion, mlp, output)."""
    _ = role  # reserved for future role-specific styling
    title_with_freeze = f"{title} ❄" if frozen else title
    sub_svg = ""
    for i, line in enumerate(sub_lines):
        sub_svg += (
            f'<text x="{x + w / 2:.1f}" y="{y + 28 + i * 11:.1f}" '
            f'text-anchor="middle" font-size="9" fill="#555">{_esc(line)}</text>'
        )
    return (
        f"<g>"
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
        f'fill="{color}" fill-opacity="{fill_strength}" '
        f'stroke="{color}" stroke-width="1.4"/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
        f'font-size="11" font-weight="700" fill="#222">{_esc(title_with_freeze)}</text>'
        f"{sub_svg}"
        f"</g>"
    )


def _arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    kind: str = "solid",
    color: str = "#666",
) -> str:
    dash = ' stroke-dasharray="4,3"' if kind == "dashed" else ""
    if kind == "gated":
        color = "#C4A800"
    label_svg = ""
    if label:
        mx = (x1 + x2) / 2 + 10
        my = (y1 + y2) / 2 + 3
        label_svg = (
            f'<text x="{mx:.1f}" y="{my:.1f}" font-size="9" font-style="italic" fill="#666">{_esc(label)}</text>'
        )
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="1.2"{dash} marker-end="url(#arrowhead)"/>'
        f"{label_svg}"
    )


# =============================================================================
# SVG layout
# =============================================================================


def _layout_feature_strip(
    groups: List[FeatureGroup],
    y_start: float,
    canvas_width: float,
) -> Tuple[List[str], Dict[str, _NodePos], float]:
    """Feature pills in a grid. Returns (fragments, positions, end_y)."""
    if not groups:
        return [], {}, y_start

    fragments: List[str] = []
    positions: Dict[str, _NodePos] = {}

    fragments.append(
        f'<text x="{MARGIN_X}" y="{y_start + 14}" font-size="11" font-weight="700" '
        f'fill="#333">Input Features ({sum(len(g.columns) for g in groups)} columns)</text>'
    )
    y = y_start + PILL_BLOCK_TITLE_HEIGHT

    rows = (len(groups) + PILL_MAX_PER_ROW - 1) // PILL_MAX_PER_ROW
    panel_h = rows * (PILL_HEIGHT + PILL_GAP_Y) + 16
    fragments.append(
        f'<rect x="{MARGIN_X - 4}" y="{y - 6}" width="{canvas_width - 2 * MARGIN_X + 8}" '
        f'height="{panel_h:.1f}" rx="8" fill="#222" fill-opacity="0.03" stroke="#ddd"/>'
    )

    inner_x = MARGIN_X + 8
    pill_area_w = canvas_width - 2 * MARGIN_X - 16
    pill_w = (pill_area_w - (PILL_MAX_PER_ROW - 1) * PILL_GAP_X) / PILL_MAX_PER_ROW

    for i, g in enumerate(groups):
        row = i // PILL_MAX_PER_ROW
        col = i % PILL_MAX_PER_ROW
        x = inner_x + col * (pill_w + PILL_GAP_X)
        y_pill = y + row * (PILL_HEIGHT + PILL_GAP_Y) + 4
        color = ROLE_COLORS.get(g.role, "#999")
        sub = g.shape if g.shape else None
        fragments.append(
            _pill(x, y_pill, pill_w, PILL_HEIGHT, g.label, color, starred=g.starred, disabled=g.disabled, sub=sub)
        )
        positions[g.label] = _NodePos(x, y_pill, pill_w, PILL_HEIGHT)

    return fragments, positions, y + panel_h


def _layout_backbone(
    blocks: List[Block],
    y_start: float,
    canvas_width: float,
) -> Tuple[List[str], Dict[str, _NodePos], float]:
    """Render backbone blocks vertically, with parallel groups side-by-side."""
    if not blocks:
        return [], {}, y_start

    by_role: Dict[str, List[Block]] = {}
    for b in blocks:
        by_role.setdefault(b.role, []).append(b)

    fragments: List[str] = []
    positions: Dict[str, _NodePos] = {}
    y = y_start + 8

    for role in _BACKBONE_ORDER:
        members = by_role.get(role)
        if not members:
            continue

        color = ROLE_COLORS.get(role, "#666")
        title = _ROLE_TITLES.get(role, role.title())

        fragments.append(
            f'<text x="{MARGIN_X}" y="{y + 12}" font-size="10" font-weight="700" fill="{color}">{_esc(title)}</text>'
        )
        y += 18

        n = len(members)
        per_row = min(n, 4)
        total_w = canvas_width - 2 * MARGIN_X
        block_w = max(BLOCK_WIDTH_NARROW, min(BLOCK_WIDTH_WIDE, (total_w - (per_row - 1) * BLOCK_GAP) / per_row))
        rows = (n + per_row - 1) // per_row
        for i, blk in enumerate(members):
            row = i // per_row
            col = i % per_row
            row_count = min(per_row, n - row * per_row)
            row_w = row_count * block_w + (row_count - 1) * BLOCK_GAP
            row_x_start = (canvas_width - row_w) / 2
            x = row_x_start + col * (block_w + BLOCK_GAP)
            y_blk = y + row * (BLOCK_HEIGHT + 12)

            sub_lines: List[str] = []
            if blk.details:
                d = blk.details
                if len(d) > 60:
                    d = d[:57] + "…"
                sub_lines.append(d)
            if blk.params > 0:
                sub_lines.append(f"{_fmt_params(blk.params)} params")
            if blk.shape_out:
                sub_lines.append(f"→ {blk.shape_out}")

            fragments.append(
                _block_box(
                    x,
                    y_blk,
                    block_w,
                    BLOCK_HEIGHT,
                    title=blk.label,
                    role=role,
                    sub_lines=sub_lines,
                    color=color,
                    frozen=blk.frozen,
                    fill_strength=0.28 if role in ("attention", "rnn") else 0.18,
                )
            )
            positions[blk.name] = _NodePos(x, y_blk, block_w, BLOCK_HEIGHT)

        y += rows * (BLOCK_HEIGHT + 12) + 8

    return fragments, positions, y


def _connect_backbone(
    blocks: List[Block],
    positions: Dict[str, _NodePos],
    feature_anchor_x: float,
    feature_anchor_y: float,
) -> List[str]:
    """Draw vertical arrows between role groups."""
    fragments: List[str] = []
    by_role: Dict[str, List[Block]] = {}
    for b in blocks:
        by_role.setdefault(b.role, []).append(b)

    stages: List[Tuple[str, List[Block]]] = []
    for role in _BACKBONE_ORDER:
        if role in by_role:
            stages.append((role, by_role[role]))

    if not stages:
        return fragments

    _, first_blocks = stages[0]
    for b in first_blocks:
        if b.name not in positions:
            continue
        p = positions[b.name]
        fragments.append(_arrow(feature_anchor_x, feature_anchor_y, p.cx, p.top - 2, kind="dashed", color="#AAA"))

    for i in range(len(stages) - 1):
        _, prev_blocks = stages[i]
        _, next_blocks = stages[i + 1]
        prev_centers = [(positions[b.name].cx, positions[b.name].bottom) for b in prev_blocks if b.name in positions]
        next_tops = [(positions[b.name].cx, positions[b.name].top) for b in next_blocks if b.name in positions]
        if not prev_centers or not next_tops:
            continue
        if len(prev_centers) == 1 and len(next_tops) == 1:
            px, py = prev_centers[0]
            nx, ny = next_tops[0]
            fragments.append(_arrow(px, py + 2, nx, ny - 2))
        elif len(prev_centers) == 1:
            px, py = prev_centers[0]
            for nx, ny in next_tops:
                fragments.append(_arrow(px, py + 2, nx, ny - 2))
        elif len(next_tops) == 1:
            nx, ny = next_tops[0]
            for px, py in prev_centers:
                fragments.append(_arrow(px, py + 2, nx, ny - 2))
        else:
            for px, py in prev_centers:
                best = min(next_tops, key=lambda t: abs(t[0] - px))
                fragments.append(_arrow(px, py + 2, best[0], best[1] - 2))

    return fragments


def _legend(
    y_start: float,
    canvas_width: float,
    roles_used: List[str],
) -> Tuple[List[str], float]:
    fragments = []
    fragments.append(
        f'<text x="{MARGIN_X}" y="{y_start + 14}" font-size="10" font-weight="700" fill="#333">Legend</text>'
    )
    y = y_start + 20
    x = MARGIN_X
    swatch_w = 14
    swatch_h = 12
    pad = 6
    for role in roles_used:
        color = ROLE_COLORS.get(role, "#999")
        label = _ROLE_TITLES.get(role, role)
        text_w = len(label) * 6.5
        if x + swatch_w + 4 + text_w > canvas_width - MARGIN_X:
            x = MARGIN_X
            y += 20
        fragments.append(
            f'<rect x="{x}" y="{y - swatch_h + 2}" width="{swatch_w}" height="{swatch_h}" '
            f'rx="2" fill="{color}" fill-opacity="0.35"/>'
        )
        fragments.append(f'<text x="{x + swatch_w + 4}" y="{y + 1}" font-size="9" fill="#555">{_esc(label)}</text>')
        x += swatch_w + 4 + text_w + pad * 2

    y += 18
    fragments.append(
        f'<text x="{MARGIN_X}" y="{y}" font-size="9" fill="#666" font-style="italic">'
        f"★ = injected as gated residual / special path  ·  "
        f"❄ = frozen parameters  ·  "
        f"greyed pill = disabled by default</text>"
    )
    return fragments, y + 10


# =============================================================================
# SVG render
# =============================================================================


def render_svg_flow(ir: ModelIR, output_path: str) -> str:
    """Render the IR as an SVG flow diagram and write to ``output_path``."""
    fragments: List[str] = []
    y = MARGIN_TOP

    fragments.append(
        f'<text x="{CANVAS_WIDTH / 2}" y="{y + 22}" text-anchor="middle" '
        f'font-size="18" font-weight="700" fill="#222">{_esc(ir.title)}</text>'
    )
    fragments.append(
        f'<text x="{CANVAS_WIDTH / 2}" y="{y + 42}" text-anchor="middle" '
        f'font-size="10" fill="#666">{_esc(ir.subtitle)}</text>'
    )
    y += TITLE_HEIGHT

    feat_fragments, _feat_positions, y = _layout_feature_strip(
        ir.feature_groups,
        y,
        CANVAS_WIDTH,
    )
    fragments.extend(feat_fragments)
    y += 16

    feature_bus_y = y - 8
    feature_bus_x = CANVAS_WIDTH / 2

    bb_fragments, bb_positions, y_after_backbone = _layout_backbone(
        ir.blocks,
        y,
        CANVAS_WIDTH,
    )

    arrow_fragments = _connect_backbone(
        ir.blocks,
        bb_positions,
        feature_bus_x,
        feature_bus_y,
    )
    fragments.extend(arrow_fragments)
    fragments.extend(bb_fragments)

    y = y_after_backbone + 8

    if ir.annotations:
        for ann in ir.annotations:
            fragments.append(
                f'<text x="{MARGIN_X}" y="{y + 12}" font-size="9" font-style="italic" '
                f'fill="#666">• {_esc(ann.anchor)}: {_esc(ann.text)}</text>'
            )
            y += 14
        y += 6

    roles_used: List[str] = []
    seen: set = set()
    for g in ir.feature_groups:
        if g.role not in seen:
            roles_used.append(g.role)
            seen.add(g.role)
    for b in ir.blocks:
        if b.role not in seen:
            roles_used.append(b.role)
            seen.add(b.role)
    legend_fragments, y = _legend(y, CANVAS_WIDTH, roles_used)
    fragments.extend(legend_fragments)

    total_height = int(y + MARGIN_TOP)

    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS_WIDTH} {total_height}" '
        f'role="img" '
        f'style="font-family: system-ui, -apple-system, sans-serif;">\n'
        f"  <title>{_esc(ir.title)} — architecture</title>\n"
        f"  <desc>{_esc(ir.subtitle)}</desc>\n"
        f"  <defs>\n"
        f'    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">\n'
        f'      <path d="M0,0 L8,3 L0,6" fill="#666"/>\n'
        f"    </marker>\n"
        f"  </defs>\n"
        f'  <rect width="100%" height="100%" fill="white"/>\n  ' + "\n  ".join(fragments) + "\n</svg>\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg)
    return output_path


# =============================================================================
# Text summary
# =============================================================================


def introspect_model(model: nn.Module) -> List[dict]:
    """Walk ``model.named_modules()`` and extract one row per layer."""
    layers = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            cls = module.__class__.__name__
            if isinstance(module, nn.ModuleList):
                n_children = sum(1 for _ in module.children())
                child_classes = set(c.__class__.__name__ for c in module.children())
                child_label = "/".join(child_classes) if child_classes else "?"
                layers.append(
                    {
                        "name": name,
                        "class_name": f"{cls}[{child_label} x{n_children}]",
                        "params": sum(p.numel() for p in module.parameters()),
                        "details": f"{n_children} sub-modules",
                        "is_container": True,
                    }
                )
            continue

        cls = module.__class__.__name__
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        details = ""

        if isinstance(module, nn.Embedding):
            details = f"vocab={module.num_embeddings}, dim={module.embedding_dim}"
            if not module.weight.requires_grad:
                details += " [frozen]"
        elif isinstance(module, nn.Linear):
            details = f"in={module.in_features}, out={module.out_features}"
        elif isinstance(module, nn.MultiheadAttention):
            details = f"d={module.embed_dim}, heads={module.num_heads}"
        elif isinstance(module, nn.LayerNorm):
            details = f"shape={list(module.normalized_shape)}"
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            details = f"in={module.in_channels}, out={module.out_channels}, k={module.kernel_size}"
        elif isinstance(module, nn.BatchNorm1d):
            details = f"features={module.num_features}"
        elif isinstance(module, nn.Dropout):
            details = f"p={module.p}"

        layers.append(
            {
                "name": name,
                "class_name": cls,
                "params": n_params,
                "details": details,
                "is_container": False,
            }
        )

    return layers


def print_text_summary(
    model: nn.Module,
    project_config: ProjectConfig,
    params: dict,
) -> None:
    """Print an enhanced text summary of the model.

    Used by ``mars-rec visualize --mode summary``. No I/O beyond stdout.
    """
    model_cls = params.get("recommender_module_class", "Unknown").split(".")[-1]
    loss_fn = params.get("loss_function", "?")
    extra = params.get("recommender_extra_params", {})

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    width = 90
    print("=" * width)
    print(f"  Model: {model_cls}")
    print(f"  Loss:  {loss_fn}")
    print(f"  Params: {_fmt_params(total)} total | {_fmt_params(trainable)} trainable | {_fmt_params(frozen)} frozen")
    if extra:
        print(f"  Extra:  {json.dumps(extra, indent=None)}")
    print("=" * width)

    input_cols = project_config.input_columns
    feature_groups = _group_features_legacy(input_cols)

    print(f"\n{'─' * width}")
    print(f"  INPUT FEATURES ({len(input_cols)} columns)")
    print(f"{'─' * width}")

    row_fmt = "  {:<40} {:<20} {:>6}"
    print(row_fmt.format("Group", "Type", "Count"))
    print("  " + "-" * 68)

    for grp in feature_groups:
        type_label = IOTYPE_LABELS.get(grp["type"], str(grp["type"]))
        count = len(grp["columns"])
        print(row_fmt.format(grp["label"][:40], type_label, count))

    type_counts: Dict[Any, int] = defaultdict(int)
    for col in input_cols:
        type_counts[col.type] += 1
    print()
    print("  Type Summary:")
    for io_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {IOTYPE_LABELS.get(io_type, str(io_type)):<25} {count:>4} columns")

    out = project_config.output_column
    print(f"\n  Output: {out.name} ({IOTYPE_LABELS.get(out.type, str(out.type))})")

    layers = introspect_model(model)
    print(f"\n{'─' * width}")
    print(f"  MODEL LAYERS ({len(layers)} modules)")
    print(f"{'─' * width}")

    hdr = "  {:<35} {:<25} {:>12}  {}"
    print(hdr.format("Name", "Class", "Params", "Details"))
    print("  " + "-" * 86)

    for layer in layers:
        n = layer["params"]
        p_str = _fmt_params(n) if n > 0 else "-"
        name_short = layer["name"][-35:] if len(layer["name"]) > 35 else layer["name"]
        cls_short = layer["class_name"][-25:] if len(layer["class_name"]) > 25 else layer["class_name"]
        print(hdr.format(name_short, cls_short, p_str, layer["details"]))

    print("=" * width)


# =============================================================================
# Loading
# =============================================================================


def load_model_from_task(
    task_dir: str,
    device: str = "cpu",
    load_weights: bool = False,
) -> Tuple[nn.Module, ProjectConfig, dict, dict]:
    """Load model, config, index_mapping, and params from a task directory.

    By default does NOT load checkpoint weights (only architecture needed for
    visualization). Pass ``load_weights=True`` for inference scenarios.
    """
    params = get_params(task_dir)
    project_path = params["project"]
    module_class_path = params["recommender_module_class"]
    extra_params = params.get("recommender_extra_params", {})

    project_config: ProjectConfig = deepcopy(load_attr(project_path, ProjectConfig))

    idx_path = os.path.join(task_dir, "index_mapping.pkl")
    with open(idx_path, "rb") as f:
        index_mapping = pickle.load(f)

    module_class = load_attr(module_class_path, nn.Module)
    model = module_class(
        project_config=project_config,
        index_mapping=index_mapping,
        **extra_params,
    )

    if load_weights:
        ckpt_files = glob.glob(os.path.join(task_dir, "weights*.ckpt"))
        if ckpt_files:
            ckpt_path = max(ckpt_files, key=os.path.getmtime)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict)

    model.eval()
    return model, project_config, index_mapping, params


# =============================================================================
# Public entry point
# =============================================================================


def generate_architecture_diagram(
    model: nn.Module,
    project_config: ProjectConfig,
    params: dict,
    output_path: str,
    *,
    index_mapping: Optional[dict] = None,
    trace: bool = True,
    device: str = "cpu",
) -> str:
    """Generate ``architecture.svg`` and return its path.

    Called automatically during training (with ``trace=True`` default,
    requiring ``index_mapping``) and by ``mars-rec visualize``. Always
    writes a single SVG file with no external dependencies.
    """
    trace_data: Optional[dict] = None
    if trace:
        if index_mapping is None:
            logger.warning("trace=True requires index_mapping; skipping forward pass")
        else:
            try:
                trace_data = trace_model_shapes(
                    model,
                    project_config,
                    index_mapping,
                    device=device,
                )
                if trace_data.get("error"):
                    logger.warning(
                        "Partial trace captured (%d modules): %s",
                        len(trace_data.get("shapes", {})),
                        trace_data["error"],
                    )
            except Exception:
                logger.warning(
                    "trace_model_shapes failed; falling back to static introspection",
                    exc_info=True,
                )
                trace_data = None

    ir = extract_model_ir(model, project_config, params, trace=trace_data)
    rendered = render_svg_flow(ir, output_path)
    logger.info("Architecture diagram saved → %s", rendered)
    return rendered
