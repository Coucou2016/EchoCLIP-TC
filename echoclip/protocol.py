"""EchoCLIP-TC paper experiment matrix (Table 1 modes B0 / M1 / M2 / M4).

These IDs are the locked protocol labels used by ``scripts/run_protocol.py``.
They do not invent clinical performance numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Locked Table-1 experiment IDs (paper path).
EXPERIMENT_IDS = ("B0", "M1", "M2", "M4")

# EchoCLIP external-protocol subset size (Christensen et al. Nature Medicine 2024).
ECHOCLIP_EXTERNAL_SUBSET_N = 5000
ECHOCLIP_EXTERNAL_SUBSET_SEED = 42


@dataclass(frozen=True)
class ExperimentSpec:
    """One row of the paper protocol matrix."""

    id: str
    title: str
    description: str
    train: bool
    pool: str  # frames | mean | temporal
    calibrate: bool
    video_frames: Optional[int] = None  # None → use config
    sample_strategy: Optional[str] = None
    temporal_type: Optional[str] = None  # for training
    init_official: bool = True
    requires_checkpoint: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "B0": ExperimentSpec(
        id="B0",
        title="Official EchoCLIP zero-shot",
        description=(
            "Load hf-hub:mkaichristensen/echo-clip (or local ckpt / scratch fallback). "
            "No temporal training. Per-frame features + official top-20% median EF."
        ),
        train=False,
        pool="frames",
        calibrate=False,
        video_frames=16,
        sample_strategy="uniform",
        temporal_type="none",
        init_official=True,
        requires_checkpoint=False,
        notes=(
            "Reproduce EchoCLIP external EF MAE ~7.1% only with official weights + "
            "EchoNet seed-42 5000 subset and/or full TEST — not with demo data."
        ),
    ),
    "M1": ExperimentSpec(
        id="M1",
        title="Cycle sampling + mean pool",
        description=(
            "Same frozen towers as B0. Sample T frames (cycle/uniform/mixed), "
            "mean-pool frame embeddings to one video vector, then EF regression. "
            "No extra trainable parameters."
        ),
        train=False,
        pool="mean",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        temporal_type="none",
        init_official=True,
        requires_checkpoint=False,
        notes="Ablation: temporal aggregation without a learned module.",
    ),
    "M2": ExperimentSpec(
        id="M2",
        title="Temporal aggregator (trained)",
        description=(
            "Freeze vision/text towers; train Temporal Transformer (or attn pool) "
            "with TemporalClipLoss on structured EchoNet captions."
        ),
        train=True,
        pool="temporal",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        temporal_type="transformer",
        init_official=True,
        requires_checkpoint=True,
        notes="Primary EchoCLIP-TC model before calibration.",
    ),
    "M4": ExperimentSpec(
        id="M4",
        title="M2 + val-fit temperature / conformal",
        description=(
            "Same as M2 encoding; fit temperature (EF<50) and split-conformal "
            "quantiles on VAL only; report ECE/Brier/coverage/abstention on TEST."
        ),
        train=True,
        pool="temporal",
        calibrate=True,
        video_frames=16,
        sample_strategy="mixed",
        temporal_type="transformer",
        init_official=True,
        requires_checkpoint=True,
        notes="Calibration never retuned on TEST.",
    ),
}


def get_experiment(exp_id: str) -> ExperimentSpec:
    key = str(exp_id).strip().upper()
    if key not in EXPERIMENTS:
        known = ", ".join(EXPERIMENT_IDS)
        raise KeyError(f"Unknown experiment {exp_id!r}. Known: {known}")
    return EXPERIMENTS[key]


def list_experiments() -> List[ExperimentSpec]:
    return [EXPERIMENTS[i] for i in EXPERIMENT_IDS]


def protocol_output_dir(root: Path, exp_id: str) -> Path:
    return Path(root) / "checkpoints" / "protocol" / str(exp_id).upper()


def metrics_path(root: Path, exp_id: str) -> Path:
    return protocol_output_dir(root, exp_id) / "metrics.json"


def write_subset_ids(
    pairs: Sequence[dict],
    path: Path,
    *,
    seed: int = ECHOCLIP_EXTERNAL_SUBSET_SEED,
    n: int = ECHOCLIP_EXTERNAL_SUBSET_N,
    source: str = "EchoNet-Dynamic",
    already_sampled: bool = False,
) -> Path:
    """Lock the seed-42 subset file names for paper reproducibility.

    If ``already_sampled`` is False and ``len(pairs) > n``, draw a deterministic
    random subset of size ``n`` with ``seed`` (matching ``subset_n`` in the
    EchoNet builder). Pass ``already_sampled=True`` when the caller already
    subsetted (e.g. ``build_echonet_manifest.py``).
    """
    import random

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(pairs)
    if not already_sampled and n < len(records):
        rng = random.Random(int(seed))
        idx = list(range(len(records)))
        rng.shuffle(idx)
        records = [records[i] for i in idx[: int(n)]]
    ids: List[str] = []
    for rec in records:
        name = rec.get("file_name") or rec.get("image") or ""
        ids.append(str(name))
    payload = {
        "protocol": "EchoCLIP external-style random subset",
        "source": source,
        "seed": int(seed),
        "n_requested": int(n),
        "n_written": len(ids),
        "ids": ids,
        "note": (
            "IDs are FileList stems / video names for the subset drawn with this seed. "
            "Do not reshuffle when comparing to published EchoCLIP external MAE."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also a plain text list for easy diffing
    txt = path.with_suffix(".txt")
    txt.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return path


def load_subset_ids(path: Path) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("ids") or [])


def merge_metrics_meta(
    metrics: dict,
    *,
    experiment: ExperimentSpec,
    demo: bool = False,
    extra: Optional[dict] = None,
) -> dict:
    out = dict(metrics)
    out["experiment_id"] = experiment.id
    out["experiment_title"] = experiment.title
    out["protocol_pool"] = experiment.pool
    out["protocol_calibrate"] = experiment.calibrate
    out["protocol_notes"] = experiment.notes
    if demo:
        out["demo_mode"] = True
        out["demo_is_not_clinical"] = True
        out["note"] = (
            "DEMO MODE: metrics measure pipeline wiring only. "
            "Do not report as EchoNet / paper EF MAE."
        )
    if extra:
        out.update(extra)
    return out


# Columns written into the cross-experiment comparison table (paper path).
COMPARISON_FIELDS = (
    "experiment_id",
    "mae",
    "rmse",
    "r2",
    "auc_ef_lt_50",
    "auc_ef_lt_40",
    "auc_ef_lt_30",
    "ece_ef_lt_50",
    "brier_ef_lt_50",
    "conformal_coverage",
    "conformal_mean_width",
    "abstention_mae",
    "abstention_rule",
    "temperature_ef_lt_50",
    "n_eval",
    "pool",
    "load_source",
    "ef_source",
    "demo_is_not_clinical",
    "video_frames",
    "sample_strategy",
    "seed",
)


def load_protocol_metrics(
    protocol_root: Path,
    exp_ids: Optional[Sequence[str]] = None,
) -> Dict[str, dict]:
    """Load ``metrics.json`` for each experiment under ``checkpoints/protocol``."""
    root = Path(protocol_root)
    ids = [str(i).upper() for i in (exp_ids or EXPERIMENT_IDS)]
    out: Dict[str, dict] = {}
    for exp_id in ids:
        path = root / exp_id / "metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        data = dict(data)
        data.setdefault("experiment_id", exp_id)
        out[exp_id] = data
    return out


def build_comparison_rows(
    metrics_by_id: Dict[str, dict],
    *,
    fields: Sequence[str] = COMPARISON_FIELDS,
) -> List[dict]:
    """Flatten per-experiment metrics into ordered table rows."""
    rows: List[dict] = []
    order = [i for i in EXPERIMENT_IDS if i in metrics_by_id]
    order.extend(sorted(k for k in metrics_by_id if k not in EXPERIMENT_IDS))
    for exp_id in order:
        src = metrics_by_id[exp_id]
        row = {"experiment_id": exp_id}
        for key in fields:
            if key == "experiment_id":
                continue
            if key in src:
                row[key] = src[key]
        rows.append(row)
    return rows


def comparison_to_markdown(rows: Sequence[dict]) -> str:
    """Render a compact markdown table (primary columns only)."""
    primary = (
        "experiment_id",
        "mae",
        "rmse",
        "auc_ef_lt_50",
        "ece_ef_lt_50",
        "conformal_coverage",
        "load_source",
        "demo_is_not_clinical",
    )
    if not rows:
        return (
            "| (empty) |\n|---|\n| No protocol metrics.json found. |\n"
        )
    headers = [h for h in primary if any(h in r for r in rows)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for h in headers:
            val = row.get(h, "")
            if val is None:
                cells.append("")
            elif isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    any_demo = any(r.get("demo_is_not_clinical") for r in rows)
    footer = (
        "\n\n> **Honesty:** demo / `scratch_fallback` / `simple_cnn` rows are "
        "pipeline wiring only — never report as EchoNet or Nature Medicine EF MAE.\n"
        if any_demo
        else "\n\n> State `load_source` and split (TEST vs subset_5000) next to every table number.\n"
    )
    return "\n".join(lines) + footer


def write_protocol_comparison(
    protocol_root: Path,
    *,
    exp_ids: Optional[Sequence[str]] = None,
    json_name: str = "comparison.json",
    md_name: str = "comparison.md",
) -> Dict[str, Path]:
    """Write comparison.json + comparison.md under the protocol root."""
    root = Path(protocol_root)
    root.mkdir(parents=True, exist_ok=True)
    metrics = load_protocol_metrics(root, exp_ids=exp_ids)
    rows = build_comparison_rows(metrics)
    payload = {
        "protocol": "EchoCLIP-TC Table-1 comparison",
        "experiment_ids": [r["experiment_id"] for r in rows],
        "n_experiments": len(rows),
        "fields": list(COMPARISON_FIELDS),
        "rows": rows,
        "note": (
            "Aggregate of checkpoints/protocol/<ID>/metrics.json. "
            "Demo / scratch rows are not clinical."
        ),
        "any_demo": any(bool(r.get("demo_is_not_clinical")) for r in rows),
    }
    json_path = root / json_name
    md_path = root / md_name
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(comparison_to_markdown(rows), encoding="utf-8")
    return {"json": json_path, "md": md_path}

