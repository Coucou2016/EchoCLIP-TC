"""EchoCLIP-TC / EchoCLIP-TA paper experiment matrix.

Primary IDs follow an R0–R6 + Oracle-EDES matrix (uniform-16 on VAL/TEST).
Legacy aliases B0/M1/M2/M4 remain for CLI compatibility.

These IDs do **not** invent clinical performance numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Primary Table-1 experiment IDs (paper path).
EXPERIMENT_IDS = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "ORACLE_EDES",
)

# Legacy aliases → primary IDs (kept so existing scripts/docs keep working).
LEGACY_ALIASES: Dict[str, str] = {
    "B0": "R0",
    "M1": "R1",
    "M2": "R5",
    "M4": "R6",
    "S0": "R2",
    "S1": "R3",
    "S2": "R4",
    "ORACLE-EDES": "ORACLE_EDES",
    "ORACLE": "ORACLE_EDES",
}

# Strategies forbidden on primary VAL/TEST (annotation-assisted or train-style).
ANNOTATION_ASSISTED_STRATEGIES = frozenset({"ed_es", "edes", "ed-es", "mixed"})
PRIMARY_EVAL_SPLITS = frozenset({"val", "valid", "validation", "test", "eval"})

# EchoCLIP external-protocol subset size (Christensen et al. Nature Medicine 2024).
ECHOCLIP_EXTERNAL_SUBSET_N = 5000
ECHOCLIP_EXTERNAL_SUBSET_SEED = 42

# Official EF grid when ``--paper`` / official_reproduction (0–100 inclusive, step 1).
OFFICIAL_EF_VALUES = list(range(0, 101))

# Non-paper default EF grid (coarse; not claimed as official parity).
DEFAULT_EF_VALUES = list(range(15, 81, 5))


@dataclass(frozen=True)
class ExperimentSpec:
    """One row of the paper protocol matrix."""

    id: str
    title: str
    description: str
    train: bool
    pool: str  # frames | mean | temporal | supervised
    calibrate: bool
    video_frames: Optional[int] = None  # None → use config
    sample_strategy: Optional[str] = None  # train / default
    eval_sample_strategy: Optional[str] = None  # VAL/TEST primary (uniform unless Oracle)
    temporal_type: Optional[str] = None
    supervised_head: Optional[str] = None  # none|linear|mlp|temporal_l1
    init_official: bool = True
    requires_checkpoint: bool = False
    annotation_assisted: bool = False
    legacy_aliases: Tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "R0": ExperimentSpec(
        id="R0",
        title="EchoCLIP-based zero-shot baseline",
        description=(
            "Frozen EchoCLIP towers; per-frame features + top-20% median EF. "
            "Default naming is 'EchoCLIP-based' until golden parity with official "
            "echo_CLIP is proven. Use --paper / --official-reproduction for the "
            "strict path (EF 0–100 step 1; no scratch fallback)."
        ),
        train=False,
        pool="frames",
        calibrate=False,
        video_frames=16,
        sample_strategy="uniform",
        eval_sample_strategy="uniform",
        temporal_type="none",
        init_official=True,
        requires_checkpoint=False,
        legacy_aliases=("B0",),
        notes=(
            "Do not invent clinical MAE. Published external ~7.1% is from Christensen "
            "et al.; reproduce only with official weights + EchoNet seed-42 subset "
            "and/or full TEST under --paper."
        ),
    ),
    "R1": ExperimentSpec(
        id="R1",
        title="Uniform-16 mean pool (no extra params)",
        description=(
            "Same frozen towers as R0. Uniform sample T=16 frames, mean-pool "
            "embeddings to one video vector, then EF regression."
        ),
        train=False,
        pool="mean",
        calibrate=False,
        video_frames=16,
        sample_strategy="uniform",
        eval_sample_strategy="uniform",
        temporal_type="none",
        init_official=True,
        requires_checkpoint=False,
        legacy_aliases=("M1",),
        notes="Primary ablation: temporal aggregation without a learned module.",
    ),
    "R2": ExperimentSpec(
        id="R2",
        title="S0: frozen mean + linear/ridge EF head",
        description=(
            "Supervised baseline: freeze EchoCLIP, mean-pool frame features, "
            "fit a linear (or ridge) EF head on TRAIN EF labels."
        ),
        train=True,
        pool="supervised",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        eval_sample_strategy="uniform",
        temporal_type="none",
        supervised_head="linear",
        init_official=True,
        requires_checkpoint=True,
        legacy_aliases=("S0",),
        notes="Label-supervised; not zero-shot.",
    ),
    "R3": ExperimentSpec(
        id="R3",
        title="S1: frozen mean + MLP EF head",
        description=(
            "Supervised baseline: freeze EchoCLIP, mean-pool features, train a "
            "small MLP EF head with L1/Huber on TRAIN EF labels."
        ),
        train=True,
        pool="supervised",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        eval_sample_strategy="uniform",
        temporal_type="none",
        supervised_head="mlp",
        init_official=True,
        requires_checkpoint=True,
        legacy_aliases=("S1",),
        notes="Label-supervised; not zero-shot.",
    ),
    "R4": ExperimentSpec(
        id="R4",
        title="S2: temporal aggregator + direct L1/Huber EF",
        description=(
            "Train temporal aggregator with direct EF regression (L1/Huber), "
            "not contrastive video–text loss."
        ),
        train=True,
        pool="supervised",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        eval_sample_strategy="uniform",
        temporal_type="transformer",
        supervised_head="temporal_l1",
        init_official=True,
        requires_checkpoint=True,
        legacy_aliases=("S2",),
        notes="Direct EF supervision; compare to R5 contrastive adaptation.",
    ),
    "R5": ExperimentSpec(
        id="R5",
        title="M2: EF-label-supervised temporal adaptation (contrastive)",
        description=(
            "Freeze vision/text towers; train Temporal Transformer with "
            "TemporalClipLoss on structured EF prompts (primary). "
            "This is parameter-efficient temporal adaptation of frozen EchoCLIP "
            "(EchoCLIP-TA), NOT a zero-shot temporal extension. "
            "EDV dilation captions are an optional ablation flag only."
        ),
        train=True,
        pool="temporal",
        calibrate=False,
        video_frames=16,
        sample_strategy="mixed",
        eval_sample_strategy="uniform",
        temporal_type="transformer",
        init_official=True,
        requires_checkpoint=True,
        legacy_aliases=("M2",),
        notes="Primary EchoCLIP-TA model before calibration.",
    ),
    "R6": ExperimentSpec(
        id="R6",
        title="R5 + val-fit temperature / conformal",
        description=(
            "Same as R5 encoding; fit temperature (EF<50) and split-conformal "
            "quantiles on VAL only; report ECE/Brier/coverage/abstention on TEST."
        ),
        train=True,
        pool="temporal",
        calibrate=True,
        video_frames=16,
        sample_strategy="mixed",
        eval_sample_strategy="uniform",
        temporal_type="transformer",
        init_official=True,
        requires_checkpoint=True,
        legacy_aliases=("M4",),
        notes="Calibration never retuned on TEST.",
    ),
    "ORACLE_EDES": ExperimentSpec(
        id="ORACLE_EDES",
        title="Oracle-EDES (annotation-assisted upper bound)",
        description=(
            "Uses ED/ES frame indices from VolumeTracings (or equivalent) at "
            "eval time. This is an annotation-assisted upper bound — NOT a "
            "primary VAL/TEST protocol result. Label clearly in tables."
        ),
        train=False,
        pool="mean",
        calibrate=False,
        video_frames=16,
        sample_strategy="ed_es",
        eval_sample_strategy="ed_es",
        temporal_type="none",
        init_official=True,
        requires_checkpoint=False,
        annotation_assisted=True,
        notes=(
            "Requires ed_frame/es_frame in the manifest. Do not mix into primary "
            "uniform-16 comparisons without an Oracle label."
        ),
    ),
}


def resolve_experiment_id(exp_id: str) -> str:
    """Map legacy aliases (B0/M1/…) to primary R* / ORACLE_EDES IDs."""
    key = str(exp_id).strip().upper().replace("-", "_")
    if key in EXPERIMENTS:
        return key
    if key in LEGACY_ALIASES:
        return LEGACY_ALIASES[key]
    # Also accept ORACLE-EDES style after normalize
    alt = str(exp_id).strip().upper().replace("_", "-")
    if alt in LEGACY_ALIASES:
        return LEGACY_ALIASES[alt]
    known = ", ".join(EXPERIMENT_IDS) + " (aliases: " + ", ".join(LEGACY_ALIASES) + ")"
    raise KeyError(f"Unknown experiment {exp_id!r}. Known: {known}")


def get_experiment(exp_id: str) -> ExperimentSpec:
    return EXPERIMENTS[resolve_experiment_id(exp_id)]


def list_experiments() -> List[ExperimentSpec]:
    return [EXPERIMENTS[i] for i in EXPERIMENT_IDS]


def assert_primary_eval_sampling(
    *,
    split: Optional[str],
    strategy: Optional[str],
    experiment_id: Optional[str] = None,
    allow_annotation_assisted: bool = False,
) -> None:
    """Hard-fail if primary VAL/TEST uses annotation-assisted sampling.

    Primary eval must be ``uniform`` (or ``random`` only if explicitly non-primary).
    ``ed_es`` / ``mixed`` on val/test raise ``ValueError`` unless the experiment
    is explicitly Oracle-EDES / ``allow_annotation_assisted=True``.
    """
    if strategy is None:
        return
    strat = str(strategy).strip().lower()
    split_key = str(split or "").strip().lower()
    if split_key not in PRIMARY_EVAL_SPLITS:
        return
    if strat not in ANNOTATION_ASSISTED_STRATEGIES:
        return
    exp = None
    if experiment_id:
        try:
            exp = get_experiment(experiment_id)
        except KeyError:
            exp = None
    if allow_annotation_assisted or (exp is not None and exp.annotation_assisted):
        return
    raise ValueError(
        f"Primary {split_key} evaluation forbids sample_strategy={strategy!r} "
        f"(annotation-assisted or train-style). Use uniform for R0–R6 / M1/M2/M4; "
        f"use experiment ORACLE_EDES (alias Oracle-EDES) for the labeled upper bound."
        + (f" (experiment_id={experiment_id})" if experiment_id else "")
    )


def resolve_eval_sample_strategy(
    *,
    spec: ExperimentSpec,
    cli_strategy: Optional[str] = None,
    cfg: Optional[dict] = None,
    split: str = "test",
) -> str:
    """VAL/TEST strategy: CLI > spec.eval_sample_strategy > cfg val_sample_strategy > uniform."""
    cfg = cfg or {}
    if cli_strategy:
        strategy = str(cli_strategy)
    elif spec.eval_sample_strategy:
        strategy = str(spec.eval_sample_strategy)
    else:
        strategy = str(
            cfg.get("val_sample_strategy", cfg.get("sample_strategy", "uniform"))
        )
    assert_primary_eval_sampling(
        split=split,
        strategy=strategy,
        experiment_id=spec.id,
        allow_annotation_assisted=spec.annotation_assisted,
    )
    return strategy


def protocol_output_dir(root: Path, exp_id: str) -> Path:
    resolved = resolve_experiment_id(exp_id)
    return Path(root) / "checkpoints" / "protocol" / resolved


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
    paper: bool = False,
    extra: Optional[dict] = None,
) -> dict:
    out = dict(metrics)
    out["experiment_id"] = experiment.id
    out["experiment_title"] = experiment.title
    out["protocol_pool"] = experiment.pool
    out["protocol_calibrate"] = experiment.calibrate
    out["protocol_notes"] = experiment.notes
    out["annotation_assisted"] = bool(experiment.annotation_assisted)
    if experiment.legacy_aliases:
        out["legacy_aliases"] = list(experiment.legacy_aliases)
    if demo:
        out["demo_mode"] = True
        out["demo_is_not_clinical"] = True
        out["note"] = (
            "DEMO MODE: metrics measure pipeline wiring only. "
            "Do not report as EchoNet / paper EF MAE."
        )
    if paper:
        out["official_reproduction"] = True
        out["paper_mode"] = True
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
    "annotation_assisted",
    "official_reproduction",
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
    if exp_ids is None:
        ids = list(EXPERIMENT_IDS)
    else:
        ids = [resolve_experiment_id(i) for i in exp_ids]
    out: Dict[str, dict] = {}
    for exp_id in ids:
        path = root / exp_id / "metrics.json"
        if not path.exists():
            # Legacy folder names (B0/M1/…)
            for alias, primary in LEGACY_ALIASES.items():
                if primary == exp_id:
                    alt = root / alias / "metrics.json"
                    if alt.exists():
                        path = alt
                        break
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
        "experiment_title",
        "mae",
        "rmse",
        "auc_ef_lt_50",
        "ece_ef_lt_50",
        "conformal_coverage",
        "load_source",
        "annotation_assisted",
        "demo_is_not_clinical",
    )
    if not rows:
        return "| (empty) |\n|---|\n| No protocol metrics.json found. |\n"
    # Enrich titles / Oracle labeling from EXPERIMENTS when missing
    enriched = []
    for row in rows:
        r = dict(row)
        exp_id = str(r.get("experiment_id", ""))
        if exp_id in EXPERIMENTS:
            spec = EXPERIMENTS[exp_id]
            r.setdefault("experiment_title", spec.title)
            r.setdefault("annotation_assisted", spec.annotation_assisted)
            if spec.annotation_assisted and not str(r.get("experiment_title", "")).lower().startswith(
                "oracle"
            ):
                r["experiment_title"] = spec.title
        enriched.append(r)
    rows = enriched
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
            elif h == "annotation_assisted" and val:
                cells.append("Oracle (annotation-assisted)")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    any_demo = any(r.get("demo_is_not_clinical") for r in rows)
    any_oracle = any(r.get("annotation_assisted") for r in rows)
    footer_bits = [
        "State `load_source` and split (TEST vs subset_5000) next to every table number.",
    ]
    if any_oracle:
        footer_bits.append(
            "**Oracle-EDES** is annotation-assisted and **not** a primary uniform-16 result."
        )
    if any_demo:
        footer_bits.insert(
            0,
            "demo / `scratch_fallback` / `simple_cnn` rows are pipeline wiring only — "
            "never report as EchoNet or Nature Medicine EF MAE.",
        )
    return "\n".join(lines) + "\n\n> **Honesty:** " + " ".join(footer_bits) + "\n"


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
        "protocol": "EchoCLIP-TC / EchoCLIP-TA R0–R6 + Oracle-EDES comparison",
        "experiment_ids": [r["experiment_id"] for r in rows],
        "n_experiments": len(rows),
        "fields": list(COMPARISON_FIELDS),
        "legacy_aliases": dict(LEGACY_ALIASES),
        "rows": rows,
        "note": (
            "Aggregate of checkpoints/protocol/<ID>/metrics.json. "
            "Demo / scratch rows are not clinical. Oracle-EDES is annotation-assisted."
        ),
        "any_demo": any(bool(r.get("demo_is_not_clinical")) for r in rows),
    }
    json_path = root / json_name
    md_path = root / md_name
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(comparison_to_markdown(rows), encoding="utf-8")
    return {"json": json_path, "md": md_path}
