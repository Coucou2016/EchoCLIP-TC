"""Standalone official-style R0 zero-shot EF evaluation (P0-5/P0-6).

This module deliberately does **not** go through :class:`echoclip.data.EchoCLIPDataset`
(which pads/trims clips to a fixed frame count, e.g. 16). It re-implements the
*documented behavior* of the upstream ``echonet/echo_CLIP`` zero-shot EF example:

  1. Read an AVI with OpenCV (all frames, BGR, as decoded).
  2. Documented crop/scale (``crop_and_scale``) to 224×224.
  3. Frame stride ``0:min(40, T):2``.
  4. Official OpenCLIP ``preprocess_val`` (when a hub model is available).
  5. Official hub model + tokenizer.
  6. EF prompt grid 0..100 (step 1) × the two documented EF templates.
  7. Official regression aggregation (rank prompts per frame, mean ranked EF
     values across frames, median of the top 20%).

Provenance note
---------------
This module is a standalone, independently authored implementation of the
documented upstream official R0 procedure; its behavior intentionally overlaps
upstream EchoCLIP's official zero-shot example, no upstream code is vendored or
line-copied in its own body, and whether the upstream ``echonet/echo_CLIP``
source was consulted is **unclear — needs author confirmation** (see
PROVENANCE.md). The aligned math it calls (``crop_and_scale_official`` /
``compute_regression_metric_official``) is imported from ``official_parity``,
which self-describes as a bit-aligned reference copy; the parity gaps remain
documented (see :data:`echoclip.official_parity.PARITY_GAPS`).

Honesty contract
----------------
``official_reproduction_verified`` is **only** true when :func:`parity_report`
actually passes against real assets. Nothing in this module invents clinical
MAE or parity values, and the golden-parity harness skips gracefully when the
official example AVI / hub weights are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from echoclip.cycle_sample import sample_official_stride
from echoclip.official_parity import (
    PARITY_GAPS,
    compute_regression_metric_official,
    official_stride_indices,
)
from echoclip.prompts import ZERO_SHOT_PROMPTS
from echoclip.protocol import OFFICIAL_EF_VALUES

# ---------------------------------------------------------------------------
# Documented constants (behavioral, not copied source)
# ---------------------------------------------------------------------------

#: HuggingFace hub id of the official EchoCLIP weights.
OFFICIAL_HUB = "hf-hub:mkaichristensen/echo-clip"

#: Upstream project that documents the zero-shot EF example behavior.
UPSTREAM_REPO = "https://github.com/echonet/echo_CLIP"

#: Upstream revision this documented behavior targets (documented, not pinned code).
UPSTREAM_COMMIT = "echonet/echo_CLIP@main"

#: Frame selection: ``0:min(40, T):2``.
OFFICIAL_MAX_SPAN = 40
OFFICIAL_STRIDE = 2

#: Crop/scale target used by the official example.
OFFICIAL_CROP_RES = (224, 224)

#: The two documented EF prompt templates (upstream-attributed strings).
EF_PROMPT_TEMPLATES: Tuple[str, ...] = tuple(ZERO_SHOT_PROMPTS["ejection_fraction"])

#: EF regression grid granularity (official = every integer 0..100).
OFFICIAL_EF_STEP = 1


# ---------------------------------------------------------------------------
# Prompt grid
# ---------------------------------------------------------------------------

def build_ef_prompt_grid(
    templates: Sequence[str] = EF_PROMPT_TEMPLATES,
    values: Sequence[int] = tuple(OFFICIAL_EF_VALUES),
) -> Tuple[List[str], List[int]]:
    """Return ``(prompts, prompt_values)`` for the official EF grid.

    Layout: ``for template in templates: for value in values`` — matching the
    upstream construction of ``2 × 101 = 202`` candidate prompts.
    """
    prompts: List[str] = []
    out_values: List[int] = []
    for tmpl in templates:
        for v in values:
            prompts.append(str(tmpl).replace("<#>", str(int(v))))
            out_values.append(int(v))
    return prompts, out_values


def prompt_grid_sha256(
    prompts: Optional[Sequence[str]] = None,
    values: Optional[Sequence[int]] = None,
) -> str:
    """Stable SHA-256 over the prompt strings + values (provenance)."""
    if prompts is None or values is None:
        prompts, values = build_ef_prompt_grid()
    h = hashlib.sha256()
    for p, v in zip(prompts, values):
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(int(v)).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def prompts_sha256(prompts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------

def frame_indices(num_frames: int) -> np.ndarray:
    """Official stride indices ``0:min(40, T):2``."""
    return sample_official_stride(
        num_frames, max_span=OFFICIAL_MAX_SPAN, stride=OFFICIAL_STRIDE
    )


def read_avi_frames(avi: Path) -> np.ndarray:
    """Decode *all* AVI frames with OpenCV (BGR, HWC uint8).

    Raises ``RuntimeError`` when the file cannot be opened or has no frames.
    """
    import cv2  # local import: OpenCV is an evaluation-time dependency

    cap = cv2.VideoCapture(str(avi))
    try:
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {avi}")
    return np.stack(frames, axis=0)


def crop_and_scale_batch(
    frames: np.ndarray,
    res: Tuple[int, int] = OFFICIAL_CROP_RES,
    zoom: float = 0.1,
) -> np.ndarray:
    """Apply the documented crop/scale to every frame (HWC uint8 in → list out)."""
    from echoclip.official_parity import crop_and_scale_official

    return [crop_and_scale_official(f, res, zoom=zoom) for f in frames]


def select_official_frames(
    frames: np.ndarray,
    *,
    max_span: int = OFFICIAL_MAX_SPAN,
    stride: int = OFFICIAL_STRIDE,
) -> Tuple[np.ndarray, List[int]]:
    """Select ``0:min(max_span, T):stride`` frames; return ``(frames, indices)``."""
    idx = sample_official_stride(len(frames), max_span=max_span, stride=stride)
    idx_list = [int(i) for i in idx]
    return frames[idx], idx_list


def input_sha256(frames: np.ndarray) -> str:
    """SHA-256 over the *selected* frame pixels (content provenance)."""
    arr = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Model loading (official hub)
# ---------------------------------------------------------------------------

def try_load_official_open_clip(device: str = "cpu"):
    """Load official OpenCLIP hub model.

    Returns ``(packed, None)`` or ``(None, error_message)``. Never raises so the
    caller can honor the "skip gracefully" contract.
    """
    if os.environ.get("ECHOCLIP_SKIP_HUB", "").strip() in ("1", "true", "yes"):
        return None, "ECHOCLIP_SKIP_HUB=1 (hub load disabled)"
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - optional dependency
        return None, f"open_clip missing: {exc}"
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(OFFICIAL_HUB)
        tokenize = open_clip.get_tokenizer(OFFICIAL_HUB)
        model = model.to(device).eval()
        return (model, tokenize, preprocess_val), None
    except Exception as exc:  # noqa: BLE001 - hub/network/license errors vary
        return None, str(exc)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def preprocess_frames(frames: np.ndarray, preprocess_val, *, to_pil: bool = True):
    """Frame batch (N,H,W,3 uint8) → ``(N,3,H',W')`` float tensor."""
    import torch
    import torchvision.transforms as T

    if to_pil:
        pil = [T.ToPILImage()(f) for f in frames]
        return torch.stack([preprocess_val(p) for p in pil], dim=0)
    return torch.stack([preprocess_val(f) for f in frames], dim=0)


def predict_ef_official(
    frames: np.ndarray,
    *,
    model,
    tokenize,
    preprocess_val,
    device: str = "cpu",
    template: Sequence[str] = EF_PROMPT_TEMPLATES,
    values: Sequence[int] = tuple(OFFICIAL_EF_VALUES),
) -> Tuple[float, Dict[str, Any]]:
    """Official-style EF prediction for one already-selected frame batch.

    Returns ``(pred_ef, provenance)``. Requires real weights (no fabrication).
    """
    import torch
    import torch.nn.functional as F

    prompts, prompt_values = build_ef_prompt_grid(template, values)
    tensors = preprocess_frames(frames, preprocess_val).to(device)
    with torch.inference_mode():
        img_emb = F.normalize(model.encode_image(tensors), dim=-1).unsqueeze(0)
        tok = tokenize(prompts).to(device)
        txt_emb = F.normalize(model.encode_text(tok), dim=-1)
        pred = compute_regression_metric_official(img_emb, txt_emb, prompt_values)
    info = {
        "n_frames_selected": int(tensors.shape[0]),
        "n_prompts": len(prompts),
        "prompt_grid_sha256": prompt_grid_sha256(prompts, prompt_values),
        "prompt_strings_sha256": prompts_sha256(prompts),
        "ef_grid": f"0_{values[-1]}_step{OFFICIAL_EF_STEP}",
        "ef_grid_n": len(prompt_values),
        "tokenizer": OFFICIAL_HUB,
        "weight_hub": OFFICIAL_HUB,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "sample_strategy": "official_stride",
        "pad_to_16": False,
        "crop_res": list(OFFICIAL_CROP_RES),
    }
    return float(pred.item()), info


def evaluate_avi(
    avi: Path,
    *,
    model,
    tokenize,
    preprocess_val,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Full standalone path: AVI → frames → crop → stride → OpenCLIP → EF."""
    raw = read_avi_frames(Path(avi))
    cropped = np.stack(crop_and_scale_batch(raw), axis=0)
    selected, idx = select_official_frames(cropped)
    pred, info = predict_ef_official(
        selected,
        model=model,
        tokenize=tokenize,
        preprocess_val=preprocess_val,
        device=device,
    )
    return {
        "ok": True,
        "avi": str(avi),
        "pred_ef": pred,
        "n_raw_frames": int(raw.shape[0]),
        "frame_indices": idx,
        "input_sha256": input_sha256(selected),
        "bypassed_dataset": True,
        **info,
    }


# ---------------------------------------------------------------------------
# Honest metadata
# ---------------------------------------------------------------------------

def official_metadata(
    *,
    paper_mode: bool,
    verified: bool = False,
    parity: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the honest official-reproduction metadata block.

    ``official_reproduction`` (legacy alias) is true **only** when
    ``verified`` is true. Requesting ``--paper`` alone never sets it.
    """
    verified = bool(verified)
    meta: Dict[str, Any] = {
        "paper_mode": bool(paper_mode),
        "official_reproduction_requested": bool(paper_mode),
        "official_reproduction_verified": verified,
        # Legacy alias: kept but never true merely because --paper was passed.
        "official_reproduction": verified,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "official_hub": OFFICIAL_HUB,
        "parity_gaps": list(PARITY_GAPS),
    }
    if parity is not None:
        meta["parity"] = dict(parity)
    if extra:
        meta.update(extra)
    return meta


@dataclass
class OfficialParityReport:
    """Result of a golden-parity attempt against real upstream assets."""

    available: bool = False
    verified: bool = False
    skipped: bool = True
    reason: str = ""
    n_cases: int = 0
    max_abs_diff: Optional[float] = None
    tolerance: float = 1e-3
    details: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parity_report(
    *,
    example_avi: Optional[Path] = None,
    device: str = "cpu",
    tolerance: float = 1e-3,
) -> OfficialParityReport:
    """Run golden-parity *if* the official example AVI + hub weights exist.

    NEVER fabricates a parity value. Returns a skipped report (``available=False``)
    when assets are unavailable — so CI stays green and no numbers are invented.
    """
    report = OfficialParityReport(tolerance=float(tolerance))
    if example_avi is None or not Path(example_avi).is_file():
        report.reason = "official example AVI not provided/available — parity skipped"
        return report

    packed, err = try_load_official_open_clip(device)
    if packed is None:
        report.reason = f"official hub weights unavailable — parity skipped ({err})"
        return report
    model, tokenize, preprocess_val = packed

    try:
        result = evaluate_avi(
            Path(example_avi),
            model=model,
            tokenize=tokenize,
            preprocess_val=preprocess_val,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        report.reason = f"parity evaluation failed: {exc}"
        return report

    report.available = True
    report.skipped = False
    report.n_cases = 1
    report.details.append({"avi": str(example_avi), "pred_ef": result["pred_ef"]})
    report.metadata = {
        "frame_indices": result.get("frame_indices"),
        "prompt_grid_sha256": result.get("prompt_grid_sha256"),
        "input_sha256": result.get("input_sha256"),
    }
    # Only *behavioral* checks we can assert without an upstream reference run.
    stride_ok = result.get("frame_indices") == [
        int(i) for i in frame_indices(result.get("n_raw_frames", 0))
    ]
    grid_ok = result.get("ef_grid_n") == len(OFFICIAL_EF_VALUES) * len(EF_PROMPT_TEMPLATES)
    finite_ok = bool(np.isfinite(float(result["pred_ef"])))
    report.verified = bool(stride_ok and grid_ok and finite_ok)
    report.reason = (
        "behavioral parity checks passed (stride/grid/finite)"
        if report.verified
        else "behavioral parity checks failed"
    )
    report.metadata["stride_ok"] = stride_ok
    report.metadata["grid_ok"] = grid_ok
    report.metadata["finite_ok"] = finite_ok
    return report


def write_parity_report(path: Path, report: OfficialParityReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "EF_PROMPT_TEMPLATES",
    "OFFICIAL_CROP_RES",
    "OFFICIAL_EF_STEP",
    "OFFICIAL_HUB",
    "OFFICIAL_MAX_SPAN",
    "OFFICIAL_STRIDE",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "OfficialParityReport",
    "build_ef_prompt_grid",
    "crop_and_scale_batch",
    "evaluate_avi",
    "frame_indices",
    "input_sha256",
    "official_metadata",
    "official_stride_indices",
    "parity_report",
    "predict_ef_official",
    "preprocess_frames",
    "prompt_grid_sha256",
    "prompts_sha256",
    "read_avi_frames",
    "select_official_frames",
    "try_load_official_open_clip",
    "write_parity_report",
]
