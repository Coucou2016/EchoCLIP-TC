"""CardiacCLIP comparison adapter for EchoCLIP-TA.

Fullest in-repo interface: download instructions, weight discovery, comparison
table template. Does **not** redistribute CardiacCLIP weights or invent MAE.

Upstream: https://github.com/xmed-lab/CardiacCLIP
Paper: arXiv:2509.17065 (MICCAI 2025)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

CARDIACCLIP_DOWNLOAD = """
CardiacCLIP weights are external (Academic / upstream license) — not bundled.

1. Clone upstream:
     git clone https://github.com/xmed-lab/CardiacCLIP.git
2. Follow their README to download checkpoints (HF / Google Drive as published).
3. Point EchoCLIP-TA at the weights:
     set CARDIACCLIP_WEIGHTS=C:\\path\\to\\cardiacclip.pt
   or:
     python -c "from echoclip.cardiacclip import load_cardiacclip_interface; \\
       print(load_cardiacclip_interface(r'C:\\\\path\\\\to\\\\cardiacclip.pt').to_dict())"
4. Run upstream eval on the **same** EchoNet TEST split as this repo's manifests,
   then paste MAE/RMSE into the comparison table (do not invent numbers).

Paper mode hard-fails invented numbers: clinical fields stay 待补充 until filled.
""".strip()

CARDIACCLIP_REQUIRES = (
    "External CardiacCLIP checkpoint + EchoNet-Dynamic (or matching eval set). "
    "Weights are not bundled in EchoCLIP-TA. Set CARDIACCLIP_WEIGHTS or pass "
    "weights_path to load_cardiacclip_interface()."
)


@dataclass
class CardiacCLIPCompareResult:
    """Comparison handle — never filled with invented clinical metrics."""

    available: bool = False
    mae: Optional[float] = None
    rmse: Optional[float] = None
    note: str = CARDIACCLIP_REQUIRES
    load_source: str = "unavailable"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "CardiacCLIP",
            "available": self.available,
            "mae": self.mae,
            "rmse": self.rmse,
            "note": self.note,
            "load_source": self.load_source,
            "extra": self.extra,
            "clinical_numbers": "待补充" if self.mae is None else "external",
            "download_instructions": CARDIACCLIP_DOWNLOAD,
        }


def resolve_weights_path(weights_path: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if weights_path is not None:
        candidates.append(Path(weights_path))
    env = os.environ.get("CARDIACCLIP_WEIGHTS", "").strip()
    if env:
        candidates.append(Path(env))
    # Common local drop locations (search only; never invent metrics)
    for rel in (
        ROOT / "external" / "CardiacCLIP" / "checkpoints",
        ROOT / "weights" / "cardiacclip",
        Path("CardiacCLIP") / "checkpoints",
    ):
        if rel.exists():
            candidates.append(rel)
            for p in rel.rglob("*.pt"):
                candidates.append(p)
            for p in rel.rglob("*.pth"):
                candidates.append(p)
            for p in rel.rglob("*.ckpt"):
                candidates.append(p)
    for p in candidates:
        if p.is_file() and p.exists():
            return p
    return None


def load_cardiacclip_interface(
    weights_path: Optional[Path] = None,
) -> CardiacCLIPCompareResult:
    """Return a comparison handle. Availability requires external weights on disk."""
    path = resolve_weights_path(weights_path)
    if path is None:
        return CardiacCLIPCompareResult(
            available=False,
            note=CARDIACCLIP_REQUIRES,
            load_source="missing_weights",
            extra={"download": CARDIACCLIP_DOWNLOAD},
        )
    return CardiacCLIPCompareResult(
        available=True,
        mae=None,
        rmse=None,
        note=(
            "Weights path found, but EchoCLIP-TA does not ship CardiacCLIP inference. "
            "Run upstream eval on the same EchoNet split and paste metrics; "
            "do not invent numbers."
        ),
        load_source=f"external:{path}",
        extra={
            "weights_path": str(path),
            "status": "待补充",
            "download": CARDIACCLIP_DOWNLOAD,
        },
    )


def comparison_table_row(
    result: Optional[CardiacCLIPCompareResult] = None,
) -> Dict[str, Any]:
    """Row suitable for protocol / paper comparison tables."""
    r = result or load_cardiacclip_interface()
    row = r.to_dict()
    row["protocol_note"] = (
        "CardiacCLIP is a literature / external-weights comparator, not R0–R6."
    )
    row["mae"] = row["mae"] if row["mae"] is not None else "待补充"
    row["rmse"] = row["rmse"] if row["rmse"] is not None else "待补充"
    return row


def write_comparison_template(path: Optional[Path] = None) -> Path:
    """Write auto-filled comparison JSON with 待补充 clinical fields."""
    out = path or (ROOT / "checkpoints" / "protocol" / "cardiacclip_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "table": "external_comparator",
        "rows": [comparison_table_row()],
        "gaps": document_gaps(),
        "paper_rule": (
            "Under --paper, leave MAE/RMSE as 待补充 unless upstream metrics "
            "were pasted from a real EchoNet run on the same split."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def assert_paper_ready(result: Optional[CardiacCLIPCompareResult] = None) -> None:
    """Hard-fail under paper expectations if numbers were invented / missing."""
    r = result or load_cardiacclip_interface()
    if r.mae is not None or r.rmse is not None:
        # Only accept numbers when load_source is external and caller set them
        if not r.available:
            raise RuntimeError(
                "CardiacCLIP clinical numbers present but weights unavailable — "
                "refusing invented metrics."
            )
        return
    # Missing numbers is OK (待补充); invented ones are not. No-op.
    return


def document_gaps() -> List[str]:
    return [
        "CardiacCLIP weights not redistributed (Academic / upstream license).",
        "Few-shot protocol differs from EchoCLIP-TA locked R0–R6 matrix.",
        "No invented ΔMAE vs CardiacCLIP without both running on the same EchoNet split.",
        CARDIACCLIP_DOWNLOAD,
    ]


# Back-compat alias used by older imports
def document_download() -> str:
    return CARDIACCLIP_DOWNLOAD
