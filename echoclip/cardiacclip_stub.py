"""CardiacCLIP comparison interface (stub).

CardiacCLIP (Du, Guo & Li, MICCAI 2025) is an external few-shot LVEF method.
This module documents the comparison contract and raises clearly when weights
are absent. **Do not invent MAE numbers.**

Upstream: https://github.com/xmed-lab/CardiacCLIP
Paper: arXiv:2509.17065
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


CARDIACCLIP_REQUIRES = (
    "External CardiacCLIP checkpoint + EchoNet-Dynamic (or matching eval set). "
    "Weights are not bundled in EchoCLIP-TC. Set CARDIACCLIP_WEIGHTS or pass "
    "weights_path to load_cardiacclip_interface()."
)


@dataclass
class CardiacCLIPCompareResult:
    """Placeholder result — never filled with invented clinical metrics."""

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
            "clinical_numbers": "待补充" if not self.available else self.extra.get("status"),
        }


def resolve_weights_path(weights_path: Optional[Path] = None) -> Optional[Path]:
    import os

    if weights_path is not None:
        p = Path(weights_path)
        return p if p.exists() else None
    env = os.environ.get("CARDIACCLIP_WEIGHTS", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    return None


def load_cardiacclip_interface(
    weights_path: Optional[Path] = None,
) -> CardiacCLIPCompareResult:
    """Return a comparison handle. Hard-fails availability without external weights."""
    path = resolve_weights_path(weights_path)
    if path is None:
        return CardiacCLIPCompareResult(
            available=False,
            note=CARDIACCLIP_REQUIRES,
            load_source="missing_weights",
        )
    # Interface only — actual forward pass lives in upstream CardiacCLIP code.
    return CardiacCLIPCompareResult(
        available=True,
        mae=None,
        rmse=None,
        note=(
            "Weights path found, but EchoCLIP-TC does not ship CardiacCLIP inference. "
            "Run upstream eval and paste metrics; do not invent numbers."
        ),
        load_source=f"external:{path}",
        extra={"weights_path": str(path), "status": "待补充"},
    )


def comparison_table_row(
    result: Optional[CardiacCLIPCompareResult] = None,
) -> Dict[str, Any]:
    """Row suitable for protocol comparison tables."""
    r = result or load_cardiacclip_interface()
    row = r.to_dict()
    row["protocol_note"] = (
        "CardiacCLIP is a literature / external-weights comparator, not R0–R6."
    )
    return row


def document_gaps() -> List[str]:
    return [
        "CardiacCLIP weights not redistributed (Academic / upstream license).",
        "Few-shot protocol differs from EchoCLIP-TC locked R0–R6 matrix.",
        "No invented ΔMAE vs CardiacCLIP without both running on the same EchoNet split.",
    ]
