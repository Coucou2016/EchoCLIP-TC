"""Map EchoNet-style measurements to EchoCLIP / EchoCLIP-TA caption sentences.

Default captions fill from ``echoclip.prompts.ZERO_SHOT_PROMPTS`` (upstream-
attributed) so R0 zero-shot stays comparable. Clean-room TA templates live in
``echoclip.prompts_ta`` and activate via ``style="ta"`` or env
``ECHOCLIP_TA_CAPTIONS=1``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from echoclip.prompts import ZERO_SHOT_PROMPTS
from echoclip.prompts_ta import TA_CAPTIONS

# Absolute LV EDV (mL) → dilation prompt key.
# Heuristic for EchoNet-Dynamic (volumes are not BSA-indexed). Below 150 mL: none.
EDV_DILATION_BINS = (
    (150.0, 200.0, "mild_left_ventricle_dilation"),
    (200.0, 250.0, "moderate_left_ventricle_dilation"),
    (250.0, float("inf"), "severe_left_ventricle_dilation"),
)


def _prompt_bank(style: str):
    s = (style or "official").strip().lower()
    if s in ("ta", "clean", "cleanroom", "echoclip_ta"):
        return TA_CAPTIONS
    if os.environ.get("ECHOCLIP_TA_CAPTIONS", "").strip() in ("1", "true", "yes"):
        return TA_CAPTIONS
    return ZERO_SHOT_PROMPTS


def fill_numeric_template(template: str, value: float) -> str:
    """Replace ``<#>`` with a rounded integer."""
    rounded = int(round(float(value)))
    return template.replace("<#>", str(rounded))


def dilation_prompt_key(edv_ml: float) -> Optional[str]:
    edv = float(edv_ml)
    for lo, hi, key in EDV_DILATION_BINS:
        if lo <= edv < hi:
            return key
    return None


def captions_from_measurements(
    ef: Optional[float] = None,
    edv: Optional[float] = None,
    esv: Optional[float] = None,
    include_dilation: bool = False,
    style: str = "official",
) -> List[str]:
    """
    One or more caption sentences.

    * EF → ejection_fraction templates (primary structured captions).
    * EDV → mild/moderate/severe LV dilation templates when ``include_dilation``
      and EDV is above 150 mL (optional ablation only).
    * ESV is accepted for API completeness but has no numeric template.
    """
    del esv
    bank = _prompt_bank(style)
    captions: List[str] = []
    if ef is not None:
        for template in bank["ejection_fraction"]:
            captions.append(fill_numeric_template(template, ef))
    if include_dilation and edv is not None:
        key = dilation_prompt_key(edv)
        if key and key in bank:
            captions.extend(list(bank[key]))
    return captions


def join_captions(captions: Sequence[str]) -> str:
    parts = [c.strip() for c in captions if str(c).strip()]
    if not parts:
        return ""
    text = " ".join(parts)
    if not text.endswith(" "):
        text += " "
    return text


def measurements_to_text(
    ef: Optional[float] = None,
    edv: Optional[float] = None,
    esv: Optional[float] = None,
    include_dilation: bool = False,
    style: str = "official",
) -> str:
    """Single ``text`` field for a DATA.md-compatible manifest pair."""
    return join_captions(
        captions_from_measurements(
            ef=ef, edv=edv, esv=esv, include_dilation=include_dilation, style=style
        )
    )


def pair_record(
    image: str,
    ef: Optional[float] = None,
    edv: Optional[float] = None,
    esv: Optional[float] = None,
    extra: Optional[Dict] = None,
    include_dilation: bool = False,
    style: str = "official",
) -> Dict:
    """Manifest dict with required image/text plus optional clinical fields."""
    captions = captions_from_measurements(
        ef=ef, edv=edv, esv=esv, include_dilation=include_dilation, style=style
    )
    text = join_captions(captions)
    if not text:
        raise ValueError(f"No caption could be built for {image!r} (need EF)")
    rec = {"image": image, "text": text, "captions": captions}
    if ef is not None:
        rec["ef"] = float(ef)
    if edv is not None:
        rec["edv"] = float(edv)
    if esv is not None:
        rec["esv"] = float(esv)
    if extra:
        rec.update(extra)
    return rec
