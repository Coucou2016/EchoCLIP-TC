"""Map EchoNet-style measurements to official EchoCLIP prompt sentences.

Captions are filled from ``echoclip.prompts.ZERO_SHOT_PROMPTS`` only.
Dilation *binning* from EDV (mL) is a documented heuristic so public datasets
without free-text reports can still use the official qualitative templates.
No invented clinical phrasing.
"""

from typing import Dict, List, Optional, Sequence

from echoclip.prompts import ZERO_SHOT_PROMPTS

# Absolute LV EDV (mL) → official dilation prompt key.
# Heuristic for EchoNet-Dynamic (volumes are not BSA-indexed). Sentences
# themselves are unchanged official templates. Below 150 mL: no dilation caption.
EDV_DILATION_BINS = (
    (150.0, 200.0, "mild_left_ventricle_dilation"),
    (200.0, 250.0, "moderate_left_ventricle_dilation"),
    (250.0, float("inf"), "severe_left_ventricle_dilation"),
)


def fill_numeric_template(template: str, value: float) -> str:
    """Replace ``<#>`` with a rounded integer, matching EchoTokenizer.fill_prompt."""
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
    include_dilation: bool = True,
) -> List[str]:
    """
    One or more official-style captions.

    * EF → both ``ejection_fraction`` templates.
    * EDV → mild/moderate/severe LV dilation templates when above 150 mL.
    * ESV is accepted for API completeness but has no official numeric template,
      so it is not verbalized.
    """
    del esv  # no official ESV sentence in prompts.py
    captions: List[str] = []
    if ef is not None:
        for template in ZERO_SHOT_PROMPTS["ejection_fraction"]:
            captions.append(fill_numeric_template(template, ef))
    if include_dilation and edv is not None:
        key = dilation_prompt_key(edv)
        if key:
            captions.extend(list(ZERO_SHOT_PROMPTS[key]))
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
    include_dilation: bool = True,
) -> str:
    """Single ``text`` field for a DATA.md-compatible manifest pair."""
    return join_captions(
        captions_from_measurements(
            ef=ef, edv=edv, esv=esv, include_dilation=include_dilation
        )
    )


def pair_record(
    image: str,
    ef: Optional[float] = None,
    edv: Optional[float] = None,
    esv: Optional[float] = None,
    extra: Optional[Dict] = None,
    include_dilation: bool = True,
) -> Dict:
    """Manifest dict with required image/text plus optional clinical fields."""
    captions = captions_from_measurements(
        ef=ef, edv=edv, esv=esv, include_dilation=include_dilation
    )
    text = join_captions(captions)
    if not text:
        raise ValueError(f"No official caption could be built for {image!r} (need EF)")
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
