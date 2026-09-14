"""Clean-room EF caption templates for EchoCLIP-TA structured training.

These strings are authored for this repository (MIT) and are **not** copies of
echonet/echo_CLIP ``prompts_used.json``. For zero-shot parity with published
EchoCLIP, use ``echoclip.prompts.ZERO_SHOT_PROMPTS`` (upstream-attributed).

Default R5 structured captions still use official templates via
``structured_text`` so B0/R0 remain comparable; switch with
``ECHOCLIP_TA_CAPTIONS=1`` or ``captions_from_measurements(..., style="ta")``.
"""

from __future__ import annotations

from typing import Dict, List

# MIT clean-room templates — EF / dilation only (no device / RAP blocks).
TA_CAPTIONS: Dict[str, List[str]] = {
    "ejection_fraction": [
        "LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED AT <#>% . ",
        "THE LV EJECTION FRACTION MEASURES <#>% . ",
    ],
    "severe_left_ventricle_dilation": [
        "THE LEFT VENTRICLE IS SEVERELY DILATED. ",
    ],
    "moderate_left_ventricle_dilation": [
        "THE LEFT VENTRICLE IS MODERATELY DILATED. ",
    ],
    "mild_left_ventricle_dilation": [
        "THE LEFT VENTRICLE IS MILDLY DILATED. ",
    ],
}
