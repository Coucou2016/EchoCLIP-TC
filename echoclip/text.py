"""Clinical report cleaning and CLIP-style tokenization (GPT-2 BPE, 77 tokens).

Report normalization is a **clean-room** reimplementation for EchoCLIP-TA:
same functional goals as public echo report prep (uppercase, strip noise,
normalize severity/tense wording, collapse whitespace) but with independently
authored regexes. Official EchoCLIP prompt *strings* remain in ``prompts.py``
and are attributed separately (see ATTRIBUTION.md / NOTICE).
"""

from __future__ import annotations

import re
from typing import List

import torch
from transformers import CLIPTokenizer

# ---------------------------------------------------------------------------
# Clean-room report normalizer (EchoCLIP-TA)
# Intentional behavioral overlap with clinical echo report prep; patterns are
# rewritten here and are not copied from echonet/echo_CLIP utils.py.
# ---------------------------------------------------------------------------

_NOISE_CHARS = re.compile(r"[\^\u2021]|CRLF")
_PERIOD_OUTSIDE_DIGITS = re.compile(r"(?<=\D)\.(?=\D)|(?<=\D)\.$")
_BRACKETS = re.compile(r"[\[\]]")
_WS = re.compile(r"\s+")
_DOT_RUN = re.compile(r"\.{2,}")
# Whole-word tense / conjunction normalization
_WORD_WAS = re.compile(r"\bWAS\b")
_WORD_WERE = re.compile(r"\bWERE\b")
_WORD_AND_OR = re.compile(r"\bAND/OR\b")
# Severity / anatomy shorthand
_NORMALLY = re.compile(r"NORMALLY")
_MILDLY = re.compile(r"MILDLY")
_MODERATELY = re.compile(r"MODERATELY")
_SEVERELY = re.compile(r"SEVERELY")
_PULM_ARTERY = re.compile(r"PULMONARY ARTERY")
# Strip ICD-like codes and calendar dates (de-identification hygiene)
_ICDISH = re.compile(r"[A-Z]\d+\.\d*\b")
_DATE_SLASH = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_DATE_DOT = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
_UNIT_GAP = re.compile(r"\s+(MMHG|MM|CM|%)")
_SPACE_BEFORE_DOT = re.compile(r"\s+\.")
_PLUS_SPACED = re.compile(r"\s\+\s")
_CVP_TAG = re.compile(r"\+CVPMMHG")

# Measurement phrases that should end with a period when missing one.
_MEASUREMENT_STEMS = (
    "THE PEAK TRANSAORTIC GRADIENT IS {n}MMHG",
    "THE MEAN TRANSAORTIC GRADIENT IS {n}MMHG",
    "LV EJECTION FRACTION IS {n}%",
    "ESTIMATED PA PRESSURE IS {n}MMHG",
    "RESTING SEGMENTAL WALL MOTION ANALYSIS",
    "THE IVC DIAMETER IS {n}MM",
    "EST RV/RA PRESSURE GRADIENT IS {n}MMHG",
    "ESTIMATED PEAK RVSP IS {n}MMHG",
    "ESTIMATED PA SYSTOLIC PRESSURE IS {n}MMHG",
)
_NUM = r"(?:\d+\.?\d*)"
_ADD_PERIOD_PARTS = []
for stem in _MEASUREMENT_STEMS:
    if "{n}" in stem:
        lit = re.escape(stem).replace(re.escape("{n}"), _NUM)
    else:
        lit = re.escape(stem)
    _ADD_PERIOD_PARTS.append(f"(?:{lit})(?!\\.)")
_ENSURE_PERIOD = re.compile("(" + "|".join(_ADD_PERIOD_PARTS) + ")")


def clean_report_text(text: str) -> str:
    """Normalize echocardiography report text before tokenization (clean-room)."""
    if len(text) <= 1:
        return text
    text = text.upper().strip().replace("`", "'")
    text = _NOISE_CHARS.sub("", text)
    text = _PERIOD_OUTSIDE_DIGITS.sub(". ", text)
    text = _BRACKETS.sub("", text)
    text = _WORD_WAS.sub("IS", text)
    text = _WORD_WERE.sub("ARE", text)
    text = _WORD_AND_OR.sub("AND", text)
    text = _NORMALLY.sub("NORMAL", text)
    text = _MILDLY.sub("MILD", text)
    text = _MODERATELY.sub("MODERATE", text)
    text = _SEVERELY.sub("SEVERE", text)
    text = _PULM_ARTERY.sub("PA", text)
    text = _DATE_SLASH.sub("", text)
    text = _DATE_DOT.sub("", text)
    text = _ICDISH.sub("", text)
    text = _UNIT_GAP.sub(r"\1", text)
    text = _SPACE_BEFORE_DOT.sub(".", text)
    text = _WS.sub(" ", text)
    text = _PLUS_SPACED.sub("+", text)
    text = _CVP_TAG.sub("MMHG", text)
    text = text.strip() + " "
    text = _ENSURE_PERIOD.sub(r"\1.", text)
    text = _DOT_RUN.sub(".", text)
    return text


class EchoTokenizer:
    """GPT-2 BPE tokenizer with CLIP start/end tokens and fixed context length."""

    _shared = None

    def __init__(self, context_length: int = 77):
        self.context_length = context_length
        if EchoTokenizer._shared is None:
            EchoTokenizer._shared = self._load_clip_tokenizer()
        self._tokenizer = EchoTokenizer._shared

    @staticmethod
    def _load_clip_tokenizer():
        name = "openai/clip-vit-base-patch32"
        try:
            return CLIPTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            return CLIPTokenizer.from_pretrained(name)

    def encode(self, texts: List[str], clean: bool = True) -> torch.LongTensor:
        if clean:
            texts = [clean_report_text(t) for t in texts]
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].long()

    def fill_prompt(self, template: str, value: float) -> str:
        return template.replace("<#>", str(int(value) if float(value).is_integer() else value))
