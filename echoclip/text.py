"""Clinical report cleaning and CLIP-style tokenization (GPT-2 BPE, 77 tokens)."""

import re
from typing import List, Tuple

import torch
from transformers import CLIPTokenizer

# Report cleaning regexes (from echonet/echo_CLIP utils.py)
_REMOVABLES = re.compile(r"\^|CRLF|‡")
_IN_TEXT_PERIODS = re.compile(r"(?<=\D)\.|\.(?=\D)")
_SQUARE_BRACKETS = re.compile(r"[\[\]]")
_MULTI_WHITESPACE = re.compile(r"\s+")
_MULTI_PERIOD = re.compile(r"\.+")
_SELECT_WAS = re.compile(r"(?<=\b)WAS(?=\b)")
_SELECT_WERE = re.compile(r"(?<=\b)WERE(?=\b)")
_SELECT_AND_OR = re.compile(r"(?<=\b)AND/OR(?=\b)")
_SELECT_NORMALLY = re.compile(r"NORMALLY")
_SELECT_MILDLY = re.compile(r"MILDLY")
_SELECT_MODERATELY = re.compile(r"MODERATELY")
_SELECT_SEVERELY = re.compile(r"SEVERELY")
_SELECT_PA = re.compile(r"PULMONARY ARTERY")
_SELECT_ICD = re.compile(r"[A-Z](\d+\.\d*\b)")
_SELECT_SLASH_DATES = re.compile(r"\d{2}/\d{2}/\d{4}")
_SELECT_DOT_DATES = re.compile(r"\d{2}\.\d{2}\.\d{4}")
_SPACE_BEFORE_UNIT = re.compile(r"\s+(MMHG|MM|CM|%)")
_SPACE_PERIOD = re.compile(r"\s\.")
_SPACE_PLUS = re.compile(r"\s\+\s")
_VERBOSE_PRESSURE = re.compile(r"\+CVPMMHG")

_ADD_PERIOD_RAW = [
    r"THE PEAK TRANSAORTIC GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSAORTIC GRADIENT IS <#>MMHG",
    r"LV EJECTION FRACTION IS <#>%",
    r"ESTIMATED PA PRESSURE IS <#>MMHG",
    r"RESTING SEGMENTAL WALL MOTION ANALYSIS",
    r"THE IVC DIAMETER IS <#>MM",
    r"EST RV/RA PRESSURE GRADIENT IS <#>MMHG",
    r"ESTIMATED PEAK RVSP IS <#>MMHG",
    r"ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG",
]
_SELECT_NUMBER = r"(?:\d+\.?\d*)"
_ADD_PERIOD = "|".join(
    f"(?:{re.escape(a).replace(re.escape('<#>'), _SELECT_NUMBER)})(?!\\.)"
    for a in _ADD_PERIOD_RAW
)
_ADD_PERIOD_RE = re.compile(f"({_ADD_PERIOD})")


def clean_report_text(text: str) -> str:
    """Normalize echocardiography report text before tokenization."""
    if len(text) <= 1:
        return text
    text = text.upper().strip().replace("`", "'")
    text = _REMOVABLES.sub("", text)
    text = _IN_TEXT_PERIODS.sub(". ", text)
    text = _SQUARE_BRACKETS.sub("", text)
    text = _SELECT_WAS.sub("IS", text)
    text = _SELECT_WERE.sub("ARE", text)
    text = _SELECT_AND_OR.sub("AND", text)
    text = _SELECT_NORMALLY.sub("NORMAL", text)
    text = _SELECT_MILDLY.sub("MILD", text)
    text = _SELECT_MODERATELY.sub("MODERATE", text)
    text = _SELECT_SEVERELY.sub("SEVERE", text)
    text = _SELECT_PA.sub("PA", text)
    text = _SELECT_SLASH_DATES.sub("", text)
    text = _SELECT_DOT_DATES.sub("", text)
    text = _SELECT_ICD.sub("", text)
    text = _SPACE_BEFORE_UNIT.sub(r"\1", text)
    text = _SPACE_PERIOD.sub(".", text)
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _SPACE_PLUS.sub("+", text)
    text = _VERBOSE_PRESSURE.sub("MMHG", text)
    text = text.strip() + " "
    text = _ADD_PERIOD_RE.sub(r"\1.", text)
    text = _MULTI_PERIOD.sub(".", text)
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
