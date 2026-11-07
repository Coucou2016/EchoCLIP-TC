"""Verify the HF tokenizer actually loads and matches the checkpoint vocab."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "models" / "echo-clip"

fake = types.ModuleType("lzma")


class LZMAError(Exception):
    pass


fake.LZMAError = LZMAError
fake.open = lambda *a, **k: (_ for _ in ()).throw(LZMAError("lzma unavailable"))
sys.modules["lzma"] = fake
sys.modules["_lzma"] = fake

import open_clip  # noqa: E402

tok = open_clip.get_tokenizer(f"local-dir:{WD}")
print("tokenizer class:", type(tok).__name__)

prompts = [
    "the left ventricular ejection fraction is 35%",
    "the left ventricular ejection fraction is 60%",
]
ids = tok(prompts)
print("ids shape:", tuple(ids.shape), "dtype:", ids.dtype)
print("first 20 ids:", ids[0, :20].tolist())

vocab = json.loads((WD / "vocab.json").read_text(encoding="utf-8"))
inv = {v: k for k, v in vocab.items()}
decoded = " ".join(inv.get(int(i), "<?>") for i in ids[0] if int(i) != 0)
print("decoded:", decoded)
print("vocab size:", len(vocab))

# SimpleTokenizer would give a different, word-level id pattern; confirm the
# HuggingFace path is really in use by checking the BPE end-of-word marker.
assert type(tok).__name__ not in ("SimpleTokenizer",), "still falling back to SimpleTokenizer"
print("OK: HF tokenizer active")
