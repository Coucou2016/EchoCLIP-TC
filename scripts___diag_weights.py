"""Diagnose the EchoCLIP checkpoint load: architecture, state-dict match, tokenizer."""

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
import torch  # noqa: E402

print("open_clip", getattr(open_clip, "__version__", "?"))
cfg = json.loads((WD / "open_clip_config.json").read_text(encoding="utf-8"))
print("config top keys:", list(cfg.keys()))
print("model_cfg:", json.dumps(cfg.get("model_cfg"), ensure_ascii=False)[:600])

mc = cfg["model_cfg"]
arch = mc["vision_cfg"]["timm_model_name"]
print("timm_model_name ->", arch)

model = open_clip.create_model(arch, pretrained=None, precision="fp32")
print("model type:", type(model).__name__)
print("has encode_text:", hasattr(model, "encode_text"))
print("text cfg:", model.text_cfg if hasattr(model, "text_cfg") else None)

sd = torch.load(WD / "open_clip_pytorch_model.bin", map_location="cpu")
print("ckpt tensors:", len(sd), "dtype sample:", next(iter(sd.values())).dtype)
rep = model.load_state_dict(sd, strict=False)
print("missing:", len(rep.missing_keys), "unexpected:", len(rep.unexpected_keys))
if rep.missing_keys:
    print("  missing sample:", rep.missing_keys[:8])
if rep.unexpected_keys:
    print("  unexpected sample:", rep.unexpected_keys[:8])

tok = open_clip.get_tokenizer(str(WD))
print("tokenizer:", type(tok).__name__)
