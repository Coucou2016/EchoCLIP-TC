"""Weight load verification for the audit document.

Uses the same construction path as scripts/extract_frames_embeddings.py so the
verification exercises the code that actually produced the embeddings.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

if "lzma" not in sys.modules:
    fake = types.ModuleType("lzma")

    class LZMAError(Exception):
        pass

    fake.LZMAError = LZMAError
    fake.open = lambda *a, **k: (_ for _ in ()).throw(LZMAError("lzma unavailable"))
    sys.modules["lzma"] = fake
    sys.modules["_lzma"] = fake

import open_clip  # noqa: E402
import torch  # noqa: E402

W = Path(r"E:\Projects\20260522-EchoCLIP\models\echo-clip")
BIN = W / "open_clip_pytorch_model.bin"

raw = json.loads((W / "open_clip_config.json").read_text(encoding="utf-8"))
cfg = raw["model_cfg"]

t0 = time.time()
model = open_clip.create_model(
    cfg["vision_cfg"]["timm_model_name"],
    pretrained=None,
    precision="fp32",
)
sd = torch.load(BIN, map_location="cpu")
report = model.load_state_dict(sd, strict=False)

h = hashlib.sha256()
with open(BIN, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)

def _nparams(module) -> int:
    if isinstance(module, torch.nn.Parameter):
        return module.numel()
    return sum(p.numel() for p in module.parameters())


has_visual = hasattr(model, "visual")
# OpenCLIP 3.3.0 exposes the text tower as separate top-level modules
# (transformer, token_embedding, ln_final, text_projection) instead of a
# single ``.text`` submodule, so a bare hasattr(model, "text") is misleading.
TEXT_MODULES = ("transformer", "token_embedding", "ln_final", "text_projection")
text_parts = [getattr(model, m) for m in TEXT_MODULES if hasattr(model, m)]
has_text = bool(text_parts)
params_text = sum(_nparams(m) for m in text_parts)
params_visual = _nparams(model.visual) if has_visual else 0
tot = _nparams(model)
residual = tot - params_visual - params_text

lines = [
    f"model_class         = {type(model).__name__}",
    f"has_visual_tower    = {has_visual}",
    f"has_text_tower      = {has_text} (modules: {', '.join(m for m in TEXT_MODULES if hasattr(model, m))})",
    f"weights_path        = {BIN}",
    f"weights_bytes       = {BIN.stat().st_size}",
    f"weights_sha256      = {h.hexdigest()}",
    f"config_path         = {W / 'open_clip_config.json'}",
    f"config_sha256       = {hashlib.sha256((W / 'open_clip_config.json').read_bytes()).hexdigest()}",
    f"torch               = {torch.__version__}",
    f"open_clip           = {open_clip.__version__}",
    f"python              = {sys.version.split()[0]}",
    f"construction        = open_clip.create_model({cfg['vision_cfg']['timm_model_name']!r}, pretrained=None)",
    f"state_dict_entries  = {len(sd)}",
    f"missing_keys        = {len(report.missing_keys)} {list(report.missing_keys)[:8]}",
    f"unexpected_keys     = {len(report.unexpected_keys)} {list(report.unexpected_keys)[:8]}",
    f"params_total        = {tot:,}",
    f"params_visual       = {params_visual:,}",
    f"params_text         = {params_text:,}",
    f"params_other        = {residual:,} (logit_scale, positional and projection leaves outside the two towers)",
    f"image_size          = {cfg['vision_cfg']['image_size']}",
    f"embed_dim           = {cfg.get('embed_dim')}",
    f"text_width          = {cfg['text_cfg'].get('width')}",
    f"text_layers         = {cfg['text_cfg'].get('layers')}",
    f"context_length      = {cfg['text_cfg'].get('context_length')}",
    f"load_seconds        = {time.time() - t0:.1f}",
    f"timestamp           = {time.strftime('%Y-%m-%dT%H:%M:%S')}",
]

out = Path(__file__).resolve().parent
sys.path.insert(0, str(out))
from _paths import logs_dir  # noqa: E402

out = logs_dir() / "weight_check.log"
out.parent.mkdir(parents=True, exist_ok=True)
text = "\n".join(lines) + "\n"
out.write_text(text, encoding="utf-8")
print(text)

ok = not report.missing_keys and not report.unexpected_keys
print(f"VERDICT: {'PASS' if ok else 'FAIL'} (strict state_dict match)")
raise SystemExit(0 if ok else 1)
