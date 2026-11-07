"""Bitwise proof that EchoCLIP text-tower tensors were really loaded."""

from __future__ import annotations

import json
import sys
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
cfg = json.loads((W / "open_clip_config.json").read_text(encoding="utf-8"))["model_cfg"]
model = open_clip.create_model(
    cfg["vision_cfg"]["timm_model_name"], pretrained=None, precision="fp32"
)
model.eval()
sd = torch.load(W / "open_clip_pytorch_model.bin", map_location="cpu")
report = model.load_state_dict(sd, strict=False)
print(f"load_report strict=False -> missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")

# Compare every checkpoint tensor against the live model by dotted path.
live = dict(model.named_parameters())
live.update(dict(model.named_buffers()))

identical = 0
differing = []
missing_in_model = []
for key, ref in sd.items():
    if key not in live:
        missing_in_model.append(key)
        continue
    got = live[key].detach().cpu().float()
    exp = ref.detach().cpu().float()
    if got.shape != exp.shape:
        differing.append((key, "shape"))
        continue
    if torch.equal(got, exp):
        identical += 1
    else:
        differing.append((key, float((got - exp).abs().max())))

groups = {
    "visual": [k for k in sd if k.startswith("visual.")],
    "transformer": [k for k in sd if k.startswith("transformer.")],
    "token_embedding": [k for k in sd if k.startswith("token_embedding")],
    "text_projection": [k for k in sd if k.startswith("text_projection")],
    "positional_embedding": [k for k in sd if k.startswith("positional_embedding")],
    "ln_final": [k for k in sd if k.startswith("ln_final")],
}

lines = [
    f"checkpoint_tensors      = {len(sd)}",
    f"bitwise_identical       = {identical}",
    f"differing_tensors       = {len(differing)} {differing[:3]}",
    f"absent_from_model       = {len(missing_in_model)} {missing_in_model[:3]}",
]
for gname, keys in groups.items():
    gid = sum(
        1
        for k in keys
        if k in live and torch.equal(live[k].detach().cpu().float(), sd[k].float())
    )
    lines.append(f"{gname:<22} = {gid}/{len(keys)} identical")

# Numerical proof of life for the text tower.
tok = open_clip.get_tokenizer("hf-hub:mkaichristensen/echo-clip")
prompts = ["LV EJECTION FRACTION IS 20%", "LV EJECTION FRACTION IS 60%"]
with torch.inference_mode():
    a, b = model.encode_text(tok(prompts))
cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
lines.append(f"text_embed_dim          = {tuple(a.shape)}")
lines.append(f"cos_sim_20_vs_60        = {cos:.6f}")

out = Path(__file__).resolve().parent
sys.path.insert(0, str(out))
from _paths import logs_dir  # noqa: E402

out = logs_dir() / "text_tower_check.log"
text = "\n".join(lines) + "\n"
out.write_text(text, encoding="utf-8")
print(text)

all_ok = identical == len(sd) and not differing and not missing_in_model
print("VERDICT:", "PASS" if all_ok else "FAIL")
raise SystemExit(0 if all_ok else 1)
