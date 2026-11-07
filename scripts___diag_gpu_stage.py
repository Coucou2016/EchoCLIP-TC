"""Isolate where the GPU stalls: model->cuda, then encode_text, then encode_image."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "models" / "echo-clip"

fake = types.ModuleType("lzma")


class LZMAError(Exception):
    pass


fake.LZMAError = LZMAError
fake.open = lambda *a, **k: (_ for _ in ()).throw(LZMAError("x"))
sys.modules["lzma"] = fake
sys.modules["_lzma"] = fake

import torch  # noqa: E402
import open_clip  # noqa: E402


def beat(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


beat("start")
m = open_clip.create_model("convnext_base", pretrained=None, precision="fp32")
sd = torch.load(WD / "open_clip_pytorch_model.bin", map_location="cpu")
m.load_state_dict(sd, strict=True)
m.eval()
beat("weights loaded")

tok = open_clip.get_tokenizer(f"local-dir:{WD}")
ids = tok([f"ejection fraction is {v}%" for v in range(101)])
beat(f"tokenized {tuple(ids.shape)}")

beat("moving to cuda ...")
m = m.to("cuda")
torch.cuda.synchronize()
beat("on cuda")

# count devices (GTX 950M laptop may have an iGPU listed too)
beat(f"device={torch.cuda.current_device()} "
     f"{torch.cuda.get_device_name(0)}")

beat("encode_text on cuda ...")
with torch.no_grad():
    e = m.encode_text(ids.cuda())
torch.cuda.synchronize()
beat(f"text ok {tuple(e.shape)}")

beat("encode_image on cuda ...")
with torch.no_grad():
    x = torch.randn(2, 3, 224, 224, device="cuda")
    v = m.encode_image(x)
torch.cuda.synchronize()
beat(f"image ok {tuple(v.shape)}")
beat("DONE")
