"""Quick sanity check of the zero-shot path on the cached test500 embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import train_heads_from_cache as th  # noqa: E402

te = th.load_split("test500")
x = th.take_frames(te.emb, 32)
pred = th.zero_shot_predict(x, 0.2)
print("pred min/max/mean/sd", round(float(pred.min()), 2), round(float(pred.max()), 2),
      round(float(pred.mean()), 2), round(float(pred.std()), 2))
m = th.regression_metrics(te.ef, pred)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
