"""Print the rebuilt ablation numbers next to the values in the manuscript."""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import results_dir  # noqa: E402

RES = results_dir()


def runs(exp, tag=None):
    out = []
    for f in sorted(RES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d, list) or d.get("experiment") != exp:
            continue
        if (d.get("config") or {}).get("tag") != tag:
            continue
        out.append(d)
    return out


def ms(rs, key):
    v = [r["metrics"][key] for r in rs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def show(label, exp, tag, paper=None):
    rs = runs(exp, tag)
    if not rs:
        print(f"{label:<22}{'missing':>10}")
        return
    m, s = ms(rs, "mae")
    a, _ = ms(rs, "auc_ef_lt50")
    tail = f"   paper {paper:.2f}" if paper is not None else ""
    print(f"{label:<22}{m:8.2f} ± {s:4.2f}{a:9.3f}  n_seed={len(rs)}{tail}")


print("frame count (R4)")
show("8 frames", "R4", "abframes8", 6.10)
show("16 frames", "R4", "abframes16", 6.10)
show("32 frames (main)", "R4", None, 5.88)
print("\npooling")
show("CLS (main)", "R4", None, 5.88)
show("sequence mean", "R4_MEAN", "abpoolmean", 5.96)
print("\ntrain size")
show("500", "R4", "abscale500", 6.06)
show("1000", "R4", "abscale1000", 5.89)
show("2000", "R4", "abscale2000", 5.88)
show("4000", "R4", "abscale4000")
print("\nzero-shot top-k")
show("10%", "R1", "abtop10", 10.90)
show("20% (main)", "R1", None, 11.03)
show("30%", "R1", "abtop30", 10.69)
print("\ncalibration")
show("R5", "R5", None, 5.87)
show("R6", "R6", None, 5.87)
show("R5 tag abcalib", "R5", "abcalib")
show("R6 tag abcalib", "R6", "abcalib")
print("\ncontrastive pretraining on annotated ED/ES frames")
show("R5_EDES", "R5_EDES", "edes", 5.85)
