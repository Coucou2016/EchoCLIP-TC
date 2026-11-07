"""Independent recomputation of the headline metrics from result JSONs.

Deliberately written without touching the manuscript so the numbers can be
compared against it. Only strict seed files are counted.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

_ROOTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOTS / "scripts"))
from _paths import results_dir  # noqa: E402

RES = results_dir()

SEED_RE = re.compile(r"^([A-Za-z0-9_]+)_s(\d+)_f\d+_cls(?:_a\d+_\d+)?\.json$")
EXPS = ["R0", "R1", "R1U16", "ORACLE_EDES", "R2", "R3", "R4", "R5", "R6"]

groups: dict[str, list[tuple]] = {}
skipped: list[str] = []

for f in sorted(RES.glob("*.json")):
    m = SEED_RE.match(f.name)
    if not m:
        skipped.append(f.name)
        continue
    exp = m.group(1)
    if exp not in EXPS:
        skipped.append(f.name)
        continue
    j = json.loads(f.read_text(encoding="utf-8"))
    met = j.get("metrics") or {}
    groups.setdefault(exp, []).append(
        (
            j.get("seed"),
            met.get("mae"),
            met.get("auc_ef_lt50"),
            met.get("rmse"),
            met.get("r2"),
            met.get("bias"),
            met.get("n"),
        )
    )

print(f"{'exp':<13}{'seeds':<13}{'MAE':<18}{'AUC50':<9}{'RMSE':<8}{'R2':<8}{'bias':<8}{'n'}")
print("-" * 86)
rows = {}
for exp in EXPS:
    g = groups.get(exp)
    if not g:
        print(f"{exp:<13}(no strict seed file)")
        continue
    maes = [x[1] for x in g if x[1] is not None]
    aucs = [x[2] for x in g if x[2] is not None]
    rmses = [x[3] for x in g if x[3] is not None]
    r2s = [x[4] for x in g if x[4] is not None]
    biases = [x[5] for x in g if x[5] is not None]
    n = g[0][6]
    sd = f" +- {st.stdev(maes):.3f}" if len(maes) > 1 else ""
    seeds = ",".join(str(x[0]) for x in g)
    print(
        f"{exp:<13}{seeds:<13}{st.mean(maes):.3f}{sd:<10}"
        f"{st.mean(aucs):<9.3f}{st.mean(rmses):<8.2f}{st.mean(r2s):<8.3f}{st.mean(biases):<8.2f}{n}"
    )
    rows[exp] = {
        "seeds": [x[0] for x in g],
        "mae": st.mean(maes),
        "mae_sd": st.stdev(maes) if len(maes) > 1 else None,
        "auc50": st.mean(aucs),
        "rmse": st.mean(rmses),
        "r2": st.mean(r2s),
        "bias": st.mean(biases),
        "n": n,
    }

out = RES / "_independent_recompute.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
print(f"skipped files (ablations, audits, refs): {len(skipped)}")
