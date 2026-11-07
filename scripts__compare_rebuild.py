"""Compare regenerated results against the values recorded before the volume loss.

The pre-loss numbers are the ones quoted in the manuscript, so this is the
check that decides whether the rebuild reproduced the paper or diverged.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import results_dir  # noqa: E402

# manuscript values (three-seed mean for R2-R6, single run otherwise)
PAPER = {
    "R0": (10.99, 0.740),
    "R1": (11.03, 0.738),
    "R1U16": (11.00, 0.740),
    "ORACLE_EDES": (10.95, 0.734),
    "R2": (6.47, 0.858),
    "R3": (6.30, 0.862),
    "R4": (5.88, 0.877),
    "R5": (5.87, 0.879),
    "R6": (5.87, 0.879),
}

RES = results_dir()
main = re.compile(r"^(R[0-9]|R1U16|ORACLE_EDES)_s\d+_f\d+_cls(_a10_20)?\.json$")

groups: dict[str, list[dict]] = {}
for p in sorted(RES.glob("*.json")):
    if not main.match(p.name):
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    groups.setdefault(d["experiment"], []).append(d)

print(f"{'exp':<13}{'seeds':>6}{'MAE now':>9}{'MAE paper':>11}{'d':>8}"
      f"{'AUC now':>9}{'AUC paper':>11}{'d':>8}")
bad = []
for exp in ("R0", "R1", "R1U16", "ORACLE_EDES", "R2", "R3", "R4", "R5", "R6"):
    runs = groups.get(exp)
    if not runs:
        print(f"{exp:<13}{'-':>6}  not rebuilt yet")
        continue
    mae = sum(r["metrics"]["mae"] for r in runs) / len(runs)
    auc = sum(r["metrics"]["auc_ef_lt50"] for r in runs) / len(runs)
    pmae, pauc = PAPER[exp]
    dmae, dauc = mae - pmae, auc - pauc
    print(f"{exp:<13}{len(runs):>6}{mae:>9.3f}{pmae:>11.2f}{dmae:>+8.3f}"
          f"{auc:>9.4f}{pauc:>11.3f}{dauc:>+8.4f}")
    if abs(dmae) > 0.05 or abs(dauc) > 0.005:
        bad.append(exp)

print()
if bad:
    print("DIVERGED:", ", ".join(bad))
    print("These differ from the manuscript by more than rounding.")
else:
    print("all rebuilt values match the manuscript within rounding")
