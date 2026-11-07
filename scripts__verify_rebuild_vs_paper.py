"""Compare freshly rebuilt results against the numbers printed in the manuscript.

The manuscript DOCX survived the loss of the F: volume, so its Table 3 and
Table 5 cells are the reference values. Every rebuilt run must reproduce them
within the tolerances stated here before the manuscript can be called verified
again.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _paths import papers_dir, results_dir  # noqa: E402

RES = results_dir()
DOCX = papers_dir() / "EchoCLIP-TA_中文稿.docx"

# Tolerances: rounding of the printed value plus seed-to-seed spread allowed
# for the three-seed rows.
TOL_MAE = 0.02
TOL_AUC = 0.005


def cell_num(text: str):
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for p in sorted(RES.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "metrics" not in d:
            continue
        exp = d.get("experiment")
        if not exp:
            continue
        d["_file"] = p.name
        runs.setdefault(exp, []).append(d)
    return runs


def main() -> int:
    runs = load_runs()
    doc = docx.Document(str(DOCX))
    problems: list[str] = []
    checked = 0

    print(f"{'row':<14}{'metric':<8}{'paper':>8}{'rebuilt':>10}{'delta':>9}  status")
    print("-" * 60)

    t3 = None
    for t in doc.tables:
        head = [c.text.strip() for c in t.rows[0].cells]
        if head[:4] == ["编号", "方法", "种子", "MAE"]:
            t3 = t
            break
    if t3 is None:
        print("table 3 not found in the manuscript")
        return 1

    for row in t3.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        exp = cells[0]
        rs = [r for r in runs.get(exp, []) if not (r.get("config") or {}).get("tag")]
        if not rs:
            problems.append(f"{exp}: no rebuilt run")
            print(f"{exp:<14}{'--':<8}{'':>8}{'MISSING':>10}")
            continue
        for label, col, field, tol in (
            ("MAE", 3, "mae", TOL_MAE),
            ("AUC50", 9, "auc_ef_lt50", TOL_AUC),
        ):
            shown = cell_num(cells[col])
            got = st.mean([r["metrics"][field] for r in rs])
            if shown is None:
                continue
            delta = got - shown
            ok = abs(delta) <= tol
            checked += 1
            print(
                f"{exp:<14}{label:<8}{shown:>8.3f}{got:>10.3f}{delta:>+9.3f}  "
                f"{'ok' if ok else 'MISMATCH'}"
            )
            if not ok:
                problems.append(f"{exp}.{label}: paper {shown} rebuilt {got:.4f}")

    print("-" * 60)
    print(f"cells checked: {checked}")
    if problems:
        print("PROBLEMS")
        for p in problems:
            print(" -", p)
        return 1
    print("rebuilt results reproduce the manuscript tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
