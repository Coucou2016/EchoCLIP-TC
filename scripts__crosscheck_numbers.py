"""Recompute every number shown in the manuscript tables directly from the
result JSONs and report any disagreement.

Run:  python scripts/crosscheck_numbers.py
Exit code 0 means every table cell agrees with its source JSON.
"""
from __future__ import annotations

import sys

import glob
import json
import re
from pathlib import Path

import numpy as np
from docx import Document

ROOT = Path(r"E:\Projects\20260522-EchoCLIP")
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import results_dir

RES = results_dir()
DOCX = ROOT / "papers" / "EchoCLIP-TA_中文稿.docx"


def load_runs():
    runs: dict[str, list[dict]] = {}
    for p in sorted(RES.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "metrics" not in d or "experiment" not in d:
            continue
        d["_file"] = p.name
        runs.setdefault(d["experiment"], []).append(d)
    return runs


def tagged(runs, exp, tag=None):
    out = []
    for r in runs.get(exp, []):
        t = (r.get("config") or {}).get("tag")
        if t == tag:
            out.append(r)
    return out


def main_runs(runs, exp):
    """Only the exact main-row files, no ablation tag."""
    out = []
    for r in runs.get(exp, []):
        cfg = r.get("config") or {}
        if cfg.get("tag"):
            continue
        out.append(r)
    return out


def cell_num(text: str):
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def main() -> int:
    runs = load_runs()
    doc = Document(str(DOCX))
    problems: list[str] = []
    checked = 0

    # ---- table 3, the headline table -----------------------------------
    t3 = None
    for t in doc.tables:
        head = [c.text.strip() for c in t.rows[0].cells]
        if head[:4] == ["编号", "方法", "种子", "MAE"]:
            t3 = t
            break
    if t3 is None:
        problems.append("table 3 not found")
    else:
        for row in t3.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            exp = cells[0]
            rs = main_runs(runs, exp)
            if not rs:
                problems.append(f"{exp}: no main-row JSON")
                continue
            for label, col, field, tol in (
                ("MAE", 3, "mae", 0.006),
                ("RMSE", 4, "rmse", 0.006),
                ("R2", 5, "r2", 0.0016),
                ("bias", 6, "bias", 0.006),
                ("pearson", 7, "pearson", 0.0016),
                ("spearman", 8, "spearman", 0.0016),
                ("AUC50", 9, "auc_ef_lt50", 0.0016),
            ):
                shown = cell_num(cells[col])
                if shown is None:
                    problems.append(f"{exp}.{label}: unparsable cell {cells[col]!r}")
                    continue
                vals = [r["metrics"][field] for r in rs]
                want = float(np.mean(vals))
                checked += 1
                if abs(shown - want) > tol:
                    problems.append(f"{exp}.{label}: docx {shown} vs json {want:.4f}")
            # seed count
            if int(cells[2]) != len(rs):
                problems.append(f"{exp}: seed count {cells[2]} != {len(rs)}")

    # ---- table 5, ablations -------------------------------------------
    t5 = None
    for t in doc.tables:
        head = [c.text.strip() for c in t.rows[0].cells]
        if head[:3] == ["设置", "训练例数", "种子"]:
            t5 = t
            break
    if t5 is not None:
        for row in t5.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            label, shown_mae, shown_auc = cells[0], cell_num(cells[3]), cell_num(cells[4])
            pick: list[dict] = []
            m = re.match(r"帧数 (\d+)", label)
            if m:
                nf = int(m.group(1))
                pick = tagged(runs, "R4", f"abframes{nf}") or main_runs(runs, "R4")
            elif label == "池化 可学习词元":
                pick = main_runs(runs, "R4")
            elif label == "池化 输出均值":
                pick = tagged(runs, "R4_MEAN", "abpoolmean")
            elif label.startswith("训练规模"):
                n = int(re.search(r"(\d+)$", label).group(1))
                pick = tagged(runs, "R4", f"abscale{n}") or (
                    main_runs(runs, "R4") if n == 2000 else [])
            elif label.startswith("零样本比例"):
                pct = int(re.search(r"(\d+)%", label).group(1))
                pick = tagged(runs, "R1", f"abtop{pct}") or (
                    main_runs(runs, "R1") if pct == 20 else [])
            elif label == "零样本 等间隔 16 帧":
                pick = main_runs(runs, "R1U16")
            elif label.startswith("对比预训练 标注"):
                pick = tagged(runs, "R5_EDES", "edes")
            if not pick:
                problems.append(f"table5 {label}: no JSON")
                continue
            want_mae = float(np.mean([r["metrics"]["mae"] for r in pick]))
            want_auc = float(np.mean([r["metrics"]["auc_ef_lt50"] for r in pick]))
            checked += 2
            if shown_mae is None or abs(shown_mae - want_mae) > 0.006:
                problems.append(f"table5 {label}.MAE: docx {shown_mae} vs json {want_mae:.4f}")
            if shown_auc is None or abs(shown_auc - want_auc) > 0.0016:
                problems.append(f"table5 {label}.AUC: docx {shown_auc} vs json {want_auc:.4f}")

    print(f"checked table cells: {checked}")
    if problems:
        print("PROBLEMS")
        for p in problems:
            print(" -", p)
        return 1
    print("all table numbers agree with the result JSONs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
