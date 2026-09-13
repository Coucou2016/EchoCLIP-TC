#!/usr/bin/env python3
"""Plot risk–coverage curves (SciencePlots when available).

Reads ``metrics.json`` keys ``risk_coverage_coverage`` / ``risk_coverage_mae``
(and optional ``aurc``) written by ``summarize_clinical`` with adaptive conformal,
or accepts a small synthetic DEMO curve for figure wiring.

Examples
--------
::

  python scripts/plot_risk_coverage.py --metrics checkpoints/protocol/R6/metrics.json
  python scripts/plot_risk_coverage.py --demo-curve --output figures/fig_risk_coverage_demo.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _setup_style():
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:  # noqa: BLE001
        pass
    return plt


def plot_curve(
    coverage,
    risk,
    *,
    aurc: float | None,
    title: str,
    output: Path,
    demo: bool,
) -> Path:
    plt = _setup_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(coverage, risk, marker="o", markersize=3, linewidth=1.5, color="#1f4e79")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Risk (MAE)")
    label = title
    if aurc is not None and not (isinstance(aurc, float) and np.isnan(aurc)):
        label = f"{title}\nAURC={float(aurc):.4g}"
    ax.set_title(label)
    if demo:
        ax.text(
            0.98,
            0.02,
            "DEMO — not clinical",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#a31f34",
        )
    ax.set_xlim(0, 1.02)
    ax.grid(True, alpha=0.3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk–coverage plot (SciencePlots)")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument(
        "--demo-curve",
        action="store_true",
        help="Synthetic DEMO curve (not clinical)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "fig_risk_coverage_demo.png",
    )
    parser.add_argument("--title", type=str, default="Risk–coverage")
    args = parser.parse_args()

    if args.demo_curve and args.metrics is None:
        rng = np.random.default_rng(0)
        n = 40
        y = rng.normal(55, 12, size=n)
        p = y + rng.normal(0, 8, size=n)
        u = np.abs(p - y) + rng.random(n)
        order = np.argsort(u)
        cov, risk = [], []
        for k in range(1, 21):
            n_keep = max(1, int(np.ceil(k / 20 * n)))
            idx = order[:n_keep]
            cov.append(n_keep / n)
            risk.append(float(np.mean(np.abs(p[idx] - y[idx]))))
        from echoclip.calibrate import area_under_risk_coverage

        aurc = area_under_risk_coverage(cov, risk)
        out = plot_curve(
            cov,
            risk,
            aurc=aurc,
            title=args.title + " (DEMO)",
            output=args.output,
            demo=True,
        )
        print(f"Wrote DEMO risk–coverage → {out}")
        return 0

    if args.metrics is None or not args.metrics.exists():
        print(
            "No metrics.json with risk_coverage_* keys. "
            "Use --demo-curve for a labeled DEMO figure, or run R6 with "
            "--adaptive-conformal first."
        )
        return 1

    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    cov = data.get("risk_coverage_coverage")
    risk = data.get("risk_coverage_mae")
    if not cov or not risk:
        print(f"{args.metrics} missing risk_coverage_coverage / risk_coverage_mae")
        return 1
    demo = bool(data.get("demo_is_not_clinical") or data.get("demo_mode"))
    out = plot_curve(
        cov,
        risk,
        aurc=data.get("aurc"),
        title=args.title,
        output=args.output,
        demo=demo,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
