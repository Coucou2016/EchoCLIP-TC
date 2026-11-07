"""Paper figures (scienceplots + matplotlib, Times New Roman, 300 dpi).

Every figure is written as both PDF and PNG into the build figures directory
and copied into the repo papers/figures folder.

Figures
  fig1_dataset       EF distribution by split + TEST-500 vs full TEST
  fig2_framework     method block diagram (matplotlib shapes, no screenshots)
  fig3_zeroshot_sup  zero-shot vs supervised heads: MAE and AUC50
  fig4_calibration   reliability diagram + conformal coverage
  fig5_ablation      frames / pooling / train-size ablations
  fig6_scatter       predicted vs measured EF and residual plot

Numbers are read from results/zh/*.json; nothing is hard-coded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import scienceplots  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import cache_dir, figures_dir, results_dir  # noqa: E402

RES = results_dir()
FIGS = figures_dir()
PAPER_FIGS = ROOT / "papers" / "figures"
CACHE = cache_dir()
DATA = Path(r"E:\Datasets\EchoNet-Dynamic")

plt.style.use(["science", "no-latex"])
plt.rcParams.update(
    {
        # a family *list* lets matplotlib fall back per glyph: Latin text uses
        # Times New Roman and CJK glyphs use SimSun, so no glyph is dropped.
        "font.family": ["Times New Roman", "SimSun"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = ["#2E5E8C", "#C1553B", "#4F8A5B", "#A8792B", "#6B5B95", "#8C5A72", "#3E7C7B"]


def save(fig, stem: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIGS / f"{stem}.{ext}"
        fig.savefig(p, dpi=300)
        (PAPER_FIGS / f"{stem}.{ext}").write_bytes(p.read_bytes())
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png", flush=True)


def load_json(name: str) -> Optional[dict]:
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def seed_metrics(exp: str, frames: int = 32, tag: Optional[str] = None, pool: str = "cls") -> List[dict]:
    """Collect every seed JSON of one experiment.

    Without a tag only the main-row names ``{exp}_s{seed}_f{frames}_{pool}.json``
    are accepted, so ablation runs with an extra suffix stay out of the main
    figures.
    """
    out = []
    if tag:
        for p in sorted(RES.glob(f"{exp}_s*_f{frames}_{pool}_{tag}*.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out
    import re

    suffix = r"(_a10_20)?" if exp == "R6" else ""
    exact = re.compile(rf"^{re.escape(exp)}_s\d+_f{frames}_{pool}{suffix}\.json$")
    for p in sorted(RES.glob(f"{exp}_s*_f{frames}_{pool}*.json")):
        if exact.match(p.name):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


# --------------------------------------------------------------------------
# Figure 1: dataset
# --------------------------------------------------------------------------
def fig_dataset() -> None:
    import pandas as pd

    fl = pd.read_csv(DATA / "FileList.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    bins = np.arange(0, 101, 4)
    for i, sp in enumerate(["TRAIN", "VAL", "TEST"]):
        v = fl.loc[fl["Split"] == sp, "EF"].values
        ax.hist(v, bins=bins, density=True, alpha=0.55, label=f"{sp} (n={len(v)})", color=COLORS[i])
    ax.set_xlabel("左心室射血分数 EF (%)")
    ax.set_ylabel("概率密度")
    ax.set_title("(a) 各划分的 EF 分布")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 100)

    ax = axes[1]
    test_all = fl.loc[fl["Split"] == "TEST", "EF"].values
    z = np.load(CACHE / "test500_32_bgr.npz", allow_pickle=False)
    sub = z["ef"]
    parts = ax.violinplot([test_all, sub], showmedians=True, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i + 1])
        pc.set_alpha(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"完整 TEST\n(n={len(test_all)})", f"TEST-500\n(n={len(sub)})"])
    ax.set_ylabel("EF (%)")
    ax.set_title("(b) 测试子集与完整测试集")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "fig1_dataset")


# --------------------------------------------------------------------------
# Figure 2: framework
# --------------------------------------------------------------------------
def fig_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.4, alpha=0.16):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.06,rounding_size=0.10",
                linewidth=0.9, edgecolor=color, facecolor=color, alpha=alpha,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="0.12", zorder=5)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                linewidth=0.9, color="0.35", linestyle=ls, shrinkA=1, shrinkB=1,
            )
        )

    # input
    box(0.15, 3.55, 1.55, 1.15, "超声视频\n10030 例\n(AVI, 112×112)", "0.35")
    # sampling
    box(2.00, 3.55, 1.55, 1.15, "帧采样\n均匀 8/16/32 帧\n或官方步长", COLORS[0])
    # frozen tower
    box(3.85, 2.55, 1.75, 2.15, "冻结视觉塔\nConvNeXt-Base\n224×224\n每帧 512 维", COLORS[0], fs=8.2)
    ax.text(4.72, 4.85, "不更新参数", ha="center", fontsize=7.6, color=COLORS[0], style="italic")

    # temporal heads
    box(5.95, 3.95, 1.85, 0.85, "R2/R3 均值池化\n+ 线性/MLP", COLORS[2])
    box(5.95, 2.90, 1.85, 0.85, "R4 时序 Transformer\n+ 分类词元聚合", COLORS[3])
    box(5.95, 1.85, 1.85, 0.85, "R5 软标签\n对比预训练", COLORS[4])
    box(5.95, 0.80, 1.85, 0.85, "R6 校准\nPlatt + 温度 + 保形", COLORS[5])

    # zero-shot branch
    box(3.85, 0.80, 1.75, 0.95, "R1 零样本\n提示排序", COLORS[1])
    box(3.85, 1.90, 1.75, 0.62, "文本编码器\n202 条提示", COLORS[1], fs=7.8)

    # output
    box(8.15, 2.35, 1.65, 1.55, "EF 预测\nMAE / AUC\n校准区间", "0.35")

    arrow(1.70, 4.12, 1.98, 4.12)
    arrow(3.55, 4.12, 3.83, 4.12)
    arrow(5.60, 4.05, 5.93, 4.30)
    arrow(5.60, 3.70, 5.93, 3.32)
    arrow(5.60, 3.30, 5.93, 2.75)
    arrow(5.60, 2.90, 5.93, 2.25)
    arrow(5.60, 1.55, 5.93, 2.05, ls=":")
    arrow(4.72, 2.52, 4.72, 2.20, ls=":")
    arrow(5.60, 1.27, 5.93, 1.22)
    for y in (4.37, 3.32, 2.27, 1.22):
        arrow(7.80, y, 8.13, 3.12, ls=":")
    ax.text(5.0, 0.42, "ORACLE-EDES 与 R5-EDES 仅在取帧环节替换为标注的 ED/ES 帧对", fontsize=7.4, color="0.4")

    ax.set_title("EchoCLIP-TA / EchoCLIP-TC 实验框架", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig2_framework")


# --------------------------------------------------------------------------
# Figure 3: zero-shot vs supervised
# --------------------------------------------------------------------------
def fig_zeroshot_sup() -> None:
    order = ["R1", "R2", "R3", "R4", "R5", "R6", "ORACLE_EDES"]
    labels = {
        "R1": "R1\n零样本",
        "R2": "R2\n线性",
        "R3": "R3\nMLP",
        "R4": "R4\n时序",
        "R5": "R5\n对比预训练",
        "R6": "R6\n校准",
        "ORACLE_EDES": "Oracle\nED/ES",
    }
    maes, maes_sd, aucs, aucs_sd, names = [], [], [], [], []
    for exp in order:
        runs = seed_metrics(exp)
        if not runs:
            continue
        m = [r["metrics"]["mae"] for r in runs]
        a = [r["metrics"]["auc_ef_lt50"] for r in runs]
        maes.append(float(np.mean(m)))
        maes_sd.append(float(np.std(m, ddof=1)) if len(m) > 1 else 0.0)
        aucs.append(float(np.nanmean(a)))
        aucs_sd.append(float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0)
        names.append(exp)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x, maes, yerr=maes_sd, capsize=3, color=[COLORS[i % len(COLORS)] for i in range(len(names))],
           edgecolor="0.25", linewidth=0.6, error_kw={"linewidth": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=7.6)
    ax.set_ylabel("MAE (EF 百分点)")
    ax.set_title("(a) 平均绝对误差，越低越好")
    for xi, v in zip(x, maes):
        ax.text(xi, v + 0.35, f"{v:.2f}", ha="center", fontsize=7.4)

    ax = axes[1]
    ax.bar(x, aucs, yerr=aucs_sd, capsize=3, color=[COLORS[i % len(COLORS)] for i in range(len(names))],
           edgecolor="0.25", linewidth=0.6, error_kw={"linewidth": 0.9})
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=7.6)
    ax.set_ylabel("AUC（EF 低于 50%）")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("(b) 低 EF 判别能力，越高越好")
    for xi, v in zip(x, aucs):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.2)

    fig.tight_layout()
    save(fig, "fig3_zeroshot_vs_supervised")


# --------------------------------------------------------------------------
# Figure 4: calibration
# --------------------------------------------------------------------------
def fig_calibration() -> None:
    runs = [r for r in seed_metrics("R6")]
    if not runs:
        print("  skip fig4: R6 missing", flush=True)
        return
    r0 = runs[0]
    cal = r0["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    thr_key = "ef_lt50"
    rec = cal["thresholds"][thr_key]
    from train_helpers import reliability_bins  # local import to keep deps minimal

    preds = r0["predictions"]
    y = np.array(preds["test_true"])

    # recompute probabilities to draw the reliability curve
    def raw_p(pred, thr=50.0, sigma=4.0):
        return 1.0 / (1.0 + np.exp(-(thr - pred) / sigma))

    p_raw = raw_p(np.array(preds["test_pred"]))
    a, b = rec["platt_a"], rec["platt_b"]
    p_pl = 1.0 / (1.0 + np.exp(-(a * p_raw + b)))
    yb = (y < 50).astype(int)
    for name, p, color in (("校准前", p_raw, COLORS[1]), ("Platt 校准后", p_pl, COLORS[0])):
        cx, cy, cnt = reliability_bins(yb, p, 10)
        m = cnt > 0
        ax.plot(cx[m], cy[m], "o-", markersize=3.6, linewidth=1.0, label=name, color=color)
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=0.8, label="理想校准")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际阳性比例")
    ax.set_title("(a) EF 低于 50% 的可靠性图")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    alphas = sorted(cal["conformal"].keys())
    tgt = [cal["conformal"][k]["target_coverage"] for k in alphas]
    cov = [cal["conformal"][k]["coverage"] for k in alphas]
    wid = [cal["conformal"][k]["width_mean"] for k in alphas]
    xx = np.arange(len(alphas))
    ax.bar(xx - 0.18, tgt, width=0.34, color="0.65", edgecolor="0.3", linewidth=0.6, label="名义覆盖率")
    ax.bar(xx + 0.18, cov, width=0.34, color=COLORS[0], edgecolor="0.25", linewidth=0.6, label="实测覆盖率")
    for xi, c, w in zip(xx, cov, wid):
        ax.text(xi + 0.18, c + 0.012, f"{c:.3f}\n半宽 {w/2:.1f}", ha="center", fontsize=7.2)
    ax.set_xticks(xx)
    ax.set_xticklabels([k.replace("alpha_", "α = ") for k in alphas])
    ax.set_ylabel("覆盖率")
    ax.set_ylim(0, 1.12)
    ax.set_title("(b) 保形区间覆盖与半宽")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save(fig, "fig4_calibration")


# --------------------------------------------------------------------------
# Figure 5: ablations
# --------------------------------------------------------------------------
def fig_ablation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))

    # (a) frames
    ax = axes[0]
    for i, nf in enumerate((8, 16, 32)):
        runs = seed_metrics("R4", frames=nf, tag=f"abframes{nf}")
        if not runs:
            runs = [r for r in seed_metrics("R4", frames=nf)]
        if not runs:
            continue
        m = [r["metrics"]["mae"] for r in runs]
        ax.errorbar(nf, np.mean(m), yerr=np.std(m, ddof=1) if len(m) > 1 else 0,
                    fmt="o", markersize=5, capsize=3, color=COLORS[0])
    ax.set_xlabel("采样帧数")
    ax.set_ylabel("MAE (EF 百分点)")
    ax.set_xticks([8, 16, 32])
    ax.set_title("(a) 帧数")

    # (b) pooling
    ax = axes[1]
    vals, sds, labs = [], [], []
    cls_runs = [r for r in seed_metrics("R4", tag="abframes32")] or seed_metrics("R4")
    mean_runs = seed_metrics("R4_MEAN", tag="abpoolmean")
    for lab, runs, color in (("可学习词元", cls_runs, COLORS[0]), ("输出均值", mean_runs, COLORS[2])):
        if not runs:
            continue
        m = [r["metrics"]["mae"] for r in runs]
        vals.append(np.mean(m))
        sds.append(np.std(m, ddof=1) if len(m) > 1 else 0)
        labs.append(lab)
    if vals:
        xx = np.arange(len(vals))
        ax.bar(xx, vals, yerr=sds, capsize=3, color=COLORS[: len(vals)],
               edgecolor="0.25", linewidth=0.6)
        ax.set_xticks(xx)
        ax.set_xticklabels(labs, fontsize=8)
        for xi, v in zip(xx, vals):
            ax.text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.4)
    ax.set_ylabel("MAE (EF 百分点)")
    ax.set_title("(b) 池化方式")

    # (c) train size
    ax = axes[2]
    xs, ys, es = [], [], []
    for n in (500, 1000, 2000):
        runs = seed_metrics("R4", tag=f"abscale{n}")
        if not runs:
            continue
        m = [r["metrics"]["mae"] for r in runs]
        xs.append(n)
        ys.append(np.mean(m))
        es.append(np.std(m, ddof=1) if len(m) > 1 else 0)
    if xs:
        ax.errorbar(xs, ys, yerr=es, fmt="o-", markersize=5, capsize=3,
                    color=COLORS[3], linewidth=1.1)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("MAE (EF 百分点)")
    ax.set_title("(c) 学习曲线")

    fig.tight_layout()
    save(fig, "fig5_ablation")


# --------------------------------------------------------------------------
# Figure 6: scatter + residual
# --------------------------------------------------------------------------
def fig_scatter() -> None:
    runs = seed_metrics("R5") or seed_metrics("R4")
    if not runs:
        print("  skip fig6: no supervised run", flush=True)
        return
    r = runs[0]
    y = np.array(r["predictions"]["test_true"])
    p = np.array(r["predictions"]["test_pred"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.scatter(y, p, s=9, alpha=0.55, color=COLORS[0], edgecolors="none")
    lim = [5, 100]
    ax.plot(lim, lim, "--", color="0.45", linewidth=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("参考 EF (%)")
    ax.set_ylabel("预测 EF (%)")
    ax.set_title(f"(a) {r['experiment']} 预测散点 (n={len(y)})")
    ax.text(0.05, 0.92, f"MAE {r['metrics']['mae']:.2f}\nr = {r['metrics']['pearson']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[1]
    resid = p - y
    ax.scatter(y, resid, s=9, alpha=0.55, color=COLORS[1], edgecolors="none")
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
    ax.axhline(float(np.mean(resid)), color=COLORS[0], linewidth=1.0,
               label=f"平均偏差 {np.mean(resid):+.2f}")
    ax.set_xlabel("参考 EF (%)")
    ax.set_ylabel("残差 (预测 − 参考, %)")
    ax.set_title("(b) 残差随 EF 的分布")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "fig6_scatter_residual")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (fig_dataset, fig_framework, fig_zeroshot_sup, fig_calibration, fig_ablation, fig_scatter):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
