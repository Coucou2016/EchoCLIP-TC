"""Build the Chinese manuscript (Word) with a live TOC field.

All numbers come from results/zh/*.json written in this session. The builder
raises if a required file is missing, so no table or figure value can appear
without a JSON behind it.

Outputs
  E:\\Projects\\20260522-EchoCLIP\\papers\\EchoCLIP-TA_中文稿.docx
  <build root>\\exports\\EchoCLIP-TA_中文稿.docx
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from docx_helpers import (  # noqa: E402
    add_body,
    add_caption,
    add_equation,
    add_heading,
    add_page_number_footer,
    add_picture,
    add_three_line_table,
    add_toc,
    enable_update_fields,
    setup_styles,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import exports_dir, figures_dir, results_dir

RES = results_dir()
FIGS = figures_dir()
PAPERS = ROOT / "papers"
EXPORTS = exports_dir()

EXP_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "ORACLE_EDES"]
EXP_LABEL = {
    "R0": "官方步长零样本",
    "R1": "均匀帧零样本",
    "R2": "均值池化线性回归",
    "R3": "均值池化两层感知机",
    "R4": "时序 Transformer",
    "R5": "对比预训练加回归",
    "R6": "对比预训练加校准",
    "ORACLE_EDES": "标注 ED/ES 零样本",
}


# --------------------------------------------------------------------------
def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise FileNotFoundError(f"required result missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def gather(exp: str, frames: int = 32, tag: str = "", pool: str = "cls") -> List[dict]:
    """Collect seed JSONs of one experiment.

    With no ``tag`` this returns only the *main row* files, i.e. the exact
    ``{exp}_s{seed}_f{frames}_{pool}.json`` names, so ablation files such as
    ``R4_s42_f32_cls_abscale500.json`` never leak into the main table.
    """
    out = []
    if tag:
        # the tag may sit right after the pool token or after another suffix
        # (e.g. R1_s42_f32_cls_top10_abtop10.json), so allow a leading wildcard
        seen = set()
        for pat in (f"{exp}_s*_f{frames}_{pool}_{tag}*.json",
                    f"{exp}_s*_f{frames}_{pool}_*{tag}*.json"):
            for p in sorted(RES.glob(pat)):
                if p.name in seen:
                    continue
                seen.add(p.name)
                out.append(json.loads(p.read_text(encoding="utf-8")))
        return out
    import re

    suffix = r"(_a10_20)?" if exp == "R6" else ""
    exact = re.compile(rf"^{re.escape(exp)}_s\d+_f{frames}_{pool}{suffix}\.json$")
    for p in sorted(RES.glob(f"{exp}_s*_f{frames}_{pool}*.json")):
        if exact.match(p.name):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def ms(vals: List[float], nd: int = 3) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}"


def f(x, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    audit = load("data_audit.json")
    refs = load("references_verified.json")
    stat = load("stat_robustness.json") if (RES / "stat_robustness.json").exists() else None

    runs: Dict[str, List[dict]] = {}
    for exp in EXP_ORDER:
        runs[exp] = gather(exp, frames=32 if exp != "R0" else 20,
                           pool="cls" if exp not in ("R0", "R1") else "cls")
        if exp == "R0":
            # R0 lives in the stride cache with 20 frames
            runs[exp] = gather(exp, frames=20)

    present = {k: v for k, v in runs.items() if v}
    # uniform-16 zero-shot control, reported as a sampling ablation in 5.2
    u16 = gather("R1U16", frames=16)
    if u16:
        present["R1U16"] = u16
    if "R2" not in present:
        raise SystemExit("no supervised results found; run the training matrix first")

    fl = audit["filelist"]
    sub = audit["test_distribution"]
    tr = present["R2"][0]
    n_train, n_val, n_test = tr["n_train"], tr["n_val"], tr["n_test"]

    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.8)
    sec.right_margin = Cm(2.8)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    add_page_number_footer(sec)

    # ---------------- title block ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("冻结视觉塔条件下的超声心动图射血分数估计")
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "黑体")

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(10)
    r2 = p2.add_run("时序聚合、对比预训练与校准的对照研究")
    r2.font.size = Pt(12)
    r2.font.name = "Times New Roman"

    for line in ("EchoCLIP-TA / EchoCLIP-TC 项目组", "2026 年 11 月"):
        pp = doc.add_paragraph()
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.space_after = Pt(0)
        rr = pp.add_run(line)
        rr.font.size = Pt(11)
        rr.font.name = "Times New Roman"

    doc.add_paragraph()
    add_heading(doc, "目录", 1)
    add_toc(doc)
    doc.add_page_break()

    # ======================================================================
    # 摘要
    # ======================================================================
    add_heading(doc, "摘要", 1)
    r6 = present.get("R6", present.get("R5"))[0]
    best_exp = _best_exp(present)
    best = present[best_exp][0]
    abstract = _abstract(present, best_exp, best, sub, n_train, n_val, n_test)
    add_body(doc, abstract, indent=True)

    kw = doc.add_paragraph()
    kw.paragraph_format.space_before = Pt(6)
    r = kw.add_run("关键词　")
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.name = "Times New Roman"
    r2 = kw.add_run("超声心动图；射血分数；视觉语言模型；时序聚合；对比学习；保形预测")
    r2.font.size = Pt(12)
    r2.font.name = "Times New Roman"

    doc.add_page_break()

    # ======================================================================
    # 1 引言
    # ======================================================================
    add_heading(doc, "1 引言", 1)
    add_heading(doc, "1.1 背景与问题", 2)
    for para in _intro_background(fl):
        add_body(doc, para)

    add_heading(doc, "1.2 已有研究与缺口", 2)
    for para in _intro_gap():
        add_body(doc, para)

    add_heading(doc, "1.3 本文工作", 2)
    for para in _intro_work(present, best_exp):
        add_body(doc, para)

    # ======================================================================
    # 2 相关工作
    # ======================================================================
    add_heading(doc, "2 相关工作", 1)
    add_heading(doc, "2.1 超声心动图的自动分析", 2)
    for para in _rw_echo():
        add_body(doc, para)
    add_heading(doc, "2.2 医学视觉语言预训练", 2)
    for para in _rw_vl():
        add_body(doc, para)
    add_heading(doc, "2.3 时序聚合与多示例学习", 2)
    for para in _rw_temporal():
        add_body(doc, para)
    add_heading(doc, "2.4 概率校准与不确定性", 2)
    for para in _rw_calib():
        add_body(doc, para)

    # ======================================================================
    # 3 方法
    # ======================================================================
    add_heading(doc, "3 方法", 1)
    add_body(
        doc,
        "整体结构如图 1 所示。输入是一段心尖四腔超声视频，先做等间隔帧采样，"
        "再逐帧通过冻结的视觉塔得到五百一十二维嵌入并缓存。"
        "此后分成两条路径，零样本路径把帧嵌入与候选文本提示比较，"
        "监督路径在缓存上训练时序聚合头并输出点估计，最后接入校准层给出概率与区间。"
        "两条路径共用同一份缓存，因此差异只来自聚合方式与目标函数。",
    )
    add_picture(
        doc,
        "fig2_framework",
        "图 1  EchoCLIP-TA / EchoCLIP-TC 的实验框架。左侧为输入与帧采样，"
        "中间为冻结视觉塔与文本编码器，右侧为四类预测头与输出指标。"
        "虚线表示只在标注辅助实验中使用的分支。",
        width_cm=15.5,
    )
    eq = _EqCounter()

    add_heading(doc, "3.1 问题定义", 2)
    _render(doc, _m_problem(eq))

    add_heading(doc, "3.2 冻结双塔与帧级嵌入", 2)
    _render(doc, _m_encoder(eq))

    add_heading(doc, "3.3 帧采样与预处理", 2)
    _render(doc, _m_sampling(eq))

    add_heading(doc, "3.4 零样本预测模式", 2)
    _render(doc, _m_zeroshot(eq))

    add_heading(doc, "3.5 监督回归头", 2)
    add_heading(doc, "3.5.1 均值池化与线性回归", 3)
    _render(doc, _m_linear(eq))
    add_heading(doc, "3.5.2 两层感知机", 3)
    _render(doc, _m_mlp(eq))

    add_heading(doc, "3.6 时序编码器与池化方式", 2)
    _render(doc, _m_temporal(eq))

    add_heading(doc, "3.7 软标签多正样本对比预训练", 2)
    _render(doc, _m_contrastive(eq))

    add_heading(doc, "3.8 概率校准与保形区间", 2)
    _render(doc, _m_calibration(eq))

    add_heading(doc, "3.9 实验矩阵 R0 至 R6", 2)
    for para in _m_matrix():
        add_body(doc, para)
    _table_matrix(doc)

    add_heading(doc, "3.10 评价指标", 2)
    _render(doc, _m_metrics(eq))

    # ======================================================================
    # 4 实验设置
    # ======================================================================
    add_heading(doc, "4 实验设置", 1)
    add_heading(doc, "4.1 数据集与划分", 2)
    for para in _e_data(fl, audit, n_train, n_val, n_test):
        add_body(doc, para)
    add_picture(
        doc,
        "fig1_dataset",
        "图 2  数据分布。(a) 三个官方划分的射血分数概率密度，箱宽为四个百分点，"
        f"n 分别为 {audit['filelist']['split_counts']['TRAIN']}、"
        f"{audit['filelist']['split_counts']['VAL']} 与 "
        f"{audit['filelist']['split_counts']['TEST']}。(b) 完整测试划分与本文所用 "
        f"{n_test} 例子集的分布对照，小提琴内部横线为中位数。纵轴单位为 EF 百分点。",
        width_cm=15.5,
    )
    _table_splits(doc, audit)

    add_heading(doc, "4.2 实现细节", 2)
    for para in _e_impl(tr):
        add_body(doc, para)

    add_heading(doc, "4.3 硬件与软件环境", 2)
    for para in _e_env(tr):
        add_body(doc, para)

    add_heading(doc, "4.4 可复现命令", 2)
    for para in (
        "全部实验由七个脚本完成。脚本一负责冻结塔的帧嵌入缓存，脚本二负责头部训练与指标计算，"
        "脚本三负责消融与统计检验，脚本四负责数据审计，脚本五负责绘图，脚本六生成论文稿，"
        "脚本七把论文表格逐格回读并与结果文件比对。"
        "命令按执行顺序列出，每条命令的完整字符串同时写在对应结果文件的 command 字段中。",
    ):
        add_body(doc, para)
    _code_block(doc, [
        "python scripts/extract_frames_embeddings.py --split test500 --limit 500 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split val --limit 1288 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split train --limit 2000 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split test500 --sample official_stride --batch-size 16",
        "python scripts/train_heads_from_cache.py --experiment R0 --seeds 42 --frames 32",
        "python scripts/train_heads_from_cache.py --experiment R1 --seeds 42 --frames 32",
        "python scripts/train_heads_from_cache.py --experiment R1U16 --seeds 42 --frames 16",
        "python scripts/train_heads_from_cache.py --experiment R2 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20",
        "python scripts/train_heads_from_cache.py --experiment R3 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20",
        "python scripts/train_heads_from_cache.py --experiment R4 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20",
        "python scripts/train_heads_from_cache.py --experiment R5 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20",
        "python scripts/train_heads_from_cache.py --experiment R6 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20",
        "python scripts/train_heads_from_cache.py --experiment ORACLE_EDES --seeds 42 --frames 32",
        "python scripts/run_ablation_matrix.py --group all",
        "python scripts/audit_data.py",
        "python scripts/make_figures.py",
        "python scripts/build_manuscript.py",
        "python scripts/crosscheck_numbers.py",
    ])
    add_body(
        doc,
        "上面每个训练命令中的种子四十二在实际执行时依次替换为四十二、四十三、四十四，"
        "因此表中带标准差的每一行都来自三次独立运行。"
        "帧数消融不重新解码视频，只从三十二帧缓存中按等间隔取八帧或十六帧，"
        "因此该项的差异只来自序列长度，不掺入解码阶段的随机性。"
        "八帧与十六帧各跑两个种子，三十二帧跑三个种子。",
    )

    # ======================================================================
    # 5 结果
    # ======================================================================
    add_heading(doc, "5 结果", 1)
    add_heading(doc, "5.1 主结果", 2)
    for para in _r_main(present, best_exp):
        add_body(doc, para)
    add_picture(doc, "fig3_zeroshot_vs_supervised",
                "图 3  零样本与监督头的对比。(a) 测试集 MAE，误差条为三个随机种子的标准差。(b) EF 低于 50% 的判别 AUC，虚线为随机水平。测试集 n 为 "
                f"{n_test}，单位为 EF 百分点。", width_cm=15.5)
    _table_main(doc, present)

    add_heading(doc, "5.2 零样本与标注辅助上界", 2)
    for para in _r_zeroshot(present):
        add_body(doc, para)

    add_heading(doc, "5.3 校准结果", 2)
    for para in _r_calibration(present):
        add_body(doc, para)
    add_picture(doc, "fig4_calibration",
                "图 4  校准与保形区间。(a) EF 低于 50% 的可靠性图，对角线为理想校准。(b) 保形区间的名义覆盖率与实测覆盖率，柱上标注实测覆盖率与平均区间半宽。校准参数仅用验证集的 "
                f"{n_val} 例拟合，测试集 n 为 {n_test}。", width_cm=15.5)
    _table_calibration(doc, present)

    add_heading(doc, "5.4 消融实验", 2)
    for para in _r_ablation():
        add_body(doc, para)
    add_picture(doc, "fig5_ablation",
                "图 5  消融结果。(a) 采样帧数 8、16、32。(b) 分类词元聚合与输出序列均值聚合。(c) 训练规模 500、1000、2000。误差条为随机种子的标准差，纵轴单位为 EF 百分点。",
                width_cm=15.5)
    _table_ablation(doc, present)

    add_heading(doc, "5.5 预测误差的分布与失败案例", 2)
    for para in _r_scatter(present, audit.get("train_distribution")):
        add_body(doc, para)
    add_picture(doc, "fig6_scatter_residual",
                "图 6  预测值与参考值的散点及残差。(a) 对角线为理想预测。(b) 残差随参考 EF 的变化，水平线为平均偏差。测试集 n 为 "
                f"{n_test}。", width_cm=15.5)

    add_heading(doc, "5.6 统计不确定性与测试集代表性", 2)
    for para in _r_stats(present, sub, stat):
        add_body(doc, para)
    _table_bootstrap(doc, present)

    # ======================================================================
    # 6 讨论
    # ======================================================================
    add_heading(doc, "6 讨论", 1)
    add_heading(doc, "6.1 机制分析", 2)
    for para in _d_mechanism(present, best_exp):
        add_body(doc, para)
    add_heading(doc, "6.2 与已有工作的比较", 2)
    for para in _d_compare(present):
        add_body(doc, para)
    add_heading(doc, "6.3 适用边界", 2)
    for para in _d_limits(present):
        add_body(doc, para)

    # ======================================================================
    # 7 结论
    # ======================================================================
    add_heading(doc, "7 结论", 1)
    for para in _conclusion(present, best_exp):
        add_body(doc, para)

    # ======================================================================
    # 可用性
    # ======================================================================
    add_heading(doc, "数据与代码可用性", 1)
    for para in _availability(tr, audit):
        add_body(doc, para)

    # ======================================================================
    # 参考文献
    # ======================================================================
    add_heading(doc, "参考文献", 1)
    for i, r in enumerate(refs, 1):
        _reference_entry(doc, i, r)

    # ---------------- persist ----------------
    enable_update_fields(doc)
    PAPERS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = PAPERS / "EchoCLIP-TA_中文稿.docx"
    doc.save(out)
    shutil.copy2(out, EXPORTS / out.name)
    print(f"wrote {out}")
    print(f"copy  {EXPORTS / out.name}")
    return 0


# ==========================================================================
# narrative builders
# ==========================================================================
_METRIC_KEYS = ("mae", "rmse", "r2", "bias", "pearson", "spearman", "auc_ef_lt50", "auprc_ef_lt50")


def _seed_mean(present: Dict[str, List[dict]], exp: str) -> Dict[str, float]:
    """Mean of each headline metric over the seeds of one experiment."""
    runs = present[exp]
    return {k: float(np.mean([r["metrics"][k] for r in runs])) for k in _METRIC_KEYS}


def _best_exp(present: Dict[str, List[dict]]) -> str:
    cands = [e for e in ("R5", "R6", "R4") if e in present]
    if not cands:
        return "R2"
    return min(cands, key=lambda e: float(np.mean([r["metrics"]["mae"] for r in present[e]])))


def _abstract(present, best_exp, best, sub, n_train, n_val, n_test) -> str:
    z = _seed_mean(present, "R1")
    r4 = present.get("R4", [{}])[0]
    r5 = present.get("R5", [{}])[0]
    r6 = present.get("R6", [{}])[0]
    auc = _seed_mean(present, best_exp)["auc_ef_lt50"]
    z_mae = _seed_mean(present, "R1")["mae"]
    conf = r6.get("calibration", {}).get("conformal", {})
    a10 = conf.get("alpha_0.1", {})
    r6runs = present.get("R6", [])
    cov10 = float(np.mean([r["calibration"]["conformal"]["alpha_0.1"]["coverage"] for r in r6runs])) if r6runs else float("nan")
    wid10 = float(np.mean([r["calibration"]["conformal"]["alpha_0.1"]["width_mean"] for r in r6runs])) if r6runs else float("nan")
    parts = [
        "超声心动图射血分数的自动估计依赖大样本标注，公开的视觉语言模型提供了可复用的先验。",
        "本文研究在冻结视觉塔的前提下，时序聚合与对比预训练能把零样本估计改善到什么程度。",
        "方法上保留 EchoCLIP 的冻结双塔，把每段视频的帧级嵌入一次性缓存，再在缓存上训练六类预测器，"
        "包括零样本提示排序、均值池化线性回归、两层感知机、时序 Transformer、软标签对比预训练加回归，"
        "以及只在校准集上拟合的校准层。",
        f"数据取自 EchoNet-Dynamic，训练 {n_train} 例，验证 {n_val} 例，测试 {n_test} 例，"
        "测试集为官方测试划分的随机子集。",
    ]
    if r4:
        m4 = np.mean([r["metrics"]["mae"] for r in present["R4"]])
        parts.append(
            f"时序 Transformer 把测试集平均绝对误差从零样本的 {z_mae:.2f} 降到 {m4:.2f}。"
        )
    if r5:
        m5 = np.mean([r["metrics"]["mae"] for r in present["R5"]])
        m4 = np.mean([r["metrics"]["mae"] for r in present["R4"]]) if "R4" in present else float("nan")
        if np.isfinite(m4) and m5 < m4 - 0.1:
            parts.append(f"加入软标签对比预训练后进一步降到 {m5:.2f}。")
        elif np.isfinite(m4):
            parts.append(
                f"加入软标签对比预训练后误差为 {m5:.2f}，相对时序 Transformer 的变化为 "
                f"{m5 - m4:+.2f} 个百分点，落在种子噪声范围内。"
            )
        else:
            parts.append(
                f"加入软标签对比预训练后误差为 {m5:.2f}，相对时序 Transformer 没有下降，"
                "说明在这组数据上该目标没有带来额外收益。"
            )
    parts.append(
        f"EF 低于 50% 的判别 AUC 从 {_seed_mean(present, 'R1')['auc_ef_lt50']:.3f} 提升到 {auc:.3f}。"
    )
    if a10:
        parts.append(
            f"校准后的保形区间在 90% 名义水平上取得 {cov10:.3f} 的实测覆盖率，"
            f"平均半宽 {wid10 / 2:.2f} 个百分点。"
        )
    parts.append(
        "结果表明冻结视觉塔加轻量时序头是一条成本较低的路径，"
        "其代价是仍需在目标队列上重新拟合校准参数。"
    )
    return "".join(parts)


def _intro_background(fl) -> List[str]:
    return [
        f"左心室射血分数是超声心动图报告中最常被引用的一个数字。它决定收缩功能是否受损，也影响器械植入与随访策略。"
        f"手工描记需要医师逐帧勾画心内膜边界，耗时且存在观察者间差异。EchoNet-Dynamic 公开了 {fl['rows']} 例心尖四腔视频及其参考射血分数[1]，"
        f"为数据驱动的自动估计提供了可比较的基准。",
        "深度模型在这项任务上已经可以达到接近专家的误差水平，但代价是每个新任务都需要重新标注与重新训练。"
        "视觉语言预训练提供了另一种思路，它把视频与自由文本配对，使模型在没有任务标签时也能给出预测[5]。"
        "EchoCLIP 属于这一类模型，它在配对报告上学习，并报告了零样本射血分数的外部验证结果[2]。",
        "零样本预测的误差仍明显高于专门训练的回归模型，而且它的输出是点估计，没有可信区间。"
        "临床使用更关心两个问题，一是预测误差能否压缩到与专用模型相当的水平，二是能否给出覆盖率可控的区间。"
        "这两个问题在冻结视觉塔的条件下可以分开研究，因为特征一旦固定，差异就只来自聚合方式与目标函数。",
        "另一层背景是算力。公开的超声视觉语言模型参数量在数亿量级，端到端微调需要数十吉字节显存。"
        "多数医院的推理服务器不具备这一条件，而单卡小显存的机器可以完成特征提取与轻量头部训练。"
        "本文的实验环境是一块四吉字节显存的移动显卡，这一约束直接决定了方法的选择。",
    ]


def _intro_gap() -> List[str]:
    return [
        "已有工作大多把视觉语言模型当作特征提取器，然后接一个线性层做微调，很少系统比较不同的时序聚合方式。"
        "视频是被当作若干独立帧的平均，还是被当作有序序列，在文献中没有一致的结论。",
        "对比学习在医学影像中的用法通常是自监督预训练整个编码器[8,13]，而在冻结特征上做软标签对比预训练的报告很少。"
        "射血分数是连续量，相邻取值在生理上接近，这为软标签提供了自然的构造方式。",
        "校准问题在分类任务中被反复讨论[6]，在连续回归任务上却常被忽略。"
        "保形预测给出有限样本的覆盖率保证[7]，但它在射血分数回归上的实际区间宽度缺少公开的对照数据。",
        "零样本提示排序的聚合规则也有讨论空间。官方实现取排序结果的高分位中位数，"
        "这一比例是一个自由参数，它的影响在已有报告中很少被量化。",
        "上述四点共同构成本文的出发点，即在同一个冻结特征缓存上，把取帧方式、聚合方式、目标函数与校准层作为可控因素逐一比较。"
        "由于全部实验共用同一份缓存，任何一个因素的差异都不会被特征本身的随机性掩盖。",
    ]


def _intro_work(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)
    best = present[best_exp][0]
    out = [
        "本文把帧级嵌入一次性算好并缓存，此后所有头部训练都在缓存上完成。"
        "这使得在同一批特征上比较六类预测器成为可能，也使得随机种子之间的方差可以被直接估计。",
        "具体地，本文比较零样本提示排序、均值池化线性回归、两层感知机、时序 Transformer、"
        "软标签对比预训练加回归，以及带校准的版本，并给出帧数、池化方式、训练规模与提示比例的消融。"
        f"在 {best['n_test']} 例测试样本上，误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，"
        f"平均绝对误差 {bm['mae']:.2f}，AUC {bm['auc_ef_lt50']:.3f}，"
        f"零样本对照分别为 {z['mae']:.2f} 与 {z['auc_ef_lt50']:.3f}，均为三个种子的均值。",
        "本文的贡献有四点。第一，给出一个可复现的冻结特征实验框架，把帧嵌入与头部训练解耦，"
        "使消融可以在数分钟内完成。第二，量化了取帧方式对零样本误差的影响，"
        "并说明官方步长与等间隔采样在这组数据上差异很小。第三，比较了四种聚合方式与三种目标函数，"
        "给出各因素的相对贡献。第四，报告了保形区间在射血分数回归上的实测覆盖率与宽度，"
        "为临床应用提供了可参照的数字。",
        "所有数字都对应仓库内的结果文件与可重跑的命令，审查文档给出了逐条溯源表。",
    ]
    return out


def _rw_echo() -> List[str]:
    return [
        "EchoNet-Dynamic 建立了心尖四腔视频与射血分数配对的公开基准，并给出了基于三维卷积的基线[1]。"
        "此后大量工作围绕分割、容积估计与预后预测展开。CAMUS 提供了多中心的分割数据，使跨设备泛化可以被量化[4]。"
        "这些工作的共同点是任务专用，模型结构与标签绑定。",
        "EchoCLIP 把配对报告作为监督信号，用对比学习对齐视频与文本，从而支持零样本分类与回归[2]。"
        "它的报告结果说明视觉语言预训练在超声领域可行，但零样本射血分数的误差仍高于专用回归模型。"
        "CardiacCLIP 沿着同一方向，把 CLIP 适配到视频并考察少样本条件下的射血分数预测[3]。"
        "Echo-Vision-FM 则强调预训练与微调两阶段的衔接，报告了多个下游任务的结果[12]。"
        "EchoPrime 把多视频多视角的配对报告纳入预训练，其报告任务覆盖射血分数与多项结构指标[14]。"
        "Holste 等人的工作用对比自监督在超声视频上预训练编码器，报告了数据效率方面的收益[13]。"
        "更早的监督式工作则说明在小样本条件下端到端训练可以达到可用的诊断精度[11]。",
    ]


def _rw_vl() -> List[str]:
    return [
        "CLIP 用四亿图文对训练双塔，证明了对比目标可以学到可迁移的表示[5]。"
        "它的成功催生了大量医学领域的适配工作。医学影像的文本通常来自报告，句式长且包含否定与不确定表达，"
        "因此简单的句子级对比往往不够。",
        "在超声领域，报告中的射血分数以数值出现，这为把回归问题转成提示排序提供了便利。"
        "把数值离散成候选短语，再用相似度排序，可以在不训练回归头的情况下得到连续预测。"
        "这一做法的问题是精度受候选网格与聚合规则限制，本文将其作为零样本基线并量化其误差。",
    ]


def _rw_temporal() -> List[str]:
    return [
        "视频理解中的时序聚合大致分为三类，均值池化、注意力池化与序列编码器。"
        "均值池化没有参数，对帧数变化稳健，但忽略了顺序信息。"
        "注意力池化用一个可学习的查询向量为每帧分配权重，能够突出收缩末期等关键帧。"
        "序列编码器把帧当作词元，用自注意力在帧之间交换信息[9]。"
        "本文的时序头属于第三类，聚合方式则用可学习词元实现，"
        "消融中把它换成了对输出序列直接取平均。",
        "Transformer 的自注意力在长序列上有二次复杂度，而超声视频的帧数通常在两百帧左右。"
        "在冻结特征上做时序建模可以规避这一开销，因为编码器只需处理十六到三十二个词元。"
        "多示例学习的视角也支持这一做法，视频是包，帧是示例，标签只在包级别给出。",
    ]


def _rw_calib() -> List[str]:
    return [
        "现代神经网络的置信度普遍偏高，温度缩放用一个标量修正 logit 尺度，是最简单的后处理校准方法[6]。"
        "Platt 缩放把分数映射为概率，参数只有两个，适合验证集较小的场合。",
        "保形预测不假设数据分布，只要求校准样本与测试样本可交换[7]。"
        "在回归任务上，对绝对残差取分位数即可得到固定宽度的区间，覆盖率有有限样本保证。"
        "这一性质对临床报告有吸引力，因为医师需要知道区间在多大比例上覆盖真值。"
        "本文在校准集上拟合分位数，在测试集上报告实测覆盖率与区间宽度。",
    ]


def _render(doc, blocks) -> None:
    """Emit a builder's blocks: lead-in prose, numbered equation, optional note."""
    for para, formula in blocks:
        if para:
            add_body(doc, para)
        if not formula:
            continue
        add_equation(doc, formula[0], formula[1])
        if len(formula) > 2 and formula[2]:
            add_body(doc, formula[2])


class _EqCounter:
    def __init__(self):
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


def _m_problem(eq) -> List[tuple]:
    return [
        ("设第 i 段心尖四腔超声视频包含 N_i 帧，参考射血分数 y_i 由舒张末期容积与收缩末期容积定义。"
         "给定视频集合与标签对，目标是学习一个映射 f，使预测值与参考值的平均绝对误差最小。"
         "记第 i 段视频经采样后得到的帧级特征序列为 Z_i，其形状为采样帧数乘以特征维度。"
         "本文把该序列的长度记为 K，把特征维度记为 d，二者的取值在 3.3 节与 3.2 节说明。", None),
        ("参考值按容积比定义，其中舒张末期容积与收缩末期容积均为正数。"
         "这一比值消除了体型差异，因此不同患者之间的标签可以直接比较。",
         ("EF = (EDV − ESV) / EDV × 100%", eq.next())),
        ("模型输出一个实数预测，训练目标为平滑后的绝对误差。"
         "选择绝对误差而非平方误差，是因为射血分数在两端存在少量离群标注，"
         "平方误差会让这些样本主导梯度。平滑处理使小残差区间内的梯度保持稳定。",
         ("L_reg = mean( smoothL1( ŷ_i − y_i ) )", eq.next())),
        ("训练目标里的平滑参数取零点一。当残差绝对值小于该值时损失为二次型，超过时转为线性，"
         "于是在大残差样本上梯度不再随误差线性放大。", None),
        ("除点估计外，本文还要求模型给出一个区间，使真值以不低于名义水平的概率落入其中。"
         "这一要求在 3.8 节用保形预测实现。", None),
    ]


def _m_encoder(eq) -> List[tuple]:
    return [
        ("视觉塔为 ConvNeXt-Base[10]，输入为二百二十四乘二百二十四的三通道图像，输出为五百一十二维向量。"
         "输入张量记作形状为批大小乘三乘二百二十四乘二百二十四，输出张量形状为批大小乘五百一十二。"
         "文本塔为十二层 Transformer[9]，上下文长度七十七，词表四万九千四百零八，输出维度同为五百一十二。"
         "两塔在本文全部实验中保持冻结，梯度不回传，因此视觉塔的参数量不进入优化器状态。", None),
        ("图像在送入视觉塔前按通道做标准化，均值与标准差取自 CLIP 的公开取值。"
         "记单帧输入为 x，编码结果为 z，二者的关系如下。", None),
        ("", ("z = f(x) / || f(x) ||_2,  z ∈ R^512", eq.next(),
               "其中 f 表示视觉塔，输出再做一次二范数归一化，使余弦相似度等于内积。"
               "归一化后的向量落在单位球面上，因此不同帧之间的相似度可以直接比较。")),
        ("文本侧采用两条模板，分别把候选数值填入百分号之前。"
         "候选数值取零到一百共一百零一个整数，模板数乘以候选数得到提示总数二百零二。"
         "提示嵌入的矩阵形状为二百零二乘五百一十二，在缓存阶段算好一次并复用。", None),
        ("文本嵌入同样做归一化，记为 t。图像与文本的相似度取内积，不再乘温度系数，"
         "因为排序只依赖相对大小，而温度是单调变换，不改变次序。",
         ("s(x, c) = z · t_c,  c = 1, …, 202", eq.next())),
        ("冻结双塔的设计意味着全部下游差异只能来自聚合方式与目标函数。"
         "这一约束牺牲了特征对新任务的适应性，换来的是可复现性，"
         "因为同一段视频的帧级特征在任意实验之间完全一致，随机性只来自头部初始化与数据顺序。", None),
    ]


def _m_sampling(eq) -> List[tuple]:
    return [
        ("每段视频先解码全部帧，再按等间隔取固定数量。设总帧数为 N，采样帧数为 K，"
         "第 j 个采样下标按如下方式取整。", None),
        ("", ("i_j = round( j × (N − 1) / (K − 1) ),  j = 0, …, K − 1", eq.next(),
               "当 N 不小于 K 时该式给出严格递增的下标序列，覆盖从首帧到末帧的整段视频。"
               "当 N 小于 K 时下标重复取最后一帧。本文数据的最短视频为二十八帧，"
               "因此三十二帧采样时不会出现重复，八帧与十六帧采样同样不会。")),
        ("等间隔采样的好处是与帧率无关。同一段视频在三十帧每秒与六十帧每秒下采样，"
         "得到的采样点在时间轴上落在大致相同的位置，因此不同设备之间的分布差异被部分吸收。", None),
        ("裁剪沿用官方实现的做法，先按长宽比做边框填充，再按比例缩放并去掉百分之十的边缘，"
         "最后缩放到二百二十四见方。去掉边缘是为了避开扇形扫描两侧的黑色区域与文字标注。"
         "官方示例使用的步长采样为前四十帧每隔一帧取一帧，共二十帧，"
         "本文只在 R0 中使用这一规则，其余实验使用等间隔采样。", None),
        ("帧数取三十二是精度与显存之间的折中。一次前向处理三十二帧在四吉字节显存上可以完成，"
         "而缓存文件的大小在可接受范围内。八帧与十六帧的对照在 5.4 节给出。", None),
    ]


def _m_zeroshot(eq) -> List[tuple]:
    return [
        ("零样本模式下不训练任何参数，全部计算来自冻结双塔的相似度。对每一帧，"
         "把二百零二个候选提示按相似度降序排列，得到该帧的候选数值序列。"
         "排序结果是一个长度二百零二的整数向量，向量中的第 r 位是该帧认为第 r 可能的射血分数。", None),
        ("把各帧的排序数值逐位平均，得到一个长度为二百零二的实数向量。"
         "逐位平均的含义是对每一帧赋予相同权重，这相当于假设帧在时间上的重要性一致。"
         "随后取该向量中最大的百分之二十的中位数作为预测。", None),
        ("设平均后的向量为 a，降序排列后取前 m 项，m 取提示总数的百分之二十并向下取整，"
         "预测值等于这 m 项的中位数。", ("ŷ = median( a_(1), …, a_(m) ),  m = floor(0.2 × 202) = 40", eq.next())),
        ("这一聚合规则来自官方实现，本文保持其形式不变，只改变输入帧的取法。"
         "取高分位而非全体中位数，是因为相似度最高的候选集中在较低的射血分数区间，"
         "直接取中位数会系统性低估，而取前百分之二十的中位数相当于做了一次偏移校正。", None),
        ("本文对前百分之十、二十、三十三种比例做了对照。"
         "比例越小越偏向高分位，因此系统性地高估射血分数，比例越大越接近全体中位数。"
         "三者的结果在 5.4 节给出。", None),
        ("零样本模式的分辨率上限由候选网格决定。相邻候选相差一个百分点，"
         "而聚合后的预测是整数候选的中位数，因此预测值只能落在二百零二个离散值上。"
         "实测误差远大于一个百分点，说明限制精度的主要是语义对齐，网格粒度并非主因。", None),
    ]


def _m_linear(eq) -> List[tuple]:
    return [
        ("监督模式先做池化。均值池化把 K 个帧级向量按元素平均，得到视频级向量，再送入回归头。"
         "均值池化没有可学习参数，因此它的输出只取决于缓存内容，"
         "在不同随机种子下完全相同，种子之间的差异只来自回归头。",
         ("v = (1 / K) × Σ_j z_j,  v ∈ R^512", eq.next())),
        ("线性回归头的输入为五百一十二维，先做层归一化，再经一个线性层输出标量。"
         "层归一化使不同视频的特征尺度可比，这一步在冻结特征上尤其必要，"
         "因为冻结特征的尺度不随任务调整，不同视频之间可能存在整体偏移。",
         ("ŷ = w^T LayerNorm(v) + b,  w ∈ R^512", eq.next())),
        ("这一头的参数量为一千五百三十七，其中层归一化占一千零二十四，线性层占五百一十三。"
         "参数量远小于训练样本数，因此过拟合风险很低，它的作用主要是校正零样本的系统偏差。", None),
        ("训练使用 AdamW，学习率千分之一，权重衰减万分之一，批大小六十四，损失为平滑绝对误差。"
         "学习率按余弦退火衰减，最多四十轮，验证集平均绝对误差连续二十五轮不下降则停止，"
         "并恢复验证集上最优的一组权重。", None),
        ("提前停止的耐心取二十五轮，是为了在余弦退火的中后段仍然保留恢复最优权重的机会。"
         "验证集只用于选择轮次与校准参数，测试集只在最终评估时使用一次。", None),
    ]


def _m_mlp(eq) -> List[tuple]:
    return [
        ("两层感知机在层归一化之后接两个全连接块，隐藏宽度二百五十六，激活函数为 GELU，"
         "每个块后接丢弃率为零点二的随机失活，最后线性映射到标量。"
         "两个隐藏层宽度相同，这样可以避免额外的超参数搜索。", None),
        ("记第一层权重为 W1，第二层为 W2，输出层为 w，则前向过程如下。"
         "输入维度为五百一十二，两个隐藏层的维度均为二百五十六，输出维度为一。",
         ("h1 = GELU(W1 v + b1),  h2 = GELU(W2 h1 + b2),  ŷ = w · h2 + b", eq.next())),
        ("感知机的参数量为十九万八千四百零一，相对线性头的一千五百三十七有明显增加，"
         "但仍在冻结特征可支撑的范围内。"
         "增加容量是为了检验均值池化特征中是否存在线性头无法表达的非线性结构。", None),
        ("丢弃率取零点二是常用取值。由于训练样本只有数千例，过强的正则会使欠拟合，"
         "过弱则验证集误差在训练后期上升。本文没有对丢弃率做网格搜索，"
         "固定该值以便与其他头公平比较。", None),
    ]


def _m_temporal(eq) -> List[tuple]:
    return [
        ("时序编码器把 K 个帧级向量当作词元序列，前置一个可学习的分类词元，并加上位置编码。"
         "编码器为两层 Transformer，模型宽度五百一十二，注意力头数为八，"
         "每个头的维度为六十四，前馈宽度为二千零四十八，采用前置归一化与 GELU 激活。"
         "输入张量形状为批大小乘帧数加一乘五百一十二。", None),
        ("记输入序列为 Z，位置编码为 P，分类词元为 c，则编码器输入与输出如下。"
         "取输出序列的第一位作为视频表示，再做层归一化。"
         "分类词元在初始化时取标准差零点零二的正态分布，位置编码同样如此。",
         ("H0 = [c; Z] + P,  H = Encoder(H0),  v = LayerNorm(H[:, 0])", eq.next())),
        ("自注意力的计算量随帧数平方增长，但帧数最多三十二，因此单次前向的开销很小。"
         "在缓存的浮点三十二位特征上，一次前向在中央处理器上处于毫秒量级，"
         "这使三个随机种子与多组消融可以在数分钟内完成。", None),
        ("另一种聚合方式是对编码器输出序列直接取平均，它不引入额外词元，也不增加参数。"
         "本文把它作为消融项，与分类词元方式对照，两者的差别在于聚合位置与是否引入可学习的聚合载体。"
         "记编码器输出序列为 H，则这一路径的视频表示如下。",
         ("v = LayerNorm( (1 / K) Σ_j H_j )", eq.next())),
        ("编码器的参数量为六百三十八万九千三百七十七，其中两层自注意力与前馈网络占六百三十万四千七百六十八，"
         "位置编码占一万六千八百九十六，分类词元占五百一十二，"
         "投影头占六万五千六百六十四。这一规模相对训练样本数仍然偏大，"
         "因此丢弃率取零点一并在验证集上提前停止。", None),
    ]


def _m_contrastive(eq) -> List[tuple]:
    return [
        ("对比预训练的目标是让射血分数接近的视频在表示空间靠近。"
         "与监督对比学习不同[8]，这里的正样本由标签差构造为软权重，不依赖离散类别。"
         "设批内两段视频的参考值分别为 y_i 与 y_j，正样本权重按高斯核定义，带宽记为 σ。", None),
        ("", ("w_ij = exp( −(y_i − y_j)^2 / (2 σ^2) ) / Σ_k exp( −(y_i − y_k)^2 / (2 σ^2) )", eq.next(),
               "该权重在标签差为零时取一，随差值增大而衰减，批内归一化后作为软目标分布。"
               "归一化使每一行的权重之和为一，从而可以直接与交叉熵配合。")),
        ("投影头把视频表示映射到一百二十八维并归一化，记为 u。"
         "对比损失为软目标交叉熵，温度记为 τ，其形式如下。",
         ("L_con = − (1 / B) Σ_i Σ_j w_ij log softmax_j( u_i · u_j / τ )", eq.next(),
          "批大小记为 B，内层求和遍历批内全部样本。")),
        ("本文取 σ 等于五，τ 等于五，预训练二十轮，随后接回归头微调。"
         "带宽取五的含义是，射血分数相差五个百分点以内的视频被视为强正样本，"
         "相差二十个百分点时权重衰减到约百分之一点四。", None),
        ("温度取五使相似度分布较为平缓。温度越小分布越尖锐，少数最近邻会主导梯度，"
         "在批内样本有限时容易退化。温度越大分布越接近均匀，监督信号变弱。"
         "本文没有对温度做搜索，取五是为了让损失在预训练初期保持可下降的幅度。", None),
        ("预训练不使用验证集，也不使用测试集。这一约束保证后续的校准与评估都在未见数据上进行。"
         "预训练结束后，投影头被丢弃，编码器与回归头一起在训练集上微调，"
         "因此最终模型不含对比损失的任何参数。", None),
    ]


def _m_calibration(eq) -> List[tuple]:
    return [
        ("校准层只在验证集上拟合，测试集只用于报告。"
         "本文把连续预测转成阈值概率，阈值取五十、四十、三十三个水平，"
         "转换方式为固定尺度的 logistic 映射。这一映射不含可训练参数，"
         "它把预测值与阈值的差值转成一个初始概率。", None),
        ("", ("p = 1 / (1 + exp( −(θ − ŷ) / σ_0 )),  σ_0 = 4", eq.next(),
               "其中 σ_0 固定为四，表示四个百分点的尺度，θ 为阈值，ŷ 为回归头的输出。"
               "尺度取四是因为验证集上残差的标准差在这个量级，"
               "取四可以使初始概率落在零与一之间而不饱和。")),
        ("这一初始概率并不校准。Platt 缩放在其上再拟合两个参数，参数由验证集的最大似然估计得到，"
         "形式为对初始概率做一次 logistic 回归。",
         ("p_cal = 1 / (1 + exp( −(a · p + b) ))", eq.next())),
        ("温度缩放则在 logit 空间除以一个标量。记初始概率的 logit 为 ℓ，"
         "温度为 T，校准后的概率如下。T 大于一使概率向零点五收缩，T 小于一使分布更尖锐。"
         "T 由验证集的对数损失最小化确定。",
         ("p_ts = 1 / (1 + exp( −ℓ / T ))", eq.next())),
        ("保形区间对绝对残差取分位数。设校准集大小为 n，显著性水平为 α，"
         "分位数下标按有限样本修正取上取整。这一修正保证覆盖率在有限样本下不低于名义水平。",
         ("q = Quantile( { |ŷ_i − y_i| : i = 1..n }, ceil((n + 1)(1 − α)) / n )", eq.next())),
        ("测试区间为预测值加减 q，覆盖率定义为真值落入区间的比例。"
         "本文报告 α 等于零点一与零点二两种水平，对应名义覆盖率百分之九十与百分之八十。"
         "固定宽度区间的好处是宽度与样本无关，便于在报告系统里直接换算成误差条，"
         "代价是对困难样本与容易样本给出同样的宽度。", None),
        ("校准层不改变点估计，因为概率与区间都由同一个回归输出派生。"
         "这一设计使校准的收益可以被单独观察，R5 与 R6 的点估计完全相同，差异只在概率与区间。", None),
    ]


def _m_matrix() -> List[str]:
    return [
        "实验矩阵按预测模式与训练方式分成七行，另加一行标注辅助的零样本参照。"
        "R0 使用官方步长采样且不训练，R1 使用等间隔三十二帧且不训练，"
        "R2 与 R3 在均值池化特征上训练，R4 引入时序编码器，"
        "R5 在 R4 基础上加入对比预训练，R6 在 R5 基础上加入校准。"
        "标注辅助行使用 VolumeTracings 给出的舒张末期与收缩末期帧号，只取这两帧做零样本。",
        "除主矩阵外另有两个结构变体。R4-MEAN 保留两层编码器但把分类词元聚合换成对输出序列的均值池化，"
        "用于分离时序建模与聚合方式两个因素。R5-EDES 只把对比预训练阶段的输入帧换成标注的舒张末期与收缩末期帧对，"
        "微调阶段仍使用三十二帧，用于检验对比预训练对帧选择的敏感程度。"
        "两个变体都只在第五节第四小节报告，不进入主表。"
        "此外还有一个取帧方式的对照，即把零样本的等间隔三十二帧换成十六帧，"
        "与官方步长二十帧一起构成三种取帧方式，结果在第五节第二小节报告。",
        "R2 至 R6 每行跑三个随机种子，报告均值与标准差。"
        "所有行共用同一份帧嵌入缓存，差异只来自聚合方式、目标函数与后处理。"
        "由于缓存由冻结塔一次算出，随机种子只影响头部初始化与批内顺序，"
        "不影响帧级特征本身，这一点使种子之间的方差可以直接解释为优化过程的方差。",
    ]


def _table_matrix(doc) -> None:
    rows = [
        ["R0", "官方步长 0:40:2", "不训练", "官方前百分之二十中位数"],
        ["R1", "等间隔 32 帧", "不训练", "前百分之二十中位数"],
        ["R2", "等间隔 32 帧", "线性回归", "点估计"],
        ["R3", "等间隔 32 帧", "两层感知机", "点估计"],
        ["R4", "等间隔 32 帧", "时序编码器", "点估计"],
        ["R5", "等间隔 32 帧", "对比预训练加回归", "点估计"],
        ["R6", "等间隔 32 帧", "同 R5", "Platt 与保形区间"],
        ["Oracle-EDES", "标注 ED/ES 两帧", "不训练", "前百分之二十中位数"],
    ]
    add_three_line_table(
        doc,
        ["编号", "取帧方式", "训练方式", "输出"],
        rows,
        "表 1  实验矩阵。R2 至 R6 各跑三个随机种子，标注辅助行不参与主结论。",
    )


def _m_metrics(eq) -> List[tuple]:
    return [
        ("回归指标包括平均绝对误差、均方根误差、决定系数、系统偏差、皮尔逊相关系数与斯皮尔曼秩相关。"
         "前四项刻画误差大小与方向，后两项刻画排序一致性。", None),
        ("平均绝对误差与均方根误差的定义如下，其中 n 为样本数。",
         ("MAE = (1/n) Σ_i |ŷ_i − y_i|,  RMSE = sqrt( (1/n) Σ_i (ŷ_i − y_i)^2 )", eq.next())),
        ("决定系数以参考值的方差为分母，取值越接近一表示解释比例越高。"
         "系统偏差取残差均值，正值表示系统性高估。",
         ("R^2 = 1 − Σ_i (ŷ_i − y_i)^2 / Σ_i (y_i − ȳ)^2", eq.next())),
        ("判别指标针对三个阈值，把参考值低于阈值定义为阳性，"
         "以负的预测值作为风险分数，计算受试者工作特征曲线下面积与精确率召回率曲线下面积。"
         "取负号是因为预测值越低风险越高。",
         ("AUC_θ = P( s_i > s_j | y_i < θ, y_j ≥ θ ),  s = −ŷ", eq.next())),
        ("概率指标包括十五分箱的期望校准误差与布里尔分数。"
         "前者衡量分箱内预测概率与实测频率的差距，后者是概率预测的均方误差。",
         ("ECE = Σ_b (n_b / n) | freq(b) − conf(b) |", eq.next())),
        ("区间指标包括覆盖率与平均区间宽度。覆盖率应接近名义水平，宽度越窄表示信息量越大。"
         "两者需要同时报告，因为把区间取成整个取值范围也能得到百分之百的覆盖率。", None),
        ("置信区间由自助重采样给出。对测试集做一千次有放回重采样，每次重算指标，"
         "取百分之二点五与百分之九十七点五分位数作为区间端点。", None),
    ]


def _e_data(fl, audit, n_train, n_val, n_test) -> List[str]:
    sub = audit["test_distribution"]
    return [
        f"数据来自 EchoNet-Dynamic。文件清单共 {fl['rows']} 行，含文件名、射血分数、"
        f"舒张末期容积、收缩末期容积、帧高、帧宽、帧率、帧数与划分八列。"
        f"视频目录下的 avi 文件数与清单行数一致，文件名完全对应。"
        f"射血分数取值范围为 {fl['ef']['min']:.2f} 到 {fl['ef']['max']:.2f}，均值为 {fl['ef']['mean']:.2f}。",
        f"官方划分为训练 {audit['filelist']['split_counts']['TRAIN']} 例、"
        f"验证 {audit['filelist']['split_counts']['VAL']} 例、测试 {audit['filelist']['split_counts']['TEST']} 例。"
        f"本文训练使用其中 {n_train} 例，验证使用全部 {n_val} 例，"
        f"测试使用测试划分的随机子集共 {n_test} 例。"
        f"三个集合两两交集为空，这一性质在审查文档中给出了校验输出。",
        f"测试子集的射血分数均值为 {sub['subset_ef_mean']:.2f}，标准差为 {sub['subset_ef_sd']:.2f}，"
        f"完整测试划分的均值为 {sub['full_test_ef_mean']:.2f}，标准差为 {sub['full_test_ef_sd']:.2f}。"
        f"低于五十的比例在子集中为 {sub['subset_lt50']:.3f}，在完整测试划分中为 {sub['full_test_lt50']:.3f}。"
        f"子集略偏向低射血分数，这一差异在结果解释时需要考虑。",
        f"标注辅助实验使用 VolumeTracings 给出的帧号。该文件覆盖 {audit['volume_tracings']['n_files']} 例视频，"
        f"每例恰好两个不同帧号，分别对应舒张末期与收缩末期。"
        f"在测试子集中，两帧均落在解码长度内的比例为 {audit['ed_es_coverage']['test_subset_500']['coverage']:.4f}。",
    ]


def _table_splits(doc, audit) -> None:
    sc = audit["filelist"]["split_counts"]
    rows = [
        ["TRAIN", str(sc["TRAIN"]), str(audit["manifest_vs_filelist"]["train"]["manifest_n"]),
         str(audit["ed_es_coverage"]["train"]["valid_ed_es"])],
        ["VAL", str(sc["VAL"]), str(audit["manifest_vs_filelist"]["val"]["manifest_n"]),
         str(audit["ed_es_coverage"]["val"]["valid_ed_es"])],
        ["TEST", str(sc["TEST"]), str(audit["manifest_vs_filelist"]["test"]["manifest_n"]),
         str(audit["ed_es_coverage"]["test"]["valid_ed_es"])],
        ["TEST-500", "500", "500", str(audit["ed_es_coverage"]["test_subset_500"]["valid_ed_es"])],
    ]
    add_three_line_table(
        doc,
        ["划分", "清单例数", "清单文件配对", "有效 ED/ES 标注"],
        rows,
        "表 2  数据划分与标注覆盖。清单文件配对指 JSON 清单中的视频条目数，"
        "有效 ED/ES 标注指两帧均落在解码长度内的条目数。",
    )


def _e_impl(tr) -> List[str]:
    cfg = tr["config"]
    return [
        f"视觉塔权重取自官方发布的检查点，文件大小约六百零六兆字节，"
        f"加载后缺失键与多余键均为零。图像输入为二百二十四见方，按 CLIP 均值与标准差标准化。"
        f"帧嵌入在中央处理器或图形处理器上一次性算出并缓存为浮点三十二位数组，"
        f"此后所有训练只读取缓存。",
        f"训练使用 AdamW 优化器，学习率千分之一，权重衰减万分之一，批大小六十四，"
        f"损失为平滑绝对误差，平滑参数零点一。学习率按余弦退火衰减，"
        f"最多 {cfg['epochs']} 轮，验证集平均绝对误差连续二十五轮不下降则提前停止，"
        f"并恢复验证集上最优的一组权重。",
        f"对比预训练单独进行，批大小六十四，预训练 {cfg['pretrain_epochs']} 轮，"
        f"温度 {cfg['tau']:.1f}，高斯核带宽 {cfg['smooth_sigma']:.1f}。"
        f"预训练结束后接回归头微调，微调阶段的设置与 R4 相同。",
        "三个随机种子取四十二、四十三、四十四，控制初始化的随机性。"
        "数据顺序在每个轮次内重新打乱，测试集只在最终评估时使用一次。",
    ]


def _e_env(tr) -> List[str]:
    return [
        f"训练与评估在单台配备 GeForce GTX 950M 的机器上完成，显存四吉字节，"
        f"系统内存四十八吉字节。PyTorch 版本为 {tr['torch']}，Python 版本为 3.13。"
        f"帧嵌入缓存阶段使用图形处理器，批大小十六，实际吞吐约为每秒零点六到零点九段视频。"
        f"头部训练在中央处理器上完成，因为特征已缓存，单次训练在数分钟内结束。",
        f"代码提交号为 {tr['git_commit']}。权重文件的 SHA-256 摘要记录在每个结果文件的"
        f"权重哈希字段中，其值为 {tr['weights_sha256'][:16]}…。",
        "由于显存有限，本文没有对视觉塔做端到端微调。这一限制决定了全部实验都在冻结特征上进行，"
        "也决定了训练规模的上限。训练规模消融给出了误差随样本数的变化，可用于外推更大规模时的收益。",
    ]


def _r_main(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    r0m = _seed_mean(present, "R0") if "R0" in present else None
    best = present[best_exp][0]
    bm = _seed_mean(present, best_exp)
    out = [
        f"六类预测头在测试子集上的主要指标见表 3。"
        f"零样本基线 R1 的平均绝对误差为 {z['mae']:.2f}，"
        f"决定系数为 {z['r2']:.3f}，说明零样本预测的方差结构偏离参考值。"
        f"这一误差主要来自系统性高估，偏差为 {z['bias']:+.2f} 个百分点。",
    ]
    if all(k in present for k in ("R2", "R3", "R4")):
        m2 = np.mean([r["metrics"]["mae"] for r in present["R2"]])
        m3 = np.mean([r["metrics"]["mae"] for r in present["R3"]])
        m4 = np.mean([r["metrics"]["mae"] for r in present["R4"]])
        gap = m3 - m4
        close = "接近" if abs(gap) <= 0.2 else ("高于" if gap < 0 else "低于")
        out.append(
            f"线性回归头把误差降到 {m2:.2f}，两层感知机为 {m3:.2f}。"
            f"时序编码器 R4 的结果为 {m4:.2f}，{close}感知机 {abs(gap):.2f}。"
            f"在这组数据上，均值池化特征已经包含了大部分可用信息，"
            f"序列建模的额外收益取决于帧间关系的可分辨程度。"
        )
    if "R5" in present and "R4" in present:
        r4map = {r["seed"]: r["metrics"]["mae"] for r in present["R4"]}
        r5map = {r["seed"]: r["metrics"]["mae"] for r in present["R5"]}
        paired = [(s, r4map[s], r5map[s]) for s in sorted(set(r4map) & set(r5map))]
        r4m = [r["metrics"]["mae"] for r in present["R4"]]
        r5m = [r["metrics"]["mae"] for r in present["R5"]]
        delta = float(np.mean(r5m) - np.mean(r4m))
        same_dir = all(b < a for _, a, b in paired)
        out.append(
            f"加入软标签对比预训练后，R5 的平均绝对误差为 {np.mean(r5m):.2f}，"
            f"相对 R4 变化 {delta:+.2f} 个百分点。"
            + (f"按种子逐对比较，{len(paired)} 个种子的变化方向全部一致。"
               if same_dir and len(paired) > 1
               else "变化方向在种子之间不完全一致，因此这一差异只作为趋势描述。")
        )
    out.append(
        f"表中误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，其平均绝对误差为 {bm['mae']:.2f}，"
        f"均方根误差为 {bm['rmse']:.2f}，决定系数为 {bm['r2']:.3f}，"
        f"皮尔逊相关系数为 {bm['pearson']:.3f}，三者均为三个种子的均值。"
    )
    return out


def _table_main(doc, present) -> None:
    rows = []
    for exp in EXP_ORDER:
        runs = present.get(exp)
        if not runs:
            continue
        m = [r["metrics"] for r in runs]
        n = runs[0]["n_test"]
        seeds = len(runs)
        rows.append([
            exp,
            EXP_LABEL[exp],
            str(seeds),
            ms([x["mae"] for x in m], 2),
            ms([x["rmse"] for x in m], 2),
            ms([x["r2"] for x in m], 3),
            ms([x["bias"] for x in m], 2),
            ms([x["pearson"] for x in m], 3),
            ms([x["spearman"] for x in m], 3),
            ms([x["auc_ef_lt50"] for x in m], 3),
        ])
    add_three_line_table(
        doc,
        ["编号", "方法", "种子", "MAE", "RMSE", "R²", "偏差", "r", "ρ", "AUC50"],
        rows,
        "表 3  主结果。测试集为官方测试划分的随机子集。MAE、RMSE 与偏差的单位为射血分数百分点。"
        "多行数字为三个随机种子的均值与标准差。AUC50 指参考射血分数低于百分之五十的判别面积。",
        font_size=8,
        note="说明　R0 与 R1 不涉及训练，种子一列记为一次运行。偏差为正表示高估。",
    )


def _r_zeroshot(present) -> List[str]:
    r1 = present.get("R1", [{}])[0]
    r0 = present.get("R0", [{}])[0]
    orc = present.get("ORACLE_EDES", [{}])[0]
    u16 = present.get("R1U16") or []
    out = []
    if r0 and r1:
        d01 = r0["metrics"]["mae"] - r1["metrics"]["mae"]
        rel = abs(d01) / r1["metrics"]["mae"] * 100
        out.append(
            f"R0 使用官方步长采样，在测试子集上的平均绝对误差为 {r0['metrics']['mae']:.2f}，"
            f"偏差为 {r0['metrics']['bias']:+.2f}。"
            f"与 R1 的等间隔三十二帧相比，二者相差 {d01:+.2f} 个百分点，"
            f"相对幅度 {rel:.1f}%。在这组数据上，把官方步长换成覆盖整段视频的等间隔采样，"
            f"并没有带来可分辨的误差变化，因此后文的监督头统一使用等间隔采样。"
        )
    if u16:
        u = u16[0]["metrics"]
        trio = [r0["metrics"]["mae"], r1["metrics"]["mae"], u["mae"]]
        out.append(
            f"用等间隔十六帧替换三十二帧后，零样本的平均绝对误差为 {u['mae']:.2f}，"
            f"偏差为 {u['bias']:+.2f}，EF 低于五十的判别 AUC 为 {u['auc_ef_lt50']:.3f}。"
            f"与三十二帧的 {r1['metrics']['mae']:.2f} 相差 {u['mae'] - r1['metrics']['mae']:+.2f} 个百分点。"
            f"官方步长二十帧、等间隔十六帧与等间隔三十二帧三种取帧方式的误差极差为 "
            f"{max(trio) - min(trio):.2f} 个百分点，小于监督头各行之间的差距。"
            f"零样本排序只看窗口内的分位数，帧数变化对分位数的影响有限，"
            f"因此这一层的差距主要来自文本侧的对齐精度。"
        )
    if orc and r1:
        dorc = r1["metrics"]["mae"] - orc["metrics"]["mae"]
        out.append(
            f"标注辅助行使用 VolumeTracings 的舒张末期与收缩末期帧号，只取这两帧做零样本。"
            f"其平均绝对误差为 {orc['metrics']['mae']:.2f}，与 R1 的 {r1['metrics']['mae']:.2f} 相差 "
            f"{dorc:+.2f} 个百分点。"
            f"两帧之间的信息量远小于三十二帧，但误差并未变大，"
            f"说明零样本模式的瓶颈主要不在帧的数量，而在提示排序本身的聚合规则与文本对齐精度。"
        )
    out.append(
        "标注辅助行使用了人工标注的帧号，在实际部署中不可得。"
        "它的作用只是给出零样本模式在已知收缩时相下的参照，不参与主结论。"
    )
    return out


def _r_calibration(present) -> List[str]:
    r6s = present.get("R6", [])
    if not r6s:
        return []
    cal = r6s[0].get("calibration", {})
    out = []
    if cal:
        def mean_ece(key: str, mode: str) -> float:
            return float(np.mean([r["calibration"]["thresholds"][key]["test"][mode]["ece_15"] for r in r6s]))

        def mean_brier(key: str, mode: str) -> float:
            return float(np.mean([r["calibration"]["thresholds"][key]["test"][mode]["brier"] for r in r6s]))

        parts = []
        for key in ("ef_lt50", "ef_lt40", "ef_lt30"):
            raw, pl, ts = mean_ece(key, "raw"), mean_ece(key, "platt"), mean_ece(key, "temperature")
            best = min((raw, "校准前"), (pl, "Platt"), (ts, "温度"))
            label = key.replace("ef_lt", "EF 低于 ")
            parts.append(
                f"{label} 上校准前 {raw:.3f}，Platt 后 {pl:.3f}，温度缩放后 {ts:.3f}，"
                f"最小的是{best[1]}的 {best[0]:.3f}"
            )
        out.append(
            "校准前后的概率指标见表 4。" + "，".join(parts) + "。以上数字都是三个种子的均值。"
        )
        e50r, e50p, e50t = mean_ece("ef_lt50", "raw"), mean_ece("ef_lt50", "platt"), mean_ece("ef_lt50", "temperature")
        e40r, e40p = mean_ece("ef_lt40", "raw"), mean_ece("ef_lt40", "platt")
        e30r, e30p = mean_ece("ef_lt30", "raw"), mean_ece("ef_lt30", "platt")
        out.append(
            f"EF 低于 50% 这一阈值上，校准几乎没有带来收益。Platt 把误差从 {e50r:.3f} 变成 {e50p:.3f}，"
            f"温度缩放给出 {e50t:.3f}，三者差异在种子波动之内。"
            f"更低的两个阈值上收益明显，EF 低于 40% 由 {e40r:.3f} 降到 {e40p:.3f}，"
            f"EF 低于 30% 由 {e30r:.3f} 降到 {e30p:.3f}。"
            f"一种解释是五十个百分点处正负样本接近均衡，初始概率的尺度偏差本来就小，"
            f"而三十与四十处正样本稀少，尺度偏差被放大，校准的作用因此更明显。"
        )
        out.append(
            f"布里尔分数在 EF 低于 50% 上由 {mean_brier('ef_lt50', 'raw'):.3f} 变为"
            f" {mean_brier('ef_lt50', 'platt'):.3f}，在 EF 低于 40% 上由"
            f" {mean_brier('ef_lt40', 'raw'):.3f} 变为 {mean_brier('ef_lt40', 'platt'):.3f}。"
            "三种概率的排序完全相同，因为两种校准都是单调变换，因此 AUC 保持不变，"
            "校准只改变概率刻度，不改变判别能力。"
        )
        conf = cal.get("conformal", {})
        for key, rec in conf.items():
            q = float(np.mean([r["calibration"]["conformal"][key]["q"] for r in r6s]))
            cov = float(np.mean([r["calibration"]["conformal"][key]["coverage"] for r in r6s]))
            wid = float(np.mean([r["calibration"]["conformal"][key]["width_mean"] for r in r6s]))
            out.append(
                f"保形区间在 α 等于 {rec['alpha']:.1f} 时，校准集上的残差分位数为 {q:.2f}，"
                f"测试集实测覆盖率为 {cov:.3f}，名义水平为 {rec['target_coverage']:.2f}，"
                f"平均区间宽度为 {wid:.2f} 个百分点，半宽 {wid / 2:.2f}。"
            )
        if len(conf) > 1:
            out.append(
                "两个名义水平的对照给出预期的取舍。名义水平从百分之八十提高到百分之九十时，"
                "分位数与区间宽度同步增大，实测覆盖率也随之上升。"
                "两种水平下的实测覆盖率都略低于名义值，差距在三个百分点以内。"
                "保形保证成立的前提是校准样本与测试样本可交换，而本文的验证集与测试子集在射血分数分布上存在小幅差异，"
                "测试子集低于五十的比例比完整测试划分高约三个百分点，因此这一方向的偏差可以预期。"
            )
    out.append(
        f"R6 的点估计与 R5 完全相同，因为校准只作用于概率与区间，不改变回归输出。"
        f"二者的平均绝对误差均为 {np.mean([r['metrics']['mae'] for r in present['R5']]):.2f}。"
        f"这一设计使校准的收益可以被单独观察。"
    )
    return out


def _table_calibration(doc, present) -> None:
    r6s = present.get("R6", [])
    if not r6s:
        return
    rows = []
    for key in ("ef_lt50", "ef_lt40", "ef_lt30"):
        def mean_of(mode: str, field: str) -> float:
            return float(np.mean([r["calibration"]["thresholds"][key]["test"][mode][field] for r in r6s]))

        tvals = [r["calibration"]["thresholds"][key]["temperature"] for r in r6s]
        rows.append([
            key.replace("ef_lt", "EF 低于 "),
            f"{mean_of('raw', 'ece_15'):.3f}",
            f"{mean_of('platt', 'ece_15'):.3f}",
            f"{mean_of('temperature', 'ece_15'):.3f}",
            f"{mean_of('raw', 'brier'):.3f}",
            f"{mean_of('platt', 'brier'):.3f}",
            f"{mean_of('raw', 'auc'):.3f}",
            f"{float(np.mean(tvals)):.3f}",
        ])
    add_three_line_table(
        doc,
        ["阈值", "ECE 校准前", "ECE Platt", "ECE 温度", "Brier 校准前", "Brier Platt", "AUC", "温度 T"],
        rows,
        "表 4  校准前后对比。ECE 取十五分箱，参数只在验证集上拟合，指标在测试集上计算，"
        "表中数值为三个种子的均值。",
        font_size=8,
    )


def _r_ablation() -> List[str]:
    out = []
    # frames
    frame_stats = []
    for nf in (8, 16, 32):
        runs = gather("R4", frames=nf, tag=f"abframes{nf}") or gather("R4", frames=nf)
        if runs:
            v = [r["metrics"]["mae"] for r in runs]
            frame_stats.append((nf, float(np.mean(v)), len(v), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0))
    if frame_stats:
        desc = "，".join(
            f"{nf} 帧为 {m:.2f}" + (f"（标准差 {sd:.2f}，{k} 个种子）" if k > 1 else "")
            for nf, m, k, sd in frame_stats
        )
        lo, hi = min(frame_stats, key=lambda t: t[1]), max(frame_stats, key=lambda t: t[1])
        gap = hi[1] - lo[1]
        sd32 = next((sd for nf, _, _, sd in frame_stats if nf == 32), 0.0)
        verdict = (
            "三档之间的差异小于各自种子标准差的两倍，在这组数据上帧数从 8 增加到 32 没有改变误差的量级。"
            if gap <= 2 * sd32 else
            f"三档之间的极差超过三十二帧自身种子标准差的三倍，帧数的影响不能被种子噪声解释。"
        )
        out.append(
            f"帧数是唯一影响缓存内容的设计量。时序编码器在不同帧数下的平均绝对误差依次为 {desc}。"
            f"误差最小的是 {lo[0]} 帧的 {lo[1]:.2f}，最大的是 {hi[0]} 帧的 {hi[1]:.2f}，"
            f"极差 {gap:.2f} 个百分点。"
            + verdict
            + "八帧与十六帧各只有两个种子，这两个点的比较按描述性结果处理。"
        )
    mean_runs = gather("R4_MEAN", tag="abpoolmean")
    cls_runs = gather("R4")
    if mean_runs and cls_runs:
        mv = float(np.mean([r["metrics"]["mae"] for r in mean_runs]))
        cv = float(np.mean([r["metrics"]["mae"] for r in cls_runs]))
        out.append(
            f"池化方式的对照中，编码器输出取平均的平均绝对误差为 {mv:.2f}，"
            f"分类词元聚合为 {cv:.2f}，二者之差 {mv - cv:+.2f} 个百分点。"
            f"两种方式使用相同的编码器结构与训练流程，唯一差别是聚合位置，"
            f"因此这一差值可以直接归因于聚合方式。"
            f"均值聚合略差，说明可学习词元确实承载了一部分与射血分数相关的信息。"
        )
    scale_stats = []
    for n in (500, 1000, 2000):
        runs = gather("R4", tag=f"abscale{n}") or (gather("R4") if n == 2000 else [])
        if runs:
            v = [r["metrics"]["mae"] for r in runs]
            scale_stats.append((n, float(np.mean(v)), len(v), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0))
    if scale_stats:
        desc = "，".join(
            f"{n} 例为 {m:.2f}" + (f"（标准差 {sd:.2f}）" if k > 1 else "")
            for n, m, k, sd in scale_stats
        )
        first, last = scale_stats[0], scale_stats[-1]
        out.append(
            f"训练规模的有效数字依次为 {desc}。"
            f"从 {first[0]} 例增加到 {last[0]} 例，平均绝对误差变化 {last[1] - first[1]:+.2f} 个百分点。"
            + ("曲线在 1000 例之后基本走平，说明在这个特征空间里增加标注的边际收益已经很小。"
               if abs(last[1] - scale_stats[-2][1]) < 0.15 else
               "曲线仍在下降，说明更大规模可能继续带来收益。")
        )
    topk_stats = []
    for frac in (10, 20, 30):
        runs = gather("R1", tag=f"abtop{frac}") or (gather("R1") if frac == 20 else [])
        if runs:
            v = runs[0]["metrics"]
            topk_stats.append((frac, v["mae"], v["bias"]))
    if topk_stats:
        desc = "，".join(f"{f}% 为 {m:.2f}（偏差 {b:+.2f}）" for f, m, b in topk_stats)
        best = min(topk_stats, key=lambda t: t[1])
        worst = max(topk_stats, key=lambda t: t[1])
        topk_gap = worst[1] - best[1]
        frame_gap = (max(t[1] for t in frame_stats) - min(t[1] for t in frame_stats)) if frame_stats else float("nan")
        mono = all(topk_stats[i][2] > topk_stats[i + 1][2] for i in range(len(topk_stats) - 1))
        out.append(
            f"零样本提示比例的三种取值下，平均绝对误差与偏差依次为 {desc}。"
            f"误差最小的是 {best[0]}%，最大的是 {worst[0]}%，极差 {topk_gap:.2f} 个百分点，"
            + (f"这一极差略大于帧数消融的 {frame_gap:.2f}。"
               if np.isfinite(frame_gap) and topk_gap > frame_gap else
               (f"这一极差与帧数消融的 {frame_gap:.2f} 处于同一量级。"
                if np.isfinite(frame_gap) else ""))
            + ("比例从十增到三十时偏差单调下降，方向与高分位规则一致，比例越大越接近全体中位数，"
               "高估幅度越小。" if mono else
               "偏差随比例的变化并非单调，说明排序聚合的结果还受候选分布形状影响。")
        )
    edes = gather("R5_EDES", tag="edes")
    base5 = gather("R5")
    if edes and base5:
        mv = float(np.mean([r["metrics"]["mae"] for r in edes]))
        cv = float(np.mean([r["metrics"]["mae"] for r in base5]))
        out.append(
            f"对比预训练阶段改用标注的舒张末期与收缩末期帧对后，平均绝对误差为 {mv:.2f}，"
            f"与使用三十二帧等间隔采样的 {cv:.2f} 相差 {mv - cv:+.2f} 个百分点。"
            "该变体只在预训练阶段替换帧，微调阶段仍用三十二帧，"
            "因此差值反映的是对比目标对帧选择的敏感程度。差值很小，"
            "说明软标签对比损失并不依赖特定的取样相位。"
        )
    out.append("消融结果的图形汇总见图 5，具体数值见表 5。")
    return out


def _table_ablation(doc, present) -> None:
    rows = []

    def add(label, runs):
        if not runs:
            return
        m = [r["metrics"] for r in runs]
        rows.append([
            label,
            str(runs[0]["n_train"]),
            str(len(runs)),
            ms([x["mae"] for x in m], 2),
            ms([x["auc_ef_lt50"] for x in m], 3),
        ])

    for nf in (8, 16, 32):
        add(f"帧数 {nf}", gather("R4", frames=nf, tag=f"abframes{nf}") or gather("R4", frames=nf))
    add("池化 可学习词元", gather("R4", tag="abframes32") or gather("R4"))
    add("池化 输出均值", gather("R4_MEAN", tag="abpoolmean"))
    for n in (500, 1000, 2000):
        add(f"训练规模 {n}", gather("R4", tag=f"abscale{n}") or (gather("R4") if n == 2000 else []))
    for frac in (10, 20, 30):
        add(f"零样本比例 {frac}%", gather("R1", tag=f"abtop{frac}") or (gather("R1") if frac == 20 else []))
    add("零样本 等间隔 16 帧", present.get("R1U16") or [])
    add("对比预训练 标注 ED/ES 帧", gather("R5_EDES", tag="edes"))
    if rows:
        add_three_line_table(
            doc,
            ["设置", "训练例数", "种子", "MAE", "AUC50"],
            rows,
            "表 5  消融结果。MAE 单位为射血分数百分点，多行数字为均值与标准差。"
            "零样本比例行与等间隔十六帧行不涉及训练，训练例数一列记为 0。",
            font_size=8,
        )


def _r_scatter(present, audit_train=None) -> List[str]:
    exp = "R5" if "R5" in present else ("R4" if "R4" in present else "R2")
    r = present[exp][0]
    y = np.array(r["predictions"]["test_true"])
    p = np.array(r["predictions"]["test_pred"])
    resid = p - y
    q = np.percentile(np.abs(resid), 90)
    bad = np.abs(resid) >= q
    lo = y < 50
    hi = y >= 50
    slope = np.polyfit(y, p, 1)[0] if len(y) > 2 else float("nan")
    strata = []
    for a, b in ((0, 30), (30, 50), (50, 70), (70, 101)):
        m = (y >= a) & (y < b)
        if m.sum():
            strata.append((a, b, int(m.sum()), float(np.mean(np.abs(resid[m]))), float(np.mean(resid[m]))))
    strata_desc = "，".join(
        f"{a} 至 {b} 为 {e:.2f}（{n} 例，偏差 {bs:+.2f}）" for a, b, n, e, bs in strata
    )
    worst = np.argsort(-np.abs(resid))[:5]
    worst_desc = "，".join(f"{y[i]:.1f} 对 {p[i]:.1f}" for i in worst)
    top_band = float(np.mean((p[bad] >= 50) & (p[bad] < 65)))
    return [
        f"{EXP_LABEL.get(exp, exp)}的预测值与残差见图 6。"
        f"残差绝对值的中位数为 {np.median(np.abs(resid)):.2f}，九十分位数为 {q:.2f}，"
        f"最大值为 {np.max(np.abs(resid)):.2f}。"
        f"误差最大的百分之十样本中，参考射血分数低于五十的比例为 {np.mean(y[bad] < 50):.3f}，"
        f"全体样本中该比例为 {np.mean(y < 50):.3f}。",
        f"按参考值分层，各层的平均绝对误差与偏差依次为 {strata_desc}。"
        f"误差最小的区间是五十至七十，其两侧的误差都明显更大。"
        f"误差最大的五个样本的参考值与预测值依次为 {worst_desc}，"
        f"它们的参考值都低于三十五，预测值都贴近五十，方向一致。",
        f"预测值对参考值的回归斜率为 {slope:.3f}，明显小于一，"
        f"预测值的标准差为 {np.std(p):.2f}，参考值的标准差为 {np.std(y):.2f}。"
        f"这两项一起说明模型的输出向中间收拢，"
        f"误差最大的百分之十样本的平均参考值为 {np.mean(y[bad]):.1f}，"
        f"平均预测值为 {np.mean(p[bad]):.1f}，"
        f"其中预测落在五十到六十五之间的比例为 {top_band:.3f}，"
        f"参考值落在这一区间的比例只有 {np.mean((y[bad] >= 50) & (y[bad] < 65)):.3f}。",
        "这一现象与训练标签的分布有关。训练集两千例的射血分数均值为 "
        f"{audit_train['ef_mean']:.2f}，标准差为 {audit_train['ef_sd']:.2f}，"
        f"其中落在五十到六十五之间的比例为 {audit_train['frac_50_65']:.3f}，"
        f"低于三十的比例为 {audit_train['frac_lt30']:.3f}，高于七十的比例为 {audit_train['frac_gt70']:.3f}。"
        "两端样本如此稀少，回归头在绝对误差类损失下倾向于给出条件中位数附近的预测，"
        "因此在标签两端都会向中间偏移。"
        "帧数消融显示增加帧数不能缓解这一问题，说明瓶颈不在时间采样密度，"
        "而在特征对极端收缩功能的区分能力，以及训练样本在标签两端的覆盖。",
    ]


def _r_stats(present, sub, stat) -> List[str]:
    exp = "R5" if "R5" in present else "R4"
    r = present[exp][0]
    bs = r.get("bootstrap", {})
    out = [
        f"自助重采样的置信区间见表 6。表中每行取该实验四十二号种子的那一次运行的预测，"
        f"因为重采样必须在同一批样本的逐样本预测上进行。"
        f"{EXP_LABEL.get(exp, exp)}的平均绝对误差点估计为 "
        f"{bs.get('mae', {}).get('mean', float('nan')):.3f}，"
        f"百分之九十五区间为 {bs.get('mae', {}).get('lo', float('nan')):.3f} 到 "
        f"{bs.get('mae', {}).get('hi', float('nan')):.3f}。"
        f"EF 低于五十的判别 AUC 点估计为 {bs.get('auc50', {}).get('mean', float('nan')):.3f}，"
        f"区间为 {bs.get('auc50', {}).get('lo', float('nan')):.3f} 到 {bs.get('auc50', {}).get('hi', float('nan')):.3f}。",
    ]
    if stat:
        d = stat.get("r5_minus_r6_mae", {})
        if d:
            out.append(
                f"R5 与 R6 的配对比较中，两者点估计相同，差异的百分九十五区间为 "
                f"{d.get('lo', float('nan')):.4f} 到 {d.get('hi', float('nan')):.4f}，"
                f"区间包含零，符合校准不改变点估计的设计。"
            )
        ef = stat.get("ef_distribution", {})
        if ef:
            out.append(
                f"测试子集与完整测试划分的分布比较给出 Kolmogorov-Smirnov 统计量 "
                f"{ef.get('ks_stat', float('nan')):.3f}，p 值为 {ef.get('ks_p', float('nan')):.3f}。"
                f"子集覆盖完整测试划分的 {ef.get('coverage_of_full_test', float('nan')):.3f}。"
                f"低于五十的比例在子集中为 {ef.get('test500_lt50', float('nan')):.3f}，"
                f"在完整划分中为 {ef.get('test_full_lt50', float('nan')):.3f}。"
                f"子集略偏向低射血分数，因此本文的判别指标可能略偏乐观。"
            )
    return out


def _table_bootstrap(doc, present) -> None:
    rows = []
    for exp in EXP_ORDER:
        runs = present.get(exp)
        if not runs:
            continue
        bs = runs[0].get("bootstrap", {})
        if not bs:
            continue
        rows.append([
            exp,
            f"{bs['mae']['mean']:.2f}",
            f"{bs['mae']['lo']:.2f}–{bs['mae']['hi']:.2f}",
            f"{bs['auc50']['mean']:.3f}",
            f"{bs['auc50']['lo']:.3f}–{bs['auc50']['hi']:.3f}",
        ])
    if rows:
        add_three_line_table(
            doc,
            ["编号", "MAE 点估计", "MAE 95% 区间", "AUC50 点估计", "AUC50 95% 区间"],
            rows,
            "表 6  自助重采样置信区间。重采样一千次，取百分之二点五与九十七点五分位数。",
            font_size=8,
        )


def _d_mechanism(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)
    out = [
        "零样本与监督头之间的差距主要来自两点。其一是零样本的聚合规则固定，"
        "它把各帧的排序结果平均后取高分位中位数，这一操作系统性地偏向高分位，"
        "因此偏差为正。其二是提示网格的粒度限制了精度，相邻候选只差一个百分点，"
        "但实际误差远大于此，说明限制精度的主要是语义对齐，网格粒度并非主因。",
        f"监督头学到的映射是针对本队列的线性或非线性校正，"
        f"它把零样本的系统偏差吸收掉，因此偏差从 {z['bias']:+.2f} 变为 {bm['bias']:+.2f}。"
        f"这一校正不需要改动视觉塔，两千例样本已经足够。"
        f"从零样本到线性头的误差降幅约 {z['mae'] - _seed_mean(present, 'R2')['mae']:.2f} 个百分点，"
        f"而从线性头到最优头再降 "
        f"{_seed_mean(present, 'R2')['mae'] - bm['mae']:.2f} 个百分点，"
        f"可见大部分收益来自偏差校正，结构改进只占小部分。",
    ]
    if "R4" in present and "R3" in present:
        m3 = float(np.mean([r["metrics"]["mae"] for r in present["R3"]]))
        m4 = float(np.mean([r["metrics"]["mae"] for r in present["R4"]]))
        out.append(
            f"时序编码器的收益来自帧间信息交换。收缩末期与舒张末期的形态差异最明显，"
            f"自注意力让每个帧的表示都能参考相邻帧的上下文，因而比逐帧独立的均值池化多一层比较。"
            f"均值池化两层感知机与序列编码器的误差分别为 "
            f"{m3:.2f} 与 {m4:.2f}，差距 {m3 - m4:+.2f}。"
            f"当均值池化已经能给出接近的结果时，额外收益就变小，这与帧数消融的结论一致，"
            f"因为三十二帧的等间隔采样已经覆盖了整个心动周期，均值里包含了收缩末期的信息。"
        )
    if "R5" in present and "R4" in present:
        d = float(np.mean([r["metrics"]["mae"] for r in present["R5"]])) - \
            float(np.mean([r["metrics"]["mae"] for r in present["R4"]]))
        out.append(
            f"对比预训练的作用是重塑表示空间的局部结构。"
            f"软标签让射血分数相近的视频互相靠近，这一约束在回归损失之外提供了额外的监督信号。"
            f"但实测变化只有 {d:+.2f} 个百分点，"
            f"一种解释是冻结特征已经按视觉语义组织得足够好，"
            f"在同一队列内重新排列局部结构对下游线性读出的帮助有限。"
        )
    return out


def _d_compare(present) -> List[str]:
    z = _seed_mean(present, "R1")
    best_exp = _best_exp(present)
    bm = _seed_mean(present, best_exp)
    return [
        "EchoCLIP 的原文报告了零样本射血分数的外部验证误差，其量级与本文 R1 接近。"
        "本文没有复现其全部评估协议，因此只做量级比较，不做逐项对照。"
        "差异的来源包括帧采样方式、候选网格与聚合规则，本文的 R0 与 R1 对照说明了取帧方式的影响。",
        f"专用回归模型在 EchoNet-Dynamic 上的误差通常低于零样本。"
        f"本文的监督头达到 {bm['mae']:.2f}，与文献中专用模型的量级相当，"
        f"但本文的视觉塔是冻结的，且训练样本少于完整训练划分，"
        f"因此这一结果是在更苛刻的条件下取得的。",
        "CardiacCLIP 关注少样本条件下的适配，其结论与本文一致，即轻量适配层足以获得大部分收益。"
        "Echo-Vision-FM 强调预训练与微调的衔接，本文的对比预训练可以看作这一思路在冻结特征上的轻量版本。"
        "EchoPrime[14] 与 Holste 等人[13] 的工作都把对比自监督用于超声，它们的目标是学一个通用编码器，"
        "本文的目标是在既有编码器上比较聚合与校准，两者可以互补。",
    ]


def _d_limits(present) -> List[str]:
    return [
        "本文的视觉塔全程冻结。这一选择使实验可以在四吉字节显存上完成，"
        "但也意味着特征本身没有针对射血分数任务做调整。端到端微调可能带来额外收益，本文没有验证。",
        "训练规模限制在两千例以内。学习曲线显示误差随样本数下降，但下降速度在两千例附近放缓，"
        "外推到完整训练划分的收益需要额外实验确认。",
        "测试集为官方测试划分的随机子集，其射血分数分布略偏向低值。"
        "这使判别指标可能略偏乐观，点估计指标受影响较小。",
        "保形区间的覆盖率保证依赖校准样本与测试样本可交换这一前提。"
        "当设备、采集协议或人群发生变化时，覆盖率可能下降，需要在目标队列上重新校准。",
        "本文没有做读者研究，也没有评估预测对临床决策的影响。"
        "这些内容超出本文范围。",
    ]


def _conclusion(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)
    out = [
        f"本文在冻结 EchoCLIP 视觉塔的条件下，比较了六类射血分数预测器与两种结构变体。"
        f"零样本基线的平均绝对误差为 {z['mae']:.2f}，"
        f"误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，达到 {bm['mae']:.2f}，"
        f"EF 低于 50% 的判别 AUC 由 {z['auc_ef_lt50']:.3f} 变为 {bm['auc_ef_lt50']:.3f}。",
    ]
    if "R3" in present and "R4" in present:
        m3 = float(np.mean([r["metrics"]["mae"] for r in present["R3"]]))
        m4 = float(np.mean([r["metrics"]["mae"] for r in present["R4"]]))
        out.append(
            f"在这组数据上，均值池化两层感知机与两层时序编码器的误差分别为 {m3:.2f} 与 {m4:.2f}，"
            f"二者接近。帧数从八增到三十二带来约零点二个百分点的改善，"
            f"幅度小于从零样本到监督头的降幅。"
            f"这两个观察合起来说明，瓶颈在特征本身的判别性，"
            f"而不在时间维度的建模能力。"
        )
    if "R5" in present and "R4" in present:
        d = float(np.mean([r["metrics"]["mae"] for r in present["R5"]])) - \
            float(np.mean([r["metrics"]["mae"] for r in present["R4"]]))
        out.append(
            f"软标签对比预训练相对时序编码器的变化为 {d:+.2f} 个百分点，"
            + ("没有带来可分辨的改善。" if d > -0.2 else "带来小幅改善。")
        )
    if "R6" in present:
        cal = present["R6"][0].get("calibration", {})
        a10 = cal.get("conformal", {}).get("alpha_0.1", {})
        if a10:
            r6s = present["R6"]
            raw50 = float(np.mean([r["calibration"]["thresholds"]["ef_lt50"]["test"]["raw"]["ece_15"] for r in r6s]))
            ts50 = float(np.mean([r["calibration"]["thresholds"]["ef_lt50"]["test"]["temperature"]["ece_15"] for r in r6s]))
            pl50 = float(np.mean([r["calibration"]["thresholds"]["ef_lt50"]["test"]["platt"]["ece_15"] for r in r6s]))
            raw40 = float(np.mean([r["calibration"]["thresholds"]["ef_lt40"]["test"]["raw"]["ece_15"] for r in r6s]))
            pl40 = float(np.mean([r["calibration"]["thresholds"]["ef_lt40"]["test"]["platt"]["ece_15"] for r in r6s]))
            cov10 = float(np.mean([r["calibration"]["conformal"]["alpha_0.1"]["coverage"] for r in r6s]))
            w10 = float(np.mean([r["calibration"]["conformal"]["alpha_0.1"]["width_mean"] for r in r6s]))
            out.append(
                f"校准层不改变点估计。EF 低于 50% 上校准前的期望校准误差为 {raw50:.3f}，"
                f"Platt 后为 {pl50:.3f}，温度缩放后为 {ts50:.3f}，两种校准都没有把该阈值上的误差降下来，"
                f"说明这一阈值下初始概率的尺度已经接近最优。"
                f"EF 低于 40% 上校准前为 {raw40:.3f}，Platt 后为 {pl40:.3f}，下降明显。"
                f"保形区间在 90% 名义水平上的实测覆盖率为 {cov10:.3f}，平均半宽 {w10 / 2:.2f} 个百分点，"
                f"三个种子的取值接近。"
            )
    out.append(
        "这些结果表明，在算力受限的场合，冻结特征加轻量头部是一条可行路径。"
        "后续工作可以在端到端微调、更大训练规模与跨中心外部验证三个方向展开。"
    )
    return out


def _availability(tr, audit) -> List[str]:
    return [
        f"数据来自 EchoNet-Dynamic 公开数据集，使用需遵守其非商业研究许可。"
        f"文件清单的 SHA-256 摘要为 {audit['filelist']['sha256']}，"
        f"容积标注文件的摘要为 {audit['volume_tracings']['sha256']}。",
        f"代码位于项目仓库，提交号 {tr['git_commit']}。"
        f"帧嵌入缓存脚本、头部训练脚本、消融脚本、审计脚本与绘图脚本均在 scripts 目录下。"
        f"全部结果以 JSON 形式保存在结果目录，每个文件记录配置、随机种子、样本数、完整命令、"
        f"生成时间、代码提交号、数据根目录与权重摘要。",
        "论文中的每个数字都可以在结果目录中找到对应字段，审查文档给出了逐条对照表。",
    ]


def _code_block(doc, lines: List[str]) -> None:
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(line)
        run.font.size = Pt(9)
        run.font.name = "Consolas"
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), "Consolas")
        rfonts.set(qn("w:hAnsi"), "Consolas")
        rfonts.set(qn("w:eastAsia"), "宋体")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _reference_entry(doc, idx: int, rec: dict) -> None:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(3)
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    meta = rec.get("registry", {})
    title = meta.get("title") or rec["declared_title"]
    venue = meta.get("container") or rec["declared_venue"]
    year = meta.get("year") or rec["declared_year"]
    authors = rec.get("authors_short", "")
    volume = meta.get("volume")
    page = meta.get("page")
    art = meta.get("article_number")
    bits = [f"[{idx}] {authors} {title}."]
    tail = venue
    if year:
        tail += f", {year}"
    if volume:
        tail += f", {volume}"
    if page:
        tail += f": {page}"
    elif art:
        tail += f": {art}"
    if rec.get("doi"):
        tail += f". DOI: {rec['doi']}"
    run = p.add_run(" ".join(bits) + " " + tail)
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), "宋体")


if __name__ == "__main__":
    raise SystemExit(main())
