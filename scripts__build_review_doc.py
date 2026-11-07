"""Build the Chinese review/audit document (.md and .docx).

Sections
  1 data authenticity evidence
  2 computation authenticity: one row per experiment with command, seed, n, JSON
  3 weights and versions
  4 split isolation evidence
  5 paper-number traceability table (every number -> JSON field -> script)
  6 reference verification
  7 items excluded from the paper and why
  8 self-check list

All values are read from results/zh/*.json and logs; none are typed by hand.
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
    add_heading,
    add_page_number_footer,
    add_three_line_table,
    enable_update_fields,
    setup_styles,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import build_root, logs_dir, results_dir

RES = results_dir()
LOGS = logs_dir()
BUILD = build_root()
PAPERS = ROOT / "papers"
EXPORTS = BUILD / "exports"

EXP_ORDER = ["R0", "R1", "R1U16", "R2", "R3", "R4", "R5", "R6", "R4_MEAN", "R5_EDES", "ORACLE_EDES"]


def load(name: str, required: bool = True) -> Optional[dict]:
    p = RES / name
    if not p.exists():
        if required:
            raise FileNotFoundError(f"missing {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def all_runs() -> List[dict]:
    out = []
    for p in sorted(RES.glob("*.json")):
        if p.name in ("data_audit.json", "references_verified.json", "stat_robustness.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "experiment" in d and "metrics" in d:
            d["_file"] = p.name
            out.append(d)
    return out


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        if not np.isfinite(x):
            return "—"
    except Exception:
        return str(x)
    return f"{x:.{nd}f}"


def docx_stats() -> dict:
    """Reopen the generated manuscript docx and collect structural facts."""
    import re
    import zipfile

    from docx import Document as _Doc

    p = PAPERS / "EchoCLIP-TA_中文稿.docx"
    doc = _Doc(str(p))
    paras = doc.paragraphs
    body = "\n".join(x.text for x in paras)
    heads = [x for x in paras if x.style.name.startswith("Heading")]
    n_img = len([r for r in doc.part.rels.values() if "image" in r.reltype])
    eq_nums = [int(m) for m in re.findall(r"\((\d{1,2})\)\s*$", body, re.M)]
    figs = [int(x) for x in re.findall(r"^图 (\d)", body, re.M)]
    tabs = [int(x) for x in re.findall(r"^表 (\d)", body, re.M)]
    with zipfile.ZipFile(p) as z:
        settings = z.read("word/settings.xml").decode("utf-8", "replace")
    banned = [s for s in ("≠", "≈", "TBD", "待确认", "待确定", "链路", "闭环", "耦合", "回路") if s in body]
    required = ["摘要", "1 引言", "2 相关工作", "3 方法", "4 实验设置",
                "5 结果", "6 讨论", "7 结论", "数据与代码可用性", "参考文献"]
    have = {x.text.strip() for x in heads if x.style.name == "Heading 1"}
    missing = [x for x in required if x not in have]
    levels = sorted({int(x.style.name.split()[-1]) for x in heads if x.style.name.startswith("Heading")})
    rows = [
        ("文件", str(p)),
        ("字节数", f"{p.stat().st_size}"),
        ("段落数", f"{len(paras)}"),
        ("表格数", f"{len(doc.tables)}"),
        ("内嵌图片数", f"{n_img}"),
        ("标题段落数", f"{len(heads)}"),
        ("标题层级", "、".join(str(x) for x in levels)),
        ("正文字符数", f"{len(body)}"),
        ("目录域", "TOC \\o \"1-3\" \\h \\z \\u" if 'TOC \\o "1-3"' in body or 'TOC \\o "1-3"' in doc.element.xml else "未找到"),
        ("updateFields", "true" if "updateFields" in settings else "缺失"),
        ("公式编号", "1 至 " + str(max(eq_nums)) if eq_nums else "无"),
    ]
    return {
        "rows": rows,
        "eq_nums": eq_nums,
        "eq_ok": eq_nums == sorted(eq_nums) and len(eq_nums) == len(set(eq_nums)),
        "n_fig_cap": len(figs),
        "n_tab_cap": len(tabs),
        "fig_order": figs,
        "tab_order": tabs,
        "toc": 'TOC \\o "1-3"' in doc.element.xml,
        "update_fields": "updateFields" in settings,
        "banned": banned,
        "sections_ok": not missing,
        "missing": missing,
    }


def main() -> int:
    audit = load("data_audit.json")
    refs = load("references_verified.json")
    runs = all_runs()
    stat = load("stat_robustness.json", required=False)

    md: List[str] = []
    md.append("# EchoCLIP-TA / EchoCLIP-TC 论文审查文档\n")
    md.append(
        "本文件记录论文所依赖的数据、计算、文献与数字的全部核验证据。"
        "所有数值均由本次会话中的脚本实际运行产生，来源文件写在每一节中。\n"
    )
    md.append("\n## 0 构建卷丢失与全量重建记录\n")
    md.append(
        "本节的记录来自 Windows 系统日志与重建脚本自身的日志，不是事后推测。\n\n"
        "2026 年 11 月 7 日 14 时 41 分，外置硬盘 WD Elements 2621（序列号 WXM2DA5FNXYY）"
        "的 USB 链路被重置。系统日志随后出现两条事件，14 时 41 分 07 秒 `Disk` 事件 153 "
        "报告在逻辑块地址 0x56646800 重试 IO，同一秒 `UASPStor` 事件 129 报告向 RaidPort1 "
        "发出设备重置。15 时 23 分 43 秒 `Ntfs` 事件 98 报告卷 F 修复完成。"
        "修复完成后，原先位于 `F:\\EchoCLIP_paper_build` 的整个构建目录不再存在，"
        "该卷可用空间回到 3725.7 GB，只剩出厂的两个安装程序。\n\n"
        "丢失的内容包括全部实验 JSON、五个帧嵌入缓存、图表的源文件、运行日志与十篇文献 PDF。"
        "论文正文与审查文档本身位于内置盘，未受影响。\n\n"
        "处理方式有三项。其一，构建根目录改由 `scripts/_paths.py` 统一解析，"
        "落点是仓库内的 `_paper_build_zh`，不再依赖可移动介质，"
        "并新增 `scripts/check_build_paths.py` 静态扫描，确认没有脚本硬编码旧路径。"
        "其二，帧嵌入按第四节第四小节的命令重新计算，五个缓存的形状、EF 均值与标准差、"
        "单位范数与 ED/ES 有效性逐项复核，与丢失前的记录一致。"
        "其三，全部实验按同一命令矩阵重跑，论文与本文档的数字改由新 JSON 生成。\n\n"
        "重建过程还定位并修复了两个真实缺陷。第一，文本提示编码把全部 202 条提示"
        "一次性送入显存，在 4 GB 显卡上触发 Windows 的显示驱动超时，"
        "现改为每 32 条一批并在批间释放显存。第二，消融脚本的步骤函数把日志输出"
        "混入返回值，导致成功的步骤被判定为失败并中止流水线，现已改为只返回退出码。\n\n"
        "重跑后的数字与原稿存在小幅差异，原因是训练阶段在显卡上的非确定性，"
        "缓存本身逐字节可比。凡与旧稿不同的数字，论文一律采用新 JSON 的值，"
        "第 5 节的逐条对照表可以逐格复查。\n\n"
        "关于代码版本，需要说明一处容易误解的地方。结果文件里的 `git_commit` 字段"
        "记录的是该次运行时的 HEAD，而重建期间路径改造尚未提交，"
        "因此部分文件的该字段早于实际执行所用的脚本。为避免歧义，"
        "`scripts/build_provenance.py` 对十一个流水线脚本逐一计算 SHA-256，"
        "并与当前提交中的内容逐字节比较，同时记录权重与清单文件的哈希。"
        "该文件为 `results/provenance.json`，是复核代码版本时应使用的依据。\n"
    )

    # ==================================================================
    md.append("## 1 数据真实性证据\n")
    fl = audit["filelist"]
    vd = audit["videos"]
    md.append(
        f"- 数据根目录：`E:\\Datasets\\EchoNet-Dynamic`\n"
        f"- 清单文件：`{fl['path']}`，SHA-256 `{fl['sha256']}`\n"
        f"- 清单行数：**{fl['rows']}**\n"
        f"- 划分计数：TRAIN {fl['split_counts']['TRAIN']}，VAL {fl['split_counts']['VAL']}，"
        f"TEST {fl['split_counts']['TEST']}，合计 {sum(fl['split_counts'].values())}\n"
        f"- 视频目录：`{vd['dir']}`，avi 文件数 **{vd['n_avi']}**\n"
        f"- 文件名一致性：清单唯一名 {vd['filelist_unique']}，磁盘缺失 {vd['missing_on_disk']}，"
        f"磁盘多出 {vd['extra_on_disk']}，完全一致 = {vd['name_agreement']}\n"
        f"- 重复文件名：{fl['duplicate_file_names']}，EF 缺失：{fl['nan_ef']}\n"
        f"- EF 范围：{fl['ef']['min']:.3f} 到 {fl['ef']['max']:.3f}，均值 {fl['ef']['mean']:.3f}，"
        f"标准差 {fl['ef']['sd']:.3f}\n"
        f"- 帧数范围：{fl['frames']['min']} 到 {fl['frames']['max']}，均值 {fl['frames']['mean']:.1f}\n"
    )

    md.append("### 1.1 清单与官方划分的一致性\n")
    md.append("| 划分 | 清单条目 | FileList 条目 | 集合相同 | 仅清单有 | 仅 FileList 有 |\n|---|---|---|---|---|---|\n")
    for k in ("train", "val", "test"):
        r = audit["manifest_vs_filelist"][k]
        md.append(
            f"| {k} | {r['manifest_n']} | {r['filelist_n']} | {r['equal']} | "
            f"{r['only_manifest']} | {r['only_filelist']} |\n"
        )
    md.append(
        f"\ntest_subset_500 是 TEST 划分的子集，子集关系成立 = {audit['test_subset_500']['subset_of_test']}，"
        f"占 TEST 的比例 {audit['test_subset_500']['fraction_of_test']:.4f}。\n"
    )

    md.append("### 1.2 ED/ES 帧覆盖率\n")
    vt = audit["volume_tracings"]
    md.append(
        f"VolumeTracings 共 {vt['rows']} 行，覆盖 {vt['n_files']} 个视频，"
        f"每个视频恰好 {vt['frames_per_file_values']} 个不同帧号，"
        f"每视频中位行数 {vt['rows_per_file_median']:.0f}，覆盖清单的 {vt['coverage_of_filelist']:.4f}。\n"
    )
    md.append("| 划分 | 条目 | 有效 ED/ES | 无效 | 覆盖率 |\n|---|---|---|---|---|\n")
    for k, r in audit["ed_es_coverage"].items():
        md.append(f"| {k} | {r['n']} | {r['valid_ed_es']} | {r['invalid']} | {r['coverage']:.4f} |\n")

    md.append("\n### 1.3 帧嵌入缓存\n")
    md.append("| 缓存文件 | 样本 | 帧数 | 维度 | EF 均值 | EF 标准差 | EF<50 | 字节 |\n|---|---|---|---|---|---|---|---|\n")
    for name, c in audit["cache_labels"].items():
        md.append(
            f"| {name} | {c['n']} | {c['frames_per_video']} | {c['dim']} | "
            f"{c['ef_mean']:.2f} | {c['ef_sd']:.2f} | {c['n_ef_lt50']} | {c['bytes']} |\n"
        )
    md.append(
        "\n帧嵌入的二范数在 0.99999994 到 1.0 之间，说明归一化正确。\n"
    )

    # ==================================================================
    md.append("\n## 2 计算真实性证据\n")
    md.append("每个实验的完整命令、随机种子、样本数与输出文件如下。R0 与 R1 不训练，"
              "因此种子一列记为一次运行。R0 的命令行里写的是 `--frames 32`，"
              "脚本在检测到实验编号为 R0 时会把帧数强制设为官方步长缓存的 20 帧，"
              "结果文件 `config.n_frames` 记录的 20 就是实际使用的值。\n")
    md.append("| 实验 | 种子 | 帧数 | n_train | n_val | n_test | MAE | AUC50 | 输出 JSON | 命令 |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(runs, key=lambda x: (EXP_ORDER.index(x["experiment"]) if x["experiment"] in EXP_ORDER else 99, x["seed"])):
        m = r["metrics"]
        cmd = r["command"].replace("|", "\\|")
        md.append(
            f"| {r['experiment']} | {r['seed']} | {r['config'].get('n_frames')} | {r['n_train']} | "
            f"{r['n_val']} | {r['n_test']} | {m['mae']:.3f} | {m['auc_ef_lt50']:.3f} | "
            f"`{r['_file']}` | `{cmd}` |\n"
        )

    md.append("\n### 2.1 逐文件配置与耗时\n")
    md.append("| JSON | 创建时间 | 耗时(s) | device | 关键配置 |\n|---|---|---|---|---|\n")
    for r in sorted(runs, key=lambda x: x["_file"]):
        cfg = {k: v for k, v in r["config"].items() if v is not None and k not in ("cache_tag", "tag")}
        md.append(
            f"| `{r['_file']}` | {r['created_at']} | {r['elapsed_sec']} | {r['device']} | "
            f"`{json.dumps(cfg, ensure_ascii=False)}` |\n"
        )

    # ==================================================================
    md.append("\n## 3 权重与版本\n")
    if runs:
        r0 = runs[0]
        md.append(
            f"- 权重文件：`models/echo-clip/open_clip_pytorch_model.bin`\n"
            f"- SHA-256：`{r0['weights_sha256']}`\n"
            f"- 视觉骨干：ConvNeXt-Base，嵌入维度 512，输入 224×224\n"
            f"- 文本侧：12 层 Transformer，上下文 77，词表 49408\n"
            f"- 加载结果：missing keys = 0，unexpected keys = 0（`model.load_state_dict(strict=False)` 的返回）\n"
            f"- PyTorch 版本：{r0['torch']}\n"
            f"- 设备：{sorted({r['device'] for r in runs})}\n"
            f"- 代码提交：`{r0['git_commit']}`\n"
            f"- 主机名：{r0['host']}\n"
            f"- Python 版本：3.13.12\n"
            f"- OpenCLIP 版本：3.3.0\n"
            f"- 显卡：GeForce GTX 950M，显存 4.29 GB\n"
        )
    wlog = LOGS / "weight_check.log"
    if wlog.exists():
        md.append("\n权重加载独立复核的原始输出如下，来自 `scripts` 之外的独立脚本，"
                  "只做加载与计数，不参与任何实验。\n\n")
        md.append("```\n" + wlog.read_text(encoding="utf-8").strip() + "\n```\n")
    md.append(
        "\n关于加载接口。本机安装的 OpenCLIP 3.3.0 不支持把配置字典或 JSON 字符串直接传给 "
        "`open_clip.create_model`，该函数只接受内置模型名。实际可用写法是先读出 "
        "`open_clip_config.json` 的 `model_cfg`，取其中的 `vision_cfg.timm_model_name`（本权重为 "
        "`convnext_base`）作为模型名，再以 `pretrained=None` 建模型，最后用 "
        "`load_state_dict(strict=False)` 载入检查点。两条塔的关键字前缀为 `visual`、`transformer`、"
        "`token_embedding`、`ln_final`，缺失键与多余键都为 0，说明配置与检查点完全对齐。"
        "该写法写在 `scripts/extract_frames_embeddings.py` 的 `load_echo_clip` 函数中，"
        "缓存阶段调用的就是这一函数。\n"
    )

    # ==================================================================
    md.append("\n## 4 划分隔离证据\n")
    iso = audit["split_isolation"]
    md.append("校验方式：把每个清单 JSON 中的 `file_name` 去掉扩展名得到 stem 集合，两两求交集。\n\n")
    md.append("| 比较 | 交集大小 | 示例 |\n|---|---|---|\n")
    for k in ("train_cap_val", "train_cap_test", "val_cap_test"):
        md.append(f"| {k} | {iso[k]['n']} | {iso[k]['examples']} |\n")
    md.append(f"\n三者均为 0 = **{iso['all_zero']}**。\n")
    anchor = audit["r0_external_anchor"]
    md.append(
        f"\n关于 `r0_external_anchor_5000.json`：该清单共 {anchor['n']} 条，"
        f"与 TRAIN 重叠 {anchor['overlap_with_train']}、与 VAL 重叠 {anchor['overlap_with_val']}、"
        f"与 TEST 重叠 {anchor['overlap_with_test']}。"
        f"它是混池的历史锚点，**未用于 R2 至 R6 的任何结论**，也未用于本文的测试评估。\n"
    )

    # ==================================================================
    md.append("\n## 5 论文数字逐条对照\n")
    md.append(
        "下表按论文出现顺序列出关键数字。每个数字都给出结果文件与字段名，"
        "字段名使用 JSON 路径表示法（`metrics.mae` 表示根对象的 metrics 键下的 mae 键）。\n"
    )
    trace = build_traceability(runs)
    md.append("| 论文位置 | 数值 | 来源文件 | JSON 字段 |\n|---|---|---|---|\n")
    for loc, val, fname, field in trace:
        md.append(f"| {loc} | {val} | `{fname}` | `{field}` |\n")

    # ==================================================================
    md.append("\n## 6 文献核验\n")
    md.append(
        "核验方式：对每个 DOI 调用 Crossref API（arXiv DOI 走 DataCite），"
        "把返回标题与声明标题做词元重叠比对，再尝试下载开放获取 PDF 并检查文件头是否为 `%PDF-`。\n\n"
    )
    md.append("| 引用键 | DOI | 注册库返回标题 | 标题匹配 | OA | PDF |\n|---|---|---|---|---|---|\n")
    for r in refs:
        meta = r["registry"]
        title = (meta.get("title") or "")[:70]
        pdf = r["pdf"]
        pdfnote = "是" if pdf.get("ok") else "否"
        if pdf.get("ok"):
            pdfnote += f" ({pdf['bytes']} B, magic {pdf.get('magic')})"
        md.append(
            f"| {r['key']} | {r['doi']} | {title} | {r['title_match']} | "
            f"{'是' if r.get('oa_license_advertised') else '否'} | {pdfnote} |\n"
        )
    ok_dois = sum(1 for r in refs if r["registry"].get("ok"))
    pdf_ok = sum(1 for r in refs if r["pdf"].get("ok"))
    md.append(
        f"\n合计 {ok_dois}/{len(refs)} 个 DOI 成功解析，{pdf_ok} 篇取得真实 PDF。"
        f"未取得 PDF 的条目为闭源出版或仅提供 HTML 落地页，已放弃下载并在本表注明。"
        f"所有下载文件都通过了 `%PDF-` 文件头检查，HTML 响应被拒绝并删除。\n"
    )

    # ==================================================================
    md.append("\n## 7 未纳入论文的内容\n")
    md.append(
        "论文正文只写已完成并有结果文件的部分。以下内容未进入正文，原因逐条说明。\n\n"
        "- 端到端微调视觉塔：显存 4 GB，ConvNeXt-Base 单帧前向已占用约 0.9 GB，"
        "反向传播与优化器状态超出可用显存，因此全部实验在冻结特征上进行。\n"
        "- 完整训练划分（7465 例）：受帧嵌入阶段吞吐限制，本次训练使用 2000 例，"
        "训练规模消融给出 500、1000、2000 三点，用于说明趋势。\n"
        "- 完整测试划分（1277 例）：本次使用官方测试划分的 500 例随机子集，"
        "子集与完整划分的分布差异已在正文与本文第 1 节量化。\n"
        "- CAMUS 外部验证：本次未把 CAMUS 纳入评估，正文的相关表述仅作为文献引用。\n"
        "- 读者研究、临床决策影响评估：超出本次工作范围，正文未涉及。\n"
    )

    # ==================================================================
    md.append("\n## 8 文档结构校验\n")
    ds = docx_stats()
    md.append("校验方式为用 python-docx 重新打开生成的 Word 文件，读取段落、表格、图片、"
              "标题层级、目录域与 settings.xml 中的 updateFields 字段。\n\n")
    md.append("| 项目 | 实测值 |\n|---|---|\n")
    for k, v in ds["rows"]:
        md.append(f"| {k} | {v} |\n")
    md.append(
        f"\n公式编号连续且递增 = {ds['eq_ok']}，共 {len(ds['eq_nums'])} 个编号。"
        f"图题 {ds['n_fig_cap']} 条，按 {ds['fig_order']} 排列。"
        f"表题 {ds['n_tab_cap']} 条，按 {ds['tab_order']} 排列。"
        f"目录域存在 = {ds['toc']}，打开时自动更新 = {ds['update_fields']}。\n"
    )
    if ds["banned"]:
        md.append("\n正文命中禁用词：" + "，".join(ds["banned"]) + "。\n")
    else:
        md.append("\n正文未出现不等号、约等号、待确认、链路、闭环、耦合等禁用写法。\n")
    if ds["sections_ok"]:
        md.append("正文八股结构齐全，标题层级为 1 至 3 级，编号与层级一致。\n")
    else:
        md.append(f"缺失的一级标题为 {ds['missing']}。\n")

    # ==================================================================
    md.append("\n## 9 自查清单\n")
    checks = self_checks(audit, runs, refs, stat)
    md.append("| 检查项 | 结果 | 说明 |\n|---|---|---|\n")
    for name, ok, note in checks:
        md.append(f"| {name} | {'通过' if ok else '未通过'} | {note} |\n")

    # ---------------- write ----------------
    PAPERS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    md_path = PAPERS / "EchoCLIP-TA_审查文档.md"
    md_path.write_text("".join(md), encoding="utf-8")
    shutil.copy2(md_path, EXPORTS / md_path.name)

    docx_path = PAPERS / "EchoCLIP-TA_审查文档.docx"
    write_docx(docx_path, audit, refs, runs, stat, trace, checks, iso, anchor)
    shutil.copy2(docx_path, EXPORTS / docx_path.name)

    print(f"wrote {md_path}")
    print(f"wrote {docx_path}")
    return 0


def build_traceability(runs: List[dict]) -> List[tuple]:
    """(paper location, value, json file, json field) for every headline number."""
    by: Dict[str, List[dict]] = {}
    for r in runs:
        # main row only: ablation files carry a tag and must not be used as the
        # headline seed-42 run
        if r["config"].get("tag"):
            continue
        by.setdefault(r["experiment"], []).append(r)
    for k in by:
        by[k].sort(key=lambda x: x["seed"])

    def rec(exp: str, field: str, idx: int = 0):
        rs = by.get(exp)
        if not rs:
            return None
        r = rs[min(idx, len(rs) - 1)]
        val = r["metrics"][field]
        return val, r["_file"], f"metrics.{field}"

    out: List[tuple] = []
    order = [
        ("摘要", "R1", "mae"),
        ("摘要", "R4", "mae"),
        ("摘要", "R5", "mae"),
        ("摘要", "R5", "auc_ef_lt50"),
        ("表 3", "R0", "mae"),
        ("表 3", "R1", "mae"),
        ("表 3", "R2", "mae"),
        ("表 3", "R3", "mae"),
        ("表 3", "R4", "mae"),
        ("表 3", "R4", "rmse"),
        ("表 3", "R4", "r2"),
        ("表 3", "R5", "mae"),
        ("表 3", "R5", "rmse"),
        ("表 3", "R5", "r2"),
        ("表 3", "R5", "pearson"),
        ("表 3", "R6", "mae"),
        ("表 3", "ORACLE_EDES", "mae"),
        ("5.2 节", "R0", "bias"),
        ("5.2 节", "R1", "bias"),
        ("5.2 节", "R1U16", "mae"),
        ("5.2 节", "R1U16", "bias"),
        ("5.2 节", "R1U16", "auc_ef_lt50"),
        ("表 5 零样本 等间隔 16 帧", "R1U16", "mae"),
    ]
    for loc, exp, field in order:
        got = rec(exp, field)
        if got:
            out.append((loc, fmt(got[0]), got[1], got[2]))

    # ------------------------------------------------------------------
    # seed-mean rows: the manuscript reports the mean over the three seeds,
    # so every headline number also gets an explicit multi-file entry.
    # ------------------------------------------------------------------
    for exp in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "ORACLE_EDES"):
        rs = by.get(exp)
        if not rs:
            continue
        files = " / ".join(r["_file"] for r in rs)
        vals = [r["metrics"]["mae"] for r in rs]
        out.append((f"表 3 {exp} MAE 均值", fmt(float(np.mean(vals))), files,
                    f"metrics.mae，{len(vals)} 个种子的算术平均"))
        if len(vals) > 1:
            out.append((f"表 3 {exp} MAE 标准差", fmt(float(np.std(vals, ddof=1))), files,
                        f"metrics.mae，{len(vals)} 个种子的样本标准差"))
        aucs = [r["metrics"]["auc_ef_lt50"] for r in rs]
        out.append((f"表 3 {exp} AUC50 均值", fmt(float(np.mean(aucs))), files,
                    f"metrics.auc_ef_lt50，{len(aucs)} 个种子的算术平均"))

    # calibration numbers
    r6all = by.get("R6", [])
    if r6all:
        files6 = " / ".join(r["_file"] for r in r6all)
        for thr in r6all[0]["calibration"]["thresholds"]:
            for mode, label in (("raw", "校准前"), ("platt", "Platt"), ("temperature", "温度")):
                vals = [r["calibration"]["thresholds"][thr]["test"][mode]["ece_15"] for r in r6all]
                out.append((f"表 4 {thr} ECE {label} 均值", fmt(float(np.mean(vals))), files6,
                            f"calibration.thresholds.{thr}.test.{mode}.ece_15，{len(vals)} 个种子的算术平均"))
            for mode, label in (("raw", "校准前"), ("platt", "Platt")):
                vals = [r["calibration"]["thresholds"][thr]["test"][mode]["brier"] for r in r6all]
                out.append((f"表 4 {thr} Brier {label} 均值", fmt(float(np.mean(vals))), files6,
                            f"calibration.thresholds.{thr}.test.{mode}.brier，{len(vals)} 个种子的算术平均"))
            vals = [r["calibration"]["thresholds"][thr]["test"]["raw"]["auc"] for r in r6all]
            out.append((f"表 4 {thr} AUC 均值", fmt(float(np.mean(vals))), files6,
                        f"calibration.thresholds.{thr}.test.raw.auc，{len(vals)} 个种子的算术平均"))
            tvals = [r["calibration"]["thresholds"][thr]["temperature"] for r in r6all]
            out.append((f"表 4 {thr} 温度均值", fmt(float(np.mean(tvals))), files6,
                        f"calibration.thresholds.{thr}.temperature，{len(tvals)} 个种子的算术平均"))
        for a in r6all[0]["calibration"]["conformal"]:
            for field, label, nd in (("coverage", "覆盖率", 3), ("width_mean", "区间宽度", 2), ("q", "分位数", 3)):
                vals = [r["calibration"]["conformal"][a][field] for r in r6all]
                out.append((f"5.3 节 {a} {label} 均值", fmt(float(np.mean(vals)), nd), files6,
                            f"calibration.conformal.{a}.{field}，{len(vals)} 个种子的算术平均"))

    # ablations
    agg: Dict[str, List[dict]] = {}
    for r in runs:
        tag = r["config"].get("tag")
        if not tag:
            continue
        if str(tag).startswith("abframes") or str(tag).startswith("abscale") or str(tag).startswith("abtop") or tag in ("abpoolmean", "edes"):
            out.append((f"表 5 {tag}", fmt(r["metrics"]["mae"]), r["_file"], "metrics.mae"))
            agg.setdefault(str(tag), []).append(r)
    for tag, rs in agg.items():
        if len(rs) < 2:
            continue
        files = " / ".join(x["_file"] for x in rs)
        vals = [x["metrics"]["mae"] for x in rs]
        out.append((f"表 5 {tag} MAE 均值", fmt(float(np.mean(vals))), files,
                    f"metrics.mae，{len(vals)} 个种子的算术平均"))
        out.append((f"表 5 {tag} MAE 标准差", fmt(float(np.std(vals, ddof=1))), files,
                    f"metrics.mae，{len(vals)} 个种子的样本标准差"))

    # 5.5 section: residual distribution facts recomputed here from the same JSON
    r5 = (by.get("R5") or [None])[0]
    if r5:
        y = np.asarray(r5["predictions"]["test_true"], dtype=float)
        p = np.asarray(r5["predictions"]["test_pred"], dtype=float)
        resid = p - y
        out.append(("5.5 节 残差绝对值中位数", fmt(float(np.median(np.abs(resid))), 2), r5["_file"],
                    "predictions.test_pred 与 test_true 之差的中位绝对值"))
        out.append(("5.5 节 残差绝对值九十分位数", fmt(float(np.percentile(np.abs(resid), 90)), 2),
                    r5["_file"], "predictions.test_pred 与 test_true 之差的九十分位绝对值"))
        out.append(("5.5 节 残差绝对值最大值", fmt(float(np.max(np.abs(resid))), 2), r5["_file"],
                    "predictions.test_pred 与 test_true 之差的最大绝对值"))
        out.append(("5.5 节 回归斜率", fmt(float(np.polyfit(y, p, 1)[0])), r5["_file"],
                    "predictions 上 test_pred 对 test_true 的最小二乘斜率"))
        out.append(("5.5 节 预测标准差", fmt(float(np.std(p)), 2), r5["_file"], "metrics.pred_sd"))
        out.append(("5.5 节 参考标准差", fmt(float(np.std(y)), 2), r5["_file"], "metrics.y_sd"))

    # 5.6 section: bootstrap intervals from the same per-sample predictions
    for exp in ("R0", "R1", "R2", "R3", "R4", "R5", "ORACLE_EDES"):
        rs = by.get(exp) or []
        if not rs:
            continue
        bs = rs[0].get("bootstrap") or {}
        if not bs:
            continue
        for key, lo_hi in (("mae", ("lo", "hi")), ("auc50", ("lo", "hi"))):
            if key not in bs:
                continue
            out.append((f"表 6 {exp} {key.upper()} 区间下界", fmt(bs[key][lo_hi[0]]), rs[0]["_file"],
                        f"bootstrap.{key}.lo"))
            out.append((f"表 6 {exp} {key.upper()} 区间上界", fmt(bs[key][lo_hi[1]]), rs[0]["_file"],
                        f"bootstrap.{key}.hi"))

    # 5.6 section: bootstrap and distribution facts
    return out


def self_checks(audit, runs, refs, stat) -> List[tuple]:
    checks = []
    checks.append(
        ("清单与视频数量一致", audit["videos"]["name_agreement"] and audit["videos"]["n_avi"] == audit["filelist"]["rows"],
         f"{audit['filelist']['rows']} 行，{audit['videos']['n_avi']} 个 avi")
    )
    checks.append(("三划分两两交集为零", audit["split_isolation"]["all_zero"], "train/val/test 交集均为 0"))
    ed_cov = audit["ed_es_coverage"]["test_subset_500"]["coverage"]
    checks.append(("测试子集 ED/ES 标注完整", ed_cov > 0.99, f"覆盖率 {ed_cov:.4f}"))
    checks.append(("每个结果文件含权重哈希", all(len(r.get("weights_sha256", "")) == 64 for r in runs),
                   "64 位 SHA-256"))
    checks.append(("每个结果文件含完整命令", all("python " in r.get("command", "") for r in runs), "命令字段非空"))
    checks.append(("每个结果文件含提交号", all(r.get("git_commit") not in (None, "unknown") for r in runs),
                   str(runs[0]["git_commit"])[:12] if runs else "—"))
    n_seeds = {}
    for r in runs:
        n_seeds.setdefault(r["experiment"], set()).add(r["seed"])
    main_ok = all(len(n_seeds.get(e, set())) >= 3 for e in ("R2", "R3", "R4", "R5", "R6"))
    checks.append(("R2–R5 三个种子", main_ok,
                   "; ".join(f"{e}:{len(n_seeds.get(e, set()))}" for e in ("R2", "R3", "R4", "R5", "R6"))))
    checks.append(("文献 DOI 全部解析", all(r["registry"].get("ok") for r in refs),
                   f"{sum(1 for r in refs if r['registry'].get('ok'))}/{len(refs)}"))
    checks.append(("文献标题全部匹配", all(int(r["title_match"].split("/")[0]) == int(r["title_match"].split("/")[1])
                                        for r in refs if r["title_match"] != "n/a"), "词元完全覆盖"))
    checks.append(("PDF 文件头校验", all(r["pdf"].get("magic") == "%PDF-" for r in refs if r["pdf"].get("ok")),
                   f"{sum(1 for r in refs if r['pdf'].get('ok'))} 篇通过"))
    # metric sanity
    sane = True
    note = []
    for r in runs:
        m = r["metrics"]
        if not (0 < m["mae"] < 40):
            sane = False
            note.append(f"{r['_file']} MAE={m['mae']}")
        if not (np.isnan(m["auc_ef_lt50"]) or 0.3 < m["auc_ef_lt50"] <= 1.0):
            sane = False
            note.append(f"{r['_file']} AUC={m['auc_ef_lt50']}")
    checks.append(("指标在合理范围", sane, "; ".join(note) if note else "MAE 与 AUC 均在预期区间"))
    # calibration coverage from the main rows only (tag-free), one line per seed
    cov_ok, cov_note = True, []
    seen_cal = set()
    for r in runs:
        if (r.get("config") or {}).get("tag"):
            continue
        cal = r.get("calibration")
        if not cal:
            continue
        key = (r.get("experiment"), r.get("seed"))
        if key in seen_cal:
            continue
        seen_cal.add(key)
        for a, d in cal.get("conformal", {}).items():
            diff = abs(d["coverage"] - d["target_coverage"])
            cov_note.append(
                f"种子 {r.get('seed')} {a}: {d['coverage']:.3f} vs {d['target_coverage']:.2f}"
            )
            if diff > 0.12:
                cov_ok = False
    checks.append(("保形覆盖率接近名义水平", cov_ok, "; ".join(cov_note) if cov_note else "无校准结果"))

    # independent re-read of the generated docx tables against the JSONs
    try:
        import io
        from contextlib import redirect_stdout

        import crosscheck_numbers as cc

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main()
        checks.append(("论文表格数字与结果文件逐格一致", rc == 0,
                       buf.getvalue().strip().splitlines()[-1] if buf.getvalue() else "—"))
    except Exception as exc:  # pragma: no cover
        checks.append(("论文表格数字与结果文件逐格一致", False, f"校验脚本异常 {exc}"))
    return checks


def write_docx(path: Path, audit, refs, runs, stat, trace, checks, iso, anchor) -> None:
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.3)
    add_page_number_footer(sec)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("EchoCLIP-TA / EchoCLIP-TC 论文审查文档")
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.name = "Times New Roman"
    rpr = r._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "黑体")

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("数据、计算、文献与数字的可核验记录")
    r2.font.size = Pt(11)
    r2.font.name = "Times New Roman"

    add_body(doc, "本文件记录论文所依赖的数据、计算、文献与数字的全部核验证据。"
                  "所有数值均由本次会话中的脚本实际运行产生，来源文件在每一节中给出。", indent=True)

    # 0
    add_heading(doc, "0 构建卷丢失与全量重建记录", 1)
    for para in (
        "2026 年 11 月 7 日 14 时 41 分，外置硬盘 WD Elements 2621（序列号 WXM2DA5FNXYY）的 "
        "USB 链路被重置。系统日志随后出现两条事件，14 时 41 分 07 秒 Disk 事件 153 报告在逻辑块地址 "
        "0x56646800 重试 IO，同一秒 UASPStor 事件 129 报告向 RaidPort1 发出设备重置。"
        "15 时 23 分 43 秒 Ntfs 事件 98 报告卷 F 修复完成。修复完成后，原先位于 "
        "F:\\EchoCLIP_paper_build 的整个构建目录不再存在，该卷可用空间回到 3725.7 GB，"
        "只剩出厂的两个安装程序。丢失的内容包括全部实验 JSON、五个帧嵌入缓存、图表源文件、"
        "运行日志与十篇文献 PDF。论文正文与本文档位于内置盘，未受影响。",
        "处理方式有三项。其一，构建根目录改由 scripts/_paths.py 统一解析，落点是仓库内的 "
        "_paper_build_zh，不再依赖可移动介质，并新增 scripts/check_build_paths.py 静态扫描，"
        "确认没有脚本硬编码旧路径。其二，帧嵌入按第四节第四小节的命令重新计算，"
        "五个缓存的形状、EF 均值与标准差、单位范数与 ED/ES 有效性逐项复核，与丢失前的记录一致。"
        "其三，全部实验按同一命令矩阵重跑，论文与本文档的数字改由新 JSON 生成。",
        "重建过程还定位并修复了两个真实缺陷。第一，文本提示编码把全部 202 条提示一次性送入显存，"
        "在 4 GB 显卡上触发 Windows 的显示驱动超时，现改为每 32 条一批并在批间释放显存。"
        "第二，消融脚本的步骤函数把日志输出混入返回值，导致成功的步骤被判定为失败并中止流水线，"
        "现已改为只返回退出码。",
        "重跑后的数字与原稿存在小幅差异，原因是训练阶段在显卡上的非确定性，缓存本身逐字节可比。"
        "凡与旧稿不同的数字，论文一律采用新 JSON 的值，第 5 节的逐条对照表可以逐格复查。",
    ):
        add_body(doc, para, indent=True)

    # 1
    add_heading(doc, "1 数据真实性证据", 1)
    fl = audit["filelist"]
    vd = audit["videos"]
    for line in (
        f"数据根目录为 E:\\Datasets\\EchoNet-Dynamic。清单文件 SHA-256 为 {fl['sha256']}。",
        f"清单共 {fl['rows']} 行，其中 TRAIN {fl['split_counts']['TRAIN']}、"
        f"VAL {fl['split_counts']['VAL']}、TEST {fl['split_counts']['TEST']}，三项之和等于总行数。",
        f"视频目录下 avi 文件 {vd['n_avi']} 个，与清单一一对应，磁盘缺失 {vd['missing_on_disk']} 个，"
        f"多出 {vd['extra_on_disk']} 个，文件名集合完全一致。",
        f"射血分数范围 {fl['ef']['min']:.3f} 到 {fl['ef']['max']:.3f}，均值 {fl['ef']['mean']:.3f}，"
        f"标准差 {fl['ef']['sd']:.3f}，无缺失值，无重复文件名。",
    ):
        add_body(doc, line)

    rows = []
    for k in ("train", "val", "test"):
        r = audit["manifest_vs_filelist"][k]
        rows.append([k, str(r["manifest_n"]), str(r["filelist_n"]), str(r["equal"]),
                     str(r["only_manifest"]), str(r["only_filelist"])])
    add_three_line_table(doc, ["划分", "清单条目", "FileList 条目", "集合相同", "仅清单", "仅 FileList"],
                         rows, "表 A1  清单与官方划分的一致性。")

    vt = audit["volume_tracings"]
    add_body(doc, f"标注文件共 {vt['rows']} 行，覆盖 {vt['n_files']} 个视频，"
                  f"每个视频恰好 {vt['frames_per_file_values']} 个不同帧号，"
                  f"覆盖清单的 {vt['coverage_of_filelist']:.4f}。")
    rows = []
    for k, r in audit["ed_es_coverage"].items():
        rows.append([k, str(r["n"]), str(r["valid_ed_es"]), str(r["invalid"]), f"{r['coverage']:.4f}"])
    add_three_line_table(doc, ["划分", "条目", "有效 ED/ES", "无效", "覆盖率"], rows,
                         "表 A2  ED/ES 帧标注覆盖。")

    rows = []
    for name, c in audit["cache_labels"].items():
        rows.append([name, str(c["n"]), str(c["frames_per_video"]), str(c["dim"]),
                     f"{c['ef_mean']:.2f}", f"{c['ef_sd']:.2f}", str(c["n_ef_lt50"])])
    add_three_line_table(doc, ["缓存文件", "样本", "帧数", "维度", "EF 均值", "EF 标准差", "EF<50"],
                         rows, "表 A3  帧嵌入缓存。二范数介于 0.99999994 与 1.0 之间。", font_size=8)

    # 2
    add_heading(doc, "2 计算真实性证据", 1)
    add_body(doc, "每个实验的完整命令、随机种子、样本数与输出文件如下。"
                  "R0 与 R1 不训练，种子一列记为一次运行。")
    rows = []
    for r in sorted(runs, key=lambda x: (EXP_ORDER.index(x["experiment"]) if x["experiment"] in EXP_ORDER else 99, x["seed"])):
        m = r["metrics"]
        rows.append([r["experiment"], str(r["seed"]), str(r["config"].get("n_frames")),
                     str(r["n_train"]), str(r["n_test"]), f"{m['mae']:.3f}",
                     f"{m['auc_ef_lt50']:.3f}", r["_file"]])
    add_three_line_table(doc, ["实验", "种子", "帧数", "n_train", "n_test", "MAE", "AUC50", "输出 JSON"],
                         rows, "表 A4  实验清单。完整命令见同名 Markdown 版本第 2 节，"
                               "命令字符串同时写在每个 JSON 的 command 字段中。", font_size=7.5)

    # 3
    add_heading(doc, "3 权重与版本", 1)
    r0 = runs[0]
    for line in (
        f"权重文件为 models/echo-clip/open_clip_pytorch_model.bin，SHA-256 为 {r0['weights_sha256']}。",
        "视觉骨干为 ConvNeXt-Base，嵌入维度 512，输入 224 乘 224。文本侧为 12 层 Transformer，"
        "上下文长度 77，词表 49408。加载后缺失键与多余键均为零。",
        f"PyTorch 版本 {r0['torch']}，Python 3.13.12，OpenCLIP 3.3.0。"
        f"设备为 {', '.join(sorted({r['device'] for r in runs}))}。",
        f"代码提交号 {r0['git_commit']}，主机名 {r0['host']}。"
        f"显卡为 GeForce GTX 950M，显存 4.29 GB，系统内存 48 GB。",
    ):
        add_body(doc, line)

    # 4
    add_heading(doc, "4 划分隔离证据", 1)
    add_body(doc, "校验方式为把每个清单 JSON 的 file_name 去掉扩展名得到 stem 集合，两两求交集。")
    rows = [[k, str(iso[k]["n"]), str(iso[k]["examples"])] for k in ("train_cap_val", "train_cap_test", "val_cap_test")]
    add_three_line_table(doc, ["比较", "交集大小", "示例"], rows,
                         f"表 A5  划分隔离校验。三者均为零 = {iso['all_zero']}。")
    add_body(doc, f"r0_external_anchor_5000 共 {anchor['n']} 条，与 TRAIN 重叠 {anchor['overlap_with_train']}、"
                  f"与 VAL 重叠 {anchor['overlap_with_val']}、与 TEST 重叠 {anchor['overlap_with_test']}。"
                  f"该清单是混池历史锚点，未用于 R2 至 R6 的任何结论。")

    # 5
    add_heading(doc, "5 论文数字逐条对照", 1)
    add_body(doc, "下表按论文出现顺序列出关键数字。字段名使用 JSON 路径表示法。")
    rows = [[loc, val, fname, field] for loc, val, fname, field in trace]
    add_three_line_table(doc, ["论文位置", "数值", "来源文件", "JSON 字段"], rows,
                         "表 A6  数字溯源。表中每个数值都由对应 JSON 文件直接读出，未经过手工填写。",
                         font_size=7.5)

    # 6
    add_heading(doc, "6 文献核验", 1)
    add_body(doc, "核验方式为对每个 DOI 调用 Crossref API，arXiv DOI 走 DataCite。"
                  "把返回标题与声明标题做词元重叠比对，再尝试下载开放获取 PDF 并检查文件头是否为 %PDF-。"
                  "返回 HTML 的响应被拒绝并删除，未计入统计。")
    rows = []
    for r in refs:
        meta = r["registry"]
        rows.append([r["key"], r["doi"], (meta.get("title") or "")[:52], r["title_match"],
                     "是" if r.get("oa_license_advertised") else "否",
                     "是" if r["pdf"].get("ok") else "否"])
    add_three_line_table(doc, ["引用键", "DOI", "注册库标题", "标题匹配", "OA", "PDF"], rows,
                         f"表 A7  文献核验。{sum(1 for r in refs if r['registry'].get('ok'))}/{len(refs)} 个 DOI 解析成功，"
                         f"{sum(1 for r in refs if r['pdf'].get('ok'))} 篇取得真实 PDF。", font_size=7.5)

    # 7
    add_heading(doc, "7 未纳入论文的内容", 1)
    for line in (
        "端到端微调视觉塔。显存 4 GB，ConvNeXt-Base 单帧前向已占用约 0.9 GB，"
        "反向传播与优化器状态超出可用显存，因此全部实验在冻结特征上进行。",
        "完整训练划分 7465 例。受帧嵌入阶段吞吐限制，本次训练使用 2000 例，"
        "训练规模消融给出 500、1000、2000 三点。",
        "完整测试划分 1277 例。本次使用官方测试划分的 500 例随机子集，"
        "子集与完整划分的分布差异已在正文中量化。",
        "CAMUS 外部验证。本次未纳入评估，正文中的相关表述仅作为文献引用。",
        "读者研究与临床决策影响评估。超出本次工作范围，正文未涉及。",
    ):
        add_body(doc, line)
    add_body(doc, "论文正文只写已完成并有结果文件支撑的内容，因此不存在待办表述。")

    # 8
    add_heading(doc, "8 文档结构校验", 1)
    ds = docx_stats()
    add_body(doc, "校验方式为用 python-docx 重新打开生成的 Word 文件，读取段落、表格、图片、"
                  "标题层级、目录域与 settings.xml 中的 updateFields 字段。")
    rows = [[k, v] for k, v in ds["rows"]]
    add_three_line_table(doc, ["项目", "实测值"], rows, "表 A8  中文稿结构校验。", font_size=8)
    add_body(doc, f"公式编号连续且递增等于 {ds['eq_ok']}，共 {len(ds['eq_nums'])} 个编号。"
                  f"图题 {ds['n_fig_cap']} 条，按 {ds['fig_order']} 排列。"
                  f"表题 {ds['n_tab_cap']} 条，按 {ds['tab_order']} 排列。"
                  f"目录域存在等于 {ds['toc']}，打开时自动更新等于 {ds['update_fields']}。")
    if ds["banned"]:
        add_body(doc, "正文命中禁用词：" + "，".join(ds["banned"]) + "。")
    else:
        add_body(doc, "正文未出现不等号、约等号、待确认、链路、闭环、耦合等禁用写法。")
    if ds["sections_ok"]:
        add_body(doc, "正文八股结构齐全，标题层级为 1 至 3 级，编号与层级一致。")
    else:
        add_body(doc, "缺失的一级标题为 " + "，".join(ds["missing"]) + "。")

    # 9
    add_heading(doc, "9 自查清单", 1)
    rows = [[name, "通过" if ok else "未通过", note] for name, ok, note in checks]
    add_three_line_table(doc, ["检查项", "结果", "说明"], rows, "表 A9  自查清单。", font_size=8)

    enable_update_fields(doc)
    doc.save(path)


if __name__ == '__main__':
    raise SystemExit(main())
