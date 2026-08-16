#!/usr/bin/env python3
"""Build self-contained HTML reports + PDF for EchoCLIP-TC dual-agent deliverables."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(r"E:\Projects\20260522-EchoCLIP")
FIG = ROOT / "figures"
PAPERS = ROOT / "papers"
REPORTS = ROOT / "reports"


def b64_img(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


FIGS = {
    "fig1": FIG / "fig1_protocol_architecture.png",
    "fig2": FIG / "fig2_ablation_schematic.png",
    "fig3": FIG / "fig3_calibration_reliability_demo.png",
    "fig4": FIG / "fig4_demo_protocol_metrics.png",
    "fig5": FIG / "fig5_roadmap_bilingual.png",
    "fig6": FIG / "fig6_conformal_demo.png",
}

PROTO = {
    k: json.loads((ROOT / "checkpoints" / "protocol" / k / "metrics.json").read_text(encoding="utf-8"))
    for k in ("B0", "M1", "M2", "M4")
}

CSS = """
:root { --ink:#1a1a1a; --muted:#555; --line:#ccc; --bg:#fafaf8; --accent:#1f4e79; --warn:#8b0000; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; font-family: "Times New Roman", "SimSun", serif; color:var(--ink); background:var(--bg); line-height:1.55; }
.wrap { max-width:920px; margin:0 auto; padding:28px 22px 64px; }
header.cover { border-bottom:2px solid var(--accent); padding-bottom:18px; margin-bottom:28px; }
header.cover h1 { font-size:1.85rem; margin:0 0 10px; color:var(--accent); }
.meta { color:var(--muted); font-size:0.95rem; }
nav.toc { background:#fff; border:1px solid var(--line); padding:16px 20px; margin:22px 0 32px; }
nav.toc h2 { margin:0 0 8px; font-size:1.15rem; }
nav.toc ol { margin:0; padding-left:1.3rem; }
nav.toc a { color:var(--accent); text-decoration:none; }
nav.toc a:hover { text-decoration:underline; }
h2 { color:var(--accent); border-bottom:1px solid var(--line); padding-bottom:4px; margin-top:2.2rem; }
h3 { margin-top:1.4rem; }
figure { margin:1.4rem 0; background:#fff; border:1px solid var(--line); padding:12px; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; }
figcaption { font-size:0.92rem; color:var(--muted); margin-top:8px; }
.fig-explain { font-size:0.95rem; background:#fff; border-left:3px solid var(--accent); padding:10px 14px; margin:8px 0 18px; }
.demo { color:var(--warn); font-weight:bold; }
.todo { background:#fff3cd; border:1px solid #e0c36a; padding:2px 6px; }
table { border-collapse:collapse; width:100%; margin:12px 0 20px; background:#fff; font-size:0.92rem; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:#eef3f8; }
.small { font-size:0.88rem; color:var(--muted); }
ul.compact li { margin:0.25rem 0; }
"""


def img_block(key: str, title: str, how: str, meaning: str, conclusion: str) -> str:
    src = b64_img(FIGS[key])
    return f"""
<figure id="{key}">
  <img src="{src}" alt="{html.escape(title)}" />
  <figcaption><strong>{html.escape(title)}</strong></figcaption>
</figure>
<div class="fig-explain">
  <p><strong>How to read（如何读图）：</strong>{how}</p>
  <p><strong>Meaning（含义）：</strong>{meaning}</p>
  <p><strong>Conclusions（可下结论）：</strong>{conclusion}</p>
</div>
"""


def protocol_table_html() -> str:
    rows = []
    for k, m in PROTO.items():
        rows.append(
            "<tr>"
            f"<td>{k}</td>"
            f"<td>{html.escape(str(m.get('mae')))}</td>"
            f"<td>{html.escape(str(round(m.get('ece_ef_lt_50', float('nan')), 4)))}</td>"
            f"<td>{html.escape(str(m.get('load_source')))}</td>"
            f"<td>{'yes' if m.get('demo_mode') else 'no'}</td>"
            f"<td>{html.escape(str(m.get('n')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>ID</th><th>DEMO MAE</th><th>DEMO ECE@50</th><th>load_source</th><th>demo</th><th>n</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<p class='small demo'>DEMO ONLY — scratch_fallback / synthetic demo data. Not EchoNet clinical MAE. "
        "Published EchoCLIP external EF MAE ≈7.1% is Christensen et al., not a local result.</p>"
    )


def research_report_html() -> str:
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>EchoCLIP-TC 学术研究报告（自包含）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="cover">
  <h1>EchoCLIP-TC：面向超声心动图视觉–语言模型的时序聚合与校准评测研究报告</h1>
  <p class="meta">EchoCLIP-TC (Temporal, Calibrated) Parallel Research Report · 单文件自包含 HTML · 2026-08-16</p>
  <p class="meta">项目路径：E:\\Projects\\20260522-EchoCLIP · 公开仓库：<a href="https://github.com/Coucou2016/EchoCLIP-TC">github.com/Coucou2016/EchoCLIP-TC</a> · 临床指标：<span class="todo">待补充</span></p>
  <p class="meta">五轮协作日志：<code>reports/echoclip_tc_five_round_collab_20260816.md</code> · 英文稿：<code>papers/echoclip_tc_manuscript.md</code></p>
  <p class="demo">声明：文中 DEMO 图与 DEMO 表格仅验证流水线，不得当作 EchoNet 临床 EF MAE / AUC。磁盘检索未发现 EchoNet-Dynamic / FileList.csv。</p>
</header>

<nav class="toc" id="toc">
  <h2>目录（Table of Contents）</h2>
  <ol>
    <li><a href="#abstract">摘要 Abstract</a></li>
    <li><a href="#bg">研究背景 Background</a></li>
    <li><a href="#methods">方法 Methods</a></li>
    <li><a href="#process">实施过程 Process</a></li>
    <li><a href="#results">结果 Results</a></li>
    <li><a href="#discussion">讨论 Discussion</a></li>
    <li><a href="#conclusions">结论 Conclusions</a></li>
    <li><a href="#limitations">局限 Limitations</a></li>
    <li><a href="#collab">五轮协作摘要 Five-round log</a></li>
    <li><a href="#refs">参考文献 References</a></li>
  </ol>
</nav>

<section id="abstract">
<h2>1. 摘要（Abstract）</h2>
<p>EchoCLIP（Christensen 等，<em>Nature Medicine</em> 2024）在百万级心超视频–报告对上对比学习，支持零样本（zero-shot）左室射血分数（left ventricular ejection fraction, EF）估计。本报告描述仓库中的 <strong>EchoCLIP-TC</strong>：在冻结双塔编码器之上增加周期感知帧采样与时序聚合器（temporal aggregator），并以验证集（validation, VAL）拟合温度缩放（temperature scaling）与分割共形预测（split-conformal prediction）区间，锁定 B0/M1/M2/M4 协议。</p>
<p><strong>本地现状：</strong>单元测试与 validate 门禁通过；EchoNet-Dynamic 与官方 Hugging Face 权重缺失，故 <span class="todo">临床 EF MAE/AUC 待补充</span>。仅有 <span class="demo">DEMO</span> 合成数据冒烟指标。</p>
</section>

<section id="bg">
<h2>2. 研究背景（Background）</h2>
<p>心超具有高时间分辨率，功能评估依赖心动周期动态。EchoCLIP 公开发布路径偏帧级编码 + prompt 排序聚合；后续 EchoPrime（多视频/多切面）与 CardiacCLIP（MICCAI 2025，注意力帧融合）强调视频建模。缺口在于：在<strong>不重训百万私有数据</strong>的前提下，如何公平消融视频向量构造，并把校准/共形不确定性写成一等公民指标。</p>
<p>可模仿的写作架构：Nature Medicine 式「问题→基础模型→零样本任务→外部验证」+ MICCAI 式「基线–消融表」+ 校准可靠性图。创新点应落在<strong>协议可复现 + 时序模块 + 可信不确定度</strong>，而非虚构私有规模预训练。</p>
</section>

<section id="methods">
<h2>3. 方法（Methods）</h2>
<h3>3.1 协议定义（与 <code>echoclip/protocol.py</code> 对齐）</h3>
<ul class="compact">
  <li><strong>B0</strong>：官方风格——逐帧 embedding 上做 EF prompt 排序，再跨帧聚合 EF 标量；采样 <code>uniform</code>，论文默认 T=16。</li>
  <li><strong>M1</strong>：先对帧 embedding 做均值池化得到 z_v，再做一次 EF 排序（无参视频向量消融）；采样 <code>mixed</code>。</li>
  <li><strong>M2</strong>：训练时序 Transformer/注意力池化得到 z_v（冻结视觉/文本塔）；采样 <code>mixed</code>。</li>
  <li><strong>M4</strong>：复用 M2，在 VAL 上拟合温度与共形分位数，在 TEST 上报 ECE、覆盖率与弃权（abstention）；非 DEMO 时 cal≠test 硬失败。</li>
</ul>
<p>B0 与 M1 因排序非线性一般不等价；仓库已用单元测试固定该语义（T=1 等价、同帧等价、rank-crossing 反例）。2D embedding 约定为 <code>(T,D)</code>，批视频向量须显式变为 <code>(B,1,D)</code>。</p>
<table>
<thead><tr><th>ID</th><th>Train</th><th>Pool</th><th>Sample</th><th>Calibrate</th></tr></thead>
<tbody>
<tr><td>B0</td><td>No</td><td>frames</td><td>uniform</td><td>No</td></tr>
<tr><td>M1</td><td>No</td><td>mean</td><td>mixed</td><td>No</td></tr>
<tr><td>M2</td><td>Yes</td><td>temporal</td><td>mixed</td><td>No</td></tr>
<tr><td>M4</td><td>Reuse M2</td><td>temporal</td><td>mixed</td><td>VAL only</td></tr>
</tbody>
</table>
{img_block("fig1", "Figure 1. EchoCLIP-TC protocol architecture (schematic)",
    "从左到右阅读数据流：视频帧 → 冻结 EchoCLIP 编码器 → 时序聚合 → 视频向量 z_v；下方虚线框对应 B0/M1/M2/M4 四种评测模式。",
    "该图是方法总览，不包含任何数值性能；用于说明 TC 层「坐在」双塔之上而非重写编码器。SciencePlots 导出 PNG/PDF。",
    "可下结论：协议结构已在代码中落地。不可下结论：任何临床精度。")}
{img_block("fig2", "Figure 2. Ablation logic for B0 / M1 / M2",
    "三列对比视频向量构造：B0 每帧直接出 EF；M1 先 mean-pool；M2 用 Temporal Transformer。箭头表示信息汇聚方向。",
    "解释为何 M1 不是「官方 EchoCLIP + 简单平均 EF」的同义反复，而是无参 z_v 对照；与 PAPER.md「B0≠M1」一致。",
    "可下结论：消融逻辑清晰、可实现。不可把示意图当作已完成的 EchoNet 消融数值。")}
</section>

<section id="process">
<h2>4. 实施过程（Process）</h2>
<ol>
  <li>阅读 README / PAPER.md / 既有 reports；公开仓库
     <a href="https://github.com/Coucou2016/EchoCLIP-TC">https://github.com/Coucou2016/EchoCLIP-TC</a> 已同步代码+文档（无权重/无患者数据）。</li>
  <li>SciencePlots（<code>science, no-latex</code>）+ Times New Roman / Microsoft YaHei 重绘 Fig.1–6 至 <code>figures/</code>。</li>
  <li>nature-writing（methods）成熟化 <code>papers/echoclip_tc_manuscript.md</code>（采样策略、B0≠M1、文献 AUC 仅作引用）。</li>
  <li>磁盘检索 EchoNet / FileList.csv：未找到；仅有 <code>data/demo/</code> 合成对。AIMI 需人工申请，无法静默下载。</li>
  <li>ChatGPT 浏览器自动化：<strong>仍 blocked</strong>（无 <code>cursor-ide-browser</code>；<code>open_resource</code> → unknown agent）。执行 ≥5 轮 <strong>surrogate</strong> 文献自检 + 粘贴包；B0/M1 语义沿用
     <a href="https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6">chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6</a>。</li>
  <li>门禁：<code>unittest</code> + <code>validate.py --skip-eval</code>（见五轮日志最新记录）。</li>
</ol>
{img_block("fig5", "Figure 5. Research roadmap（中英双语文案）",
    "四个色块从左到右：公开数据、官方权重、协议评测、校准报告；标「待补充」处表示资产未到位。",
    "用于沟通项目完成度，而非展示临床结果。红色「待补充」对应磁盘上缺失的 FileList/Videos 与 hub 权重。",
    "可下结论：脚手架就绪、临床资产缺口明确。不可捏造已完成 EchoNet 评测。")}
</section>

<section id="results">
<h2>5. 结果（Results）</h2>
<h3>5.1 临床主结果</h3>
<p><span class="todo">待补充</span>：需要 EchoNet-Dynamic（FileList + Videos）与 <code>load_source=hf-hub:mkaichristensen/echo-clip</code>（非 scratch_fallback）。Christensen 外部 EF MAE ≈7.1% 与阈值 AUC（EF&lt;50/40/30 ≈0.89–0.97）仅为文献目标，非本地结果。</p>
<h3>5.2 可声称 vs 不可声称</h3>
<table>
<thead><tr><th>Claim</th><th>Allowed?</th><th>Evidence</th></tr></thead>
<tbody>
<tr><td>Protocol B0/M1/M2/M4 defined + tested</td><td>Yes</td><td><code>protocol.py</code> + unit tests</td></tr>
<tr><td>DEMO metrics I/O works</td><td>Yes (DEMO label)</td><td><code>checkpoints/protocol/*/metrics.json</code></td></tr>
<tr><td>EchoCLIP literature external MAE ≈7.1%</td><td>Cite only</td><td>Nat Med 2024</td></tr>
<tr><td>Local EchoNet EF MAE / AUC</td><td><span class="todo">No — 待补充</span></td><td>No AIMI videos on disk</td></tr>
<tr><td>DEMO MAE ranks models clinically</td><td>No</td><td>scratch weights; toy overlap</td></tr>
</tbody>
</table>
<h3>5.3 DEMO 流水线指标（非临床；T=4）</h3>
{protocol_table_html()}
{img_block("fig4", "Figure 4. DEMO protocol MAE / ECE bars",
    "左图为各协议 ID 的 DEMO EF MAE；右图为 DEMO ECE（EF&lt;50）。红色标注强调 scratch_fallback。注意 DEMO 使用 T=4，论文协议默认 T=16。",
    "数值来自 checkpoints/protocol/*/metrics.json，数据为 synthetic demo；B0/M1 DEMO MAE 同为 11.25 不代表临床等价。",
    "仅证明评测脚本可写出指标文件。严禁当作 EchoNet 或论文主表；不可用 DEMO 给 B0/M1/M2/M4 排序。")}
{img_block("fig3", "Figure 3. Calibration reliability cartoons — DEMO / synthetic",
    "横轴置信度、纵轴准确率；虚线为理想校准；柱为分箱准确率。左：故意欠校准玩具曲线；右：温度缩放后玩具改善。",
    "说明 M4「校准前后可靠性图」在论文中应如何呈现；曲线为合成，非 EchoNet 概率。M4 DEMO ECE≈0 反映玩具设定，非临床校准成功。",
    "可下结论：绘图模板可用。不可下结论：真实 ECE 改善幅度。")}
{img_block("fig6", "Figure 6. Split-conformal interval cartoon — DEMO",
    "散点为合成真值–预测 EF；阴影带宽使用 DEMO M4 的 conformal_quantile=15 示意 90% 区间（mean width=30）。",
    "帮助读者理解共形区间宽度与覆盖率的读图方式。DEMO coverage=1.0 因玩具残差与宽区间，不可外推。",
    "不可把覆盖率=1.0 的 DEMO 指标外推到临床 TEST。")}
</section>

<section id="discussion">
<h2>6. 讨论（Discussion）</h2>
<p><strong>创新点（诚实表述）：</strong>(1) 与冻结 EchoCLIP 兼容的视频级时序模块；(2) B0/M1 语义澄清与测试锁死；(3) VAL-only 温度/共形/弃权写入主指标；(4) 公开数据可复现协议。</p>
<p><strong>与同侪对照（不宣称其数字为本结果）：</strong></p>
<ul class="compact">
  <li><strong>EchoCLIP</strong>：外部 EF MAE ≈7.1% 为 B0 复现目标。</li>
  <li><strong>EchoPrime</strong>（Nature 2026;650:970–977）：&gt;12M 多切面检查级融合——作上限对照，本工作保持单 clip 冻结双塔。</li>
  <li><strong>CardiacCLIP</strong>（MICCAI 2025）：MFL+EchoZoom；文献称 1-shot EchoNet MAE 降 2.07——最接近时序邻居；无本地 EchoNet 时不宣称 few-shot SOTA。</li>
</ul>
<p><strong>不宣称：</strong>私有百万预训练、多切面全检查融合、在无 EchoNet 时宣称超越 CardiacCLIP、任何 DEMO 数值为临床 EF MAE。</p>
</section>

<section id="conclusions">
<h2>7. 结论（Conclusions）</h2>
<p>EchoCLIP-TC 仓库已具备时序、校准、协议与门禁齐全的方法学脚手架；论文叙事应坚持「协议 + 可信度」创新面。完成官方权重与 EchoNet 评测前，所有临床数字保持 <span class="todo">待补充</span>。</p>
</section>

<section id="limitations">
<h2>8. 局限（Limitations）</h2>
<ul>
  <li>无 EchoNet-Dynamic / 官方权重 → 无真实 MAE/AUC。</li>
  <li>Windows CPU + simple_cnn / scratch 路径仅为连通性验证。</li>
  <li>DEMO 校准分割不满足严格 VAL/TEST 分离的临床解释。</li>
  <li>本轮 ChatGPT 新对话因浏览器自动化故障未建成（surrogate 五轮 + 粘贴包替代）；用户可粘贴 CONTEXT 启用 web search 后回填 URL。</li>
</ul>
</section>

<section id="collab">
<h2>9. 五轮协作摘要（Five-round collaboration）</h2>
<p>详见 <code>reports/echoclip_tc_five_round_collab_20260816.md</code>。本轮全部为 <strong>surrogate</strong>（ChatGPT 浏览器 MCP 不可用），但每轮均：CONTEXT → 独立 WebSearch/源码核对 → 改稿 → 证据回写。</p>
<ol>
  <li>R1 文献架构与创新面（EchoCLIP / EchoPrime / CardiacCLIP DOIs 核对）</li>
  <li>R2 Methods vs <code>protocol.py</code> / <code>zeroshot.py</code> / <code>calibrate.py</code> / <code>temporal.py</code></li>
  <li>R3 Results/figures 诚实边界与 DEMO 可声称表</li>
  <li>R4 Discussion vs EchoCLIP / EchoPrime / CardiacCLIP</li>
  <li>R5 全文一致性 + 本报告加深图注</li>
</ol>
<p>既有 live ChatGPT（B0/M1）：<a href="https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6">6a80922d-d1d0-83ea-970c-67b829457cd6</a>。新文献 chat URL：<strong>Unavailable</strong>（blocker 同上）。</p>
</section>

<section id="refs">
<h2>10. 参考文献（References）</h2>
<ol>
  <li>Christensen M, et al. Vision–language foundation model for echocardiogram interpretation. <em>Nat Med</em> 2024. doi:10.1038/s41591-024-02959-y</li>
  <li>Vukadinovic M, et al. Comprehensive echocardiogram evaluation with view primed vision language AI. <em>Nature</em> 2026;650:970–977. doi:10.1038/s41586-025-09850-x ; arXiv:2410.09704</li>
  <li>Du Y, Guo J, Li X. CardiacCLIP: Video-based CLIP Adaptation for LVEF Prediction in a Few-shot Manner. MICCAI 2025. arXiv:2509.17065</li>
  <li>Ouyang D, et al. EchoNet-Dynamic. <em>Nature</em> 2020. doi:10.1038/s41586-020-2145-8</li>
  <li>Leclerc S, et al. CAMUS. <em>IEEE TMI</em> 2019.</li>
  <li>Radford A, et al. CLIP. ICML 2021.</li>
  <li>Guo C, et al. On calibration of modern neural networks. ICML 2017.</li>
  <li>Angelopoulos A, Bates S. Conformal prediction primer.</li>
</ol>
<p class="small">报告生成器：<code>scripts/build_research_report_bundle.py</code>；图片均为 Base64 内嵌，无外链 CDN。</p>
</section>
</div>
</body>
</html>
"""
    return body


def manuscript_html() -> str:
    md = (PAPERS / "echoclip_tc_manuscript.md").read_text(encoding="utf-8")
    # Lightweight markdown→HTML for headings/paragraphs (enough for self-contained paper HTML)
    lines = md.splitlines()
    out = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        "<title>EchoCLIP-TC Manuscript Draft</title>",
        f"<style>{CSS}</style>",
        "</head><body><div class='wrap'>",
        "<header class='cover'><h1>EchoCLIP-TC Manuscript (HTML)</h1>",
        "<p class='meta'>Self-contained HTML twin of papers/echoclip_tc_manuscript.md · DEMO figures Base64-embedded</p>",
        "<p class='demo'>Clinical EchoNet metrics: 待补充. Demo plots are not clinical EF MAE.</p></header>",
        "<nav class='toc'><h2>Figures</h2><ol>",
        "<li><a href='#fig1'>Fig.1 Architecture</a></li>",
        "<li><a href='#fig2'>Fig.2 Ablation</a></li>",
        "<li><a href='#fig3'>Fig.3 Calibration DEMO</a></li>",
        "<li><a href='#fig4'>Fig.4 Protocol DEMO</a></li>",
        "<li><a href='#fig5'>Fig.5 Roadmap</a></li>",
        "<li><a href='#fig6'>Fig.6 Conformal DEMO</a></li>",
        "</ol></nav>",
    ]
    para = []

    def flush():
        nonlocal para
        if para:
            text = " ".join(para)
            out.append(f"<p>{html.escape(text)}</p>")
            para = []

    for line in lines:
        if line.startswith("# "):
            flush()
            out.append(f"<h2>{html.escape(line[2:])}</h2>")
        elif line.startswith("## "):
            flush()
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush()
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|") and "---" not in line:
            flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if "Term" in cells[0] or cells[0] in ("ID", "Term", "Claim") else "td"
            # crude: first table header detection
            if cells[0] in ("Term", "ID", "Gap", "Component") or "Canonical" in "".join(cells):
                out.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        elif line.startswith("|---"):
            continue
        elif line.strip() == "":
            flush()
            if out and out[-1].startswith("<tr>"):
                out.append("</table>")
            continue
        elif line.startswith("- ") or line.startswith("1. "):
            flush()
            out.append(f"<li>{html.escape(line.lstrip('0123456789.- ').strip())}</li>")
        else:
            para.append(line)
    flush()

    out.append("<h2>Embedded figures</h2>")
    out.append(
        img_block(
            "fig1",
            "Figure 1. Protocol architecture",
            "Left-to-right pipeline and four protocol cards.",
            "Methods overview without numeric claims.",
            "Scaffold exists; no clinical accuracy claim.",
        )
    )
    out.append(
        img_block(
            "fig2",
            "Figure 2. Ablation schematic",
            "Compare B0 vs M1 vs M2 aggregation.",
            "Documents nonlinear B0≠M1 design.",
            "Ablation logic fixed; EchoNet numbers 待补充.",
        )
    )
    out.append(
        img_block(
            "fig3",
            "Figure 3. DEMO reliability",
            "Confidence vs accuracy bins vs diagonal.",
            "Toy curves for figure template.",
            "DEMO only.",
        )
    )
    out.append(
        img_block(
            "fig4",
            "Figure 4. DEMO metrics",
            "Bar charts of local demo MAE/ECE.",
            "From demo metrics.json.",
            "Not EchoNet; not 7.1%.",
        )
    )
    out.append(
        img_block(
            "fig5",
            "Figure 5. Roadmap",
            "Four stages with 待补充 markers.",
            "Project status communication.",
            "Gaps explicit.",
        )
    )
    out.append(
        img_block(
            "fig6",
            "Figure 6. DEMO conformal",
            "Scatter with ±quantile band.",
            "Illustrates interval reading.",
            "DEMO quantile only.",
        )
    )
    out.append("</div></body></html>")
    return "\n".join(out)


def research_report_md() -> str:
    return f"""# EchoCLIP-TC 学术研究报告

**日期：** 2026-08-16  
**项目：** E:\\\\Projects\\\\20260522-EchoCLIP  
**GitHub：** https://github.com/Coucou2016/EchoCLIP-TC  
**并行稿：** `reports/research_report.html`（单文件自包含） / `papers/echoclip_tc_manuscript.md`  
**五轮日志：** `reports/echoclip_tc_five_round_collab_20260816.md`

> **DEMO ≠ 临床。** 下表与 DEMO 图不得写作 EchoNet EF MAE。Christensen et al. 外部 EF MAE ≈7.1% 为文献目标，非本地结果。磁盘检索未发现 EchoNet-Dynamic。

## 目录

1. 摘要  
2. 背景  
3. 方法  
4. 过程  
5. 结果  
6. 讨论  
7. 结论  
8. 局限  
9. 五轮协作  
10. 参考文献  

## 1. 摘要

EchoCLIP-TC 在冻结 EchoCLIP 双塔上增加时序聚合与 VAL-only 校准/共形评测，并锁定 B0/M1/M2/M4 协议。本地门禁通过；**临床指标待补充**（缺 EchoNet-Dynamic 与官方权重）。

## 2. 背景

帧级 VLM（EchoCLIP）与视频/多切面模型（EchoPrime、CardiacCLIP）之间，缺少「冻结权重 + 公平视频向量消融 + 校准报告」的可复现公开数据协议。写作宜模仿 Nat Med 叙事 + MICCAI 消融表 + 校准图。

## 3. 方法

见 PAPER.md / manuscript §3。要点：B0=`uniform`/frames；M1=`mixed`/mean；M2=`mixed`/temporal；M4=VAL-only cal。B0≠M1（非线性排序）。

![Fig1](../figures/fig1_protocol_architecture.png)

**读图：** 左→右为数据流；下方为四模式。**结论：** 结构说明，无临床数值。

![Fig2](../figures/fig2_ablation_schematic.png)

**读图：** 三列消融逻辑。**结论：** B0≠M1 语义成立；数值待 EchoNet。

## 4. 过程

- SciencePlots 重绘图至 `figures/`  
- manuscript Methods/Results/Discussion 成熟化  
- ChatGPT 浏览器 MCP blocked → 5 轮 surrogate + 粘贴包  
- 既有 live ChatGPT（B0/M1）：https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6  

![Fig5](../figures/fig5_roadmap_bilingual.png)

## 5. 结果

### 5.1 临床

**待补充。**

### 5.2 DEMO 流水线（非临床；T=4）

| ID | DEMO MAE | DEMO ECE@50 | load_source | demo | n |
|----|----------|-------------|-------------|------|---|
| B0 | {PROTO['B0']['mae']} | {PROTO['B0']['ece_ef_lt_50']:.4f} | {PROTO['B0']['load_source']} | yes | {PROTO['B0']['n']} |
| M1 | {PROTO['M1']['mae']} | {PROTO['M1']['ece_ef_lt_50']:.4f} | {PROTO['M1']['load_source']} | yes | {PROTO['M1']['n']} |
| M2 | {PROTO['M2']['mae']} | {PROTO['M2']['ece_ef_lt_50']:.4f} | {PROTO['M2']['load_source']} | yes | {PROTO['M2']['n']} |
| M4 | {PROTO['M4']['mae']} | {PROTO['M4']['ece_ef_lt_50']:.4f} | {PROTO['M4']['load_source']} | yes | {PROTO['M4']['n']} |

![Fig4](../figures/fig4_demo_protocol_metrics.png)

![Fig3](../figures/fig3_calibration_reliability_demo.png)

![Fig6](../figures/fig6_conformal_demo.png)

## 6. 讨论

诚实创新面：时序模块、B0/M1 语义、校准协议、公开复现。不宣称私有大规模预训练或 DEMO 临床意义。EchoPrime / CardiacCLIP 仅作定位对照。

## 7. 结论

方法学脚手架就绪；临床表待官方资产。

## 8. 局限

无真实数据/权重；浏览器咨询受阻；DEMO 校准不可外推。

## 9. 五轮协作

见 `reports/echoclip_tc_five_round_collab_20260816.md`（5× surrogate；ChatGPT 新 URL unavailable）。

## 10. 参考文献

1. Christensen et al., Nat Med 2024, doi:10.1038/s41591-024-02959-y  
2. EchoPrime, Nature 2026;650:970–977, doi:10.1038/s41586-025-09850-x; arXiv:2410.09704  
3. CardiacCLIP, MICCAI 2025, arXiv:2509.17065  
4. Radford et al., CLIP, ICML 2021  
"""


def try_pdf(html_path: Path, pdf_path: Path) -> str:
    """Return status string."""
    # 1) playwright
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.as_uri(), wait_until="load")
            page.pdf(path=str(pdf_path), format="A4", print_background=True)
            browser.close()
        return f"OK via playwright → {pdf_path}"
    except Exception as e1:
        err1 = repr(e1)

    # 2) weasyprint
    try:
        from weasyprint import HTML

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return f"OK via weasyprint → {pdf_path}"
    except Exception as e2:
        err2 = repr(e2)

    # 3) reportlab minimal text fallback (no full CSS)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # Register a Windows Chinese-capable font if present
        font_name = "Helvetica"
        for fp, name in [
            (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),
            (r"C:\Windows\Fonts\msyh.ttc", "YaHei"),
            (r"C:\Windows\Fonts\times.ttf", "TimesNewRoman"),
        ]:
            if Path(fp).exists():
                try:
                    pdfmetrics.registerFont(TTFont(name, fp))
                    font_name = name
                    break
                except Exception:
                    continue

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4
        text = c.beginText(2 * cm, h - 2 * cm)
        text.setFont(font_name if font_name != "Helvetica" else "Helvetica", 10)
        md = (REPORTS / "research_report.md").read_text(encoding="utf-8")
        for line in md.splitlines():
            # simple wrap
            while len(line) > 95:
                text.textLine(line[:95])
                line = line[95:]
                if text.getY() < 2 * cm:
                    c.drawText(text)
                    c.showPage()
                    text = c.beginText(2 * cm, h - 2 * cm)
                    text.setFont(font_name if font_name != "Helvetica" else "Helvetica", 10)
            text.textLine(line)
            if text.getY() < 2 * cm:
                c.drawText(text)
                c.showPage()
                text = c.beginText(2 * cm, h - 2 * cm)
                text.setFont(font_name if font_name != "Helvetica" else "Helvetica", 10)
        c.drawText(text)
        # embed PNGs on following pages
        for name, pth in FIGS.items():
            c.showPage()
            c.setFont("Helvetica", 12)
            c.drawString(2 * cm, h - 2 * cm, name)
            c.drawImage(str(pth), 2 * cm, h / 2 - 4 * cm, width=16 * cm, preserveAspectRatio=True, mask="auto")
        c.save()
        return f"OK via reportlab text+figures fallback → {pdf_path} (playwright:{err1}; weasyprint:{err2})"
    except Exception as e3:
        return f"PDF FAILED: playwright={err1}; weasyprint={err2}; reportlab={e3!r}"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PAPERS.mkdir(parents=True, exist_ok=True)

    rr_html = REPORTS / "research_report.html"
    rr_md = REPORTS / "research_report.md"
    ms_html = PAPERS / "echoclip_tc_manuscript.html"
    rr_pdf = REPORTS / "research_report.pdf"

    rr_html.write_text(research_report_html(), encoding="utf-8")
    rr_md.write_text(research_report_md(), encoding="utf-8")
    ms_html.write_text(manuscript_html(), encoding="utf-8")

    status = try_pdf(rr_html, rr_pdf)
    (REPORTS / "_pdf_build_status.txt").write_text(status, encoding="utf-8")
    print("Wrote", rr_html, "size", rr_html.stat().st_size)
    print("Wrote", rr_md)
    print("Wrote", ms_html, "size", ms_html.stat().st_size)
    print("PDF:", status)


if __name__ == "__main__":
    main()
