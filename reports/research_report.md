# EchoCLIP-TC 学术研究报告（Markdown 孪生稿）

**日期：** 2026-08-16  
**项目：** `E:\Projects\20260522-EchoCLIP`  
**并行交付：** `reports/research_report.html`（单文件自包含）· `reports/research_report.pdf` · `papers/echoclip_tc_manuscript.md`  
**Git：** 无 `.git`（非 git 仓库）

> **诚实声明：** DEMO / synthetic 指标与示意图仅验证流水线（pipeline），**不得**写作 EchoNet 临床 EF MAE / AUC。Christensen et al. 报道的外部 EF MAE ≈ 7.1% 是文献复现目标，**不是**本仓库本地结果。临床主表一律标 **待补充**。

---

## 目录（Table of Contents）

1. [封面信息](#封面信息)  
2. [摘要 Abstract](#1-摘要-abstract)  
3. [研究背景 Background](#2-研究背景-background)  
4. [方法 Methods](#3-方法-methods)  
5. [实施过程 Process](#4-实施过程-process)  
6. [结果 Results](#5-结果-results)  
7. [讨论 Discussion](#6-讨论-discussion)  
8. [结论 Conclusions](#7-结论-conclusions)  
9. [局限 Limitations](#8-局限-limitations)  
10. [参考文献 References](#9-参考文献-references)  
11. [图注详解（逐子图）](#11-图注详解逐子图)

---

## 封面信息

| 项 | 内容 |
|----|------|
| 题名 | EchoCLIP-TC：面向超声心动图视觉–语言模型的时序聚合与校准评测 |
| 英文 | EchoCLIP-TC: Temporal aggregation and calibrated zero-shot evaluation for echocardiogram VLMs |
| 类型 | Methods / Nature-family 风格方法学跟进稿（非宣称新百万预训练基础模型） |
| 本地资产 | 协议 B0/M1/M2/M4、校准模块、~63 tests、SciencePlots 图 |
| 缺失资产 | EchoNet-Dynamic 视频；官方 HF EchoCLIP 权重成功加载 |

---

## 1. 摘要（Abstract）

EchoCLIP（Christensen 等，*Nature Medicine* 2024）在超过一百万心超视频–报告对上学习图像–文本对齐，支持零样本（zero-shot）左室射血分数（left ventricular ejection fraction, EF）估计。本报告描述本仓库实现的 **EchoCLIP-TC（Temporal, Calibrated）**：在**冻结**双塔编码器之上增加周期感知帧采样与时序聚合器（temporal aggregator / Temporal Transformer），并以验证集（validation, VAL）拟合温度缩放（temperature scaling）与分割共形预测（split-conformal）区间，锁定实验 ID **B0 / M1 / M2 / M4**。

**本地结论边界：** 单元测试与 `validate.py --skip-eval` 门禁通过；因缺少 EchoNet-Dynamic 与官方权重，**临床 EF MAE / AUC / 共形覆盖率待补充**。现有 `checkpoints/protocol/*/metrics.json` 均为 DEMO 冒烟。

---

## 2. 研究背景（Background）

### 2.1 临床与技术动机

超声心动图时间分辨率高，心功能评估依赖心动周期动态。视觉–语言模型（vision–language model, VLM）可通过对比学习降低逐任务标注成本。

### 2.2 相关工作（经独立 WebSearch/DOI 核对；ChatGPT 本轮未建成）

| 工作 | 出处 | 与本项目关系 |
|------|------|----------------|
| EchoCLIP | *Nat Med* 2024, doi:10.1038/s41591-024-02959-y | 基座与 B0 复现目标（外部 EF MAE ≈7.1%、内部 ≈8.4% 为**文献值**，非本仓库结果） |
| EchoPrime | *Nature* 2026;650:970–977, doi:10.1038/s41586-025-09850-x；预印本 arXiv:2410.09704 | 多视频/多切面大规模对照（>12M）；本项目不宣称同级数据规模 |
| CardiacCLIP | MICCAI 2025；arXiv:2509.17065；papers.miccai.org/0034 | 视频 CLIP 适配 + MFL/EchoZoom；方法近邻，评测合同不同 |
| EchoNet-Dynamic | *Nature* 2020, doi:10.1038/s41586-020-2145-8 | 主公开 EF 视频基准与协议数据合同 |
| CAMUS | *IEEE TMI* 2019, doi:10.1109/TMI.2019.2900516 | 可选外推/分割–EF 公开集；本地 **待补充** |
| Temperature scaling | Guo et al., ICML 2017 | M4 校准方法学锚点 |
| Conformal prediction | Angelopoulos & Bates (arXiv:2107.07511) | M4 区间/覆盖率报告锚点 |

### 2.3 可模仿写作架构与创新面

宜采用：**Nature Medicine 式问题→模型→零样本任务→外部验证**（EchoCLIP） + **MICCAI 式基线–消融表**（CardiacCLIP） + **TMI 式数据集/指标合同**（CAMUS） + **校准可靠性图**；用 EchoPrime 作多切面上限对照而非模仿其训练规模叙事。  
在无 1M 私有数据时，诚实创新点应落在：

1. 与冻结 EchoCLIP 兼容的视频级 \(z_v\) 时序模块；  
2. B0 vs M1 语义澄清（排序非线性 → 一般不等价）；  
3. VAL-only 温度 / ECE / Brier / 共形 / 弃权作为一等公民指标；  
4. 公开数据可复现协议（PAPER.md 锁定）。

**图清单（稿件）：** Fig.1 协议架构；Fig.2 B0/M1/M2 消融；Fig.3 校准可靠性（DEMO）；Fig.4 协议冒烟指标（DEMO ONLY）；Fig.5 研究路线图（待补充）；Fig.6 共形区间卡通（DEMO）。

---

## 3. 方法（Methods）

### 3.1 协议矩阵

| ID | 训练 | Pool | 校准 | 角色 |
|----|------|------|------|------|
| B0 | 否 | frames（逐帧 EF 再聚合） | 否 | 官方风格基线 |
| M1 | 否 | mean（先 \(z_v\)） | 否 | 无参视频向量消融 |
| M2 | 是（仅时序模块） | temporal | 否 | 主 TC 模型 |
| M4 | 复用 M2 | temporal | 是（仅 VAL） | 校准 TC |

### 3.2 关键模块

- `echoclip/temporal.py` — 注意力池化 / Temporal Transformer  
- `echoclip/cycle_sample.py` — random / uniform / ED-ES / mixed  
- `echoclip/calibrate.py` — 温度、ECE、Brier、共形、弃权  
- `scripts/run_protocol.py` / `eval_clinical.py` — 协议与主指标  

### 3.3 主指标

EF MAE / RMSE / \(R^2\)；AUC@EF&lt;50/40/30；ECE；Brier；共形覆盖率与宽度；弃权后 MAE。检索 R@k 仅诊断用。

---

## 4. 实施过程（Process）

1. 读取 README、PAPER.md、既有 `reports/echoclip_tc_continue_20260816.md`。  
2. 确认 **SciencePlots 2.2.2** 已安装；`plt.style.use(['science','no-latex'])`；英文 Times New Roman，中文 Microsoft YaHei / SimSun。  
3. 重绘全部图至 `figures/`（并复制到 `reports/figures/`）。  
4. 使用已安装的 **nature-skills / nature-writing**（`task=manuscript`, `paper_type=methods`, `journal=nature-family`）起草英文稿。  
5. ChatGPT 新对话文献架构咨询：**Cursor 浏览器 MCP 故障**（先报 *No browser tab available*，后报 *MCP server does not exist: cursor-ide-browser*），未能新建会话；文献改由 WebSearch/WebFetch 独立核对。B0/M1 语义沿用既有对话：  
   https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6  
6. 门禁：`python -m unittest discover -s tests -v` → **63 OK**；`python scripts/validate.py --skip-eval` → **All validation steps passed**。  
7. 生成自包含 HTML（Base64 图）与 PDF（playwright）。

---

## 5. 结果（Results）

### 5.1 临床主结果

**待补充。** 需要：

- EchoNet-Dynamic（`FileList.csv` + `Videos/`）  
- `load_source` 记录为 `hf-hub:mkaichristensen/echo-clip`（非 `scratch_fallback` / `simple_cnn`）  
- seed=42 subset_5000 与/或 full TEST  

### 5.2 DEMO 流水线指标（非临床）

来源：`checkpoints/protocol/*/metrics.json`（`demo_mode=true`）。

| ID | DEMO MAE | DEMO ECE@EF&lt;50 | load_source | n | 备注 |
|----|----------|------------------|-------------|---|------|
| B0 | 11.25 | 0.6496 | scratch_fallback | 32 | 非临床 |
| M1 | 11.25 | 0.6496 | scratch_fallback | 32 | 非临床 |
| M2 | 8.125 | 0.3371 | scratch | 32 | 非临床 |
| M4 | 8.125 | ≈0（温度拟合后 DEMO） | scratch | 32 | 含 DEMO 共形字段 |

**禁止**将上表写入论文主结果；亦不可把 DEMO M2「低于 B0」解读为真实改进。

---

## 6. 讨论（Discussion）

本工作把「创新」定义为**可复现的时序–校准协议**，而非虚构数据规模。与 EchoPrime / CardiacCLIP 的差异应在讨论中诚实写出：数据规模、多切面能力、是否冻结官方 EchoCLIP、是否报告校准。

ChatGPT 外部顾问本轮（2026-08-16 重试）仍未能实时检索（`cursor-ide-browser` MCP 未挂载；详见 `reports/echoclip_tc_literature_chatgpt_20260816.md`）。独立 WebSearch 已核实 EchoCLIP / EchoPrime / CardiacCLIP / EchoNet-Dynamic / CAMUS / Guo / Angelopoulos 等锚点，并写入稿件 Related Work。先前 ChatGPT 关于 B0≠M1 的结论经本地测试采纳；其对 `(B,D)` 形状字面等价测试的建议被本地否决（见既有 continue 报告）。

---

## 7. 结论（Conclusions）

EchoCLIP-TC 方法学脚手架、协议、测试与论文/报告草稿已齐备。获得 AIMI 数据与官方权重后，按 PAPER.md 跑完 B0→M4 即可填充临床表。在此之前保持 **待补充**，不以 DEMO 充数。

---

## 8. 局限（Limitations）

1. 无 EchoNet-Dynamic / 官方权重 → 无真实临床数字。  
2. Windows CPU + `simple_cnn` 仅为连通性。  
3. DEMO 校准分割不支持临床共形解释。  
4. CAMUS / Pediatric / LVH 未落地。  
5. 本轮 ChatGPT 新会话因浏览器自动化不可用而未建立。  

---

## 9. 参考文献（References）

1. Christensen M, Vukadinovic M, Yuan N, Ouyang D. Vision–language foundation model for echocardiogram interpretation. *Nat Med*. 2024;30:1481–1488. https://doi.org/10.1038/s41591-024-02959-y  
2. Vukadinovic M, Chiu IM, Tang X, et al. Comprehensive echocardiogram evaluation with view primed vision language AI (EchoPrime). *Nature*. 2026;650:970–977. https://doi.org/10.1038/s41586-025-09850-x ; arXiv:2410.09704  
3. Du Y, Guo J, Li X. CardiacCLIP: Video-based CLIP Adaptation for LVEF Prediction in a Few-shot Manner. MICCAI 2025. arXiv:2509.17065 ; https://papers.miccai.org/miccai-2025/paper/0034_paper.pdf  
4. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function (EchoNet-Dynamic). *Nature*. 2020;580:252–256. https://doi.org/10.1038/s41586-020-2145-8  
5. Leclerc S, Smistad E, Pedrosa J, et al. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography (CAMUS). *IEEE Trans Med Imaging*. 2019;38(9):2198–2210. https://doi.org/10.1109/TMI.2019.2900516  
6. Radford A, et al. Learning transferable visual models from natural language supervision. ICML 2021.  
7. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.  
8. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.

---

## 11. 图注详解（逐子图）

图片文件位于 `figures/`（SciencePlots 导出 PNG/PDF）。HTML/PDF 中为 Base64 内嵌。

### Figure 1 — `fig1_protocol_architecture.png`

- **如何读：** 左→右：视频帧 → 冻结编码器 → 时序聚合 → \(z_v\)；下方四框为 B0/M1/M2/M4。  
- **含义：** 方法总览示意图。  
- **结论：** 协议结构已实现；**无**性能结论。

### Figure 2 — `fig2_ablation_schematic.png`

- **如何读：** 三列对比 B0 / M1 / M2 的信息汇聚。  
- **含义：** 说明为何 M1 是无参 \(z_v\) 消融而非「官方路径的同义反复」。  
- **结论：** 消融逻辑可测；EchoNet 数值 **待补充**。

### Figure 3 — `fig3_calibration_reliability_demo.png`

- **如何读：** 横轴置信度、纵轴准确率；虚线理想校准。左欠校准玩具曲线，右温度后玩具改善。  
- **含义：** 校准图模板。  
- **结论：** **DEMO / synthetic only**，非 EchoNet ECE。

### Figure 4 — `fig4_demo_protocol_metrics.png`

- **如何读：** 左 DEMO MAE，右 DEMO ECE。  
- **含义：** 本地 `metrics.json` 冒烟可视化。  
- **结论：** 流水线可写出指标；**禁止**当临床主表。

### Figure 5 — `fig5_roadmap_bilingual.png`

- **如何读：** 四阶段路线图，含「待补充」标注。  
- **含义：** 项目完成度沟通。  
- **结论：** 脚手架就绪、临床资产缺口明确。

### Figure 6 — `fig6_conformal_demo.png`

- **如何读：** 合成散点 + ±DEMO quantile 阴影带。  
- **含义：** 共形区间读法示意。  
- **结论：** DEMO 带宽；不可外推临床覆盖率。
