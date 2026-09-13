# EchoCLIP-TC 学术研究报告

**日期：** 2026-09-14（P1 续写）  
**项目：** 仓库根目录（请用环境变量 `ECHOCLIP_ROOT` / `ECHONET_ROOT`，勿硬编码本机盘符）  
**GitHub：** https://github.com/Coucou2016/EchoCLIP-TC  
**并行稿：** `reports/research_report.html`（单文件自包含） / `papers/echoclip_tc_manuscript.md`  
**协议文档：** `PAPER.md`（R0–R6 + Oracle-EDES）  
**审稿回应：** `reports/review_response_p0_20260913.md` · `reports/review_response_p1_complete_20260914.md`

> **DEMO ≠ 临床。** 下表与 DEMO 图不得写作 EchoNet EF MAE。Christensen et al. 外部 EF MAE ≈7.1% 为文献目标，非本地结果。缺 EchoNet-Dynamic 时临床指标一律 **待补充**。

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

EchoCLIP-TC / EchoCLIP-TA 在冻结 EchoCLIP 双塔上增加**参数高效、EF 感知**的时序适配（默认 `EFSoftContrastiveLoss` + EF-only captions），并锁定 **R0–R6 + Oracle-EDES** 公平评测协议（VAL/TEST 强制 uniform；ed_es/mixed 硬失败，Oracle 除外）。校准报告温度/仿射逻辑 @50/40/30、split conformal（固定宽度局限已文档化）、可选自适应 conformal + AURC，以及 bootstrap / 配对 ΔMAE CI。本地门禁通过；**临床指标待补充**（缺 EchoNet-Dynamic 与官方权重）。**不宣称**“首个时序 EchoCLIP”。

## 2. 背景

帧级 VLM（EchoCLIP）与视频/多切面模型（EchoPrime、CardiacCLIP）之间，缺少「冻结权重 + 公平视频向量消融 + 校准报告」的可复现公开数据协议。写作宜模仿 Nat Med 叙事 + MICCAI 消融表 + 校准图。新颖性定位：**parameter-efficient / EF-aware**，而非 foundation 优先权。

## 3. 方法

见 PAPER.md / manuscript §3。要点：

| ID | 别名 | 评测采样 | 说明 |
|----|------|----------|------|
| R0 | B0 | uniform / official_stride(`--paper`) | EchoCLIP-based zero-shot |
| R1 | M1 | **uniform** | mean pool |
| R2–R4 | S0–S2 | **uniform** | 监督基线 |
| R5 | M2 | **uniform** | EF 监督时序适配（非 zero-shot 时序扩展） |
| R6 | M4 | **uniform** | R5 + VAL-only 校准 |
| Oracle-EDES | — | ed_es | 标注辅助上界 |

R0≠R1（非线性排序）。`--paper` 在 train/eval/protocol 缺官方权重时硬失败。

![Fig1](../figures/fig1_protocol_architecture.png)

**读图：** 左→右为数据流；下方为协议模式。**结论：** 结构说明，无临床数值。

![Fig2](../figures/fig2_ablation_schematic.png)

**读图：** 消融逻辑。**结论：** R0≠R1 语义成立；数值待 EchoNet。

## 4. 过程

- P0：uniform VAL/TEST、Oracle、`--paper`、R0–R6、EF soft loss、仿射校准钩子  
- P1：R5 默认 EF soft + EF-only captions；@40/@30 报告；自适应 conformal；配对 bootstrap；`run_seeds.py`；ATTRIBUTION 文件级清单；`requirements-lock.txt`；去硬编码路径  
- CardiacCLIP：`echoclip/cardiacclip_stub.py`（需外部权重，不编造数字）  
- 注意力/ED-ES：`scripts/analyze_attention_edes.py`（toy tensors）

## 5. 结果

### 5.1 临床

**待补充。**（EchoNet-Dynamic + 官方 hub 权重缺失时不得填写 MAE/AUC。）

### 5.2 DEMO 流水线（非临床；T=4；遗留别名 B0/M1/M2/M4 ≡ R0/R1/R5/R6）

| ID | DEMO MAE | DEMO ECE@50 | load_source | demo | n |
|----|----------|-------------|-------------|------|---|
| R0/B0 | 11.25 | 0.6496 | scratch_fallback | yes | 32 |
| R1/M1 | 11.25 | 0.6496 | scratch_fallback | yes | 32 |
| R5/M2 | 8.125 | 0.3371 | scratch | yes | 32 |
| R6/M4 | 8.125 | 0.0000 | scratch | yes | 32 |

![Fig4](../figures/fig4_demo_protocol_metrics.png)

![Fig3](../figures/fig3_calibration_reliability_demo.png)

![Fig6](../figures/fig6_conformal_demo.png)

## 6. 讨论

诚实创新面：参数高效时序模块、R0/R1 语义与采样公平性、校准/共形协议、公开复现。不宣称私有大规模预训练、DEMO 临床意义，或“首个时序 EchoCLIP”。EchoPrime / CardiacCLIP 仅作定位对照（CardiacCLIP 数字待外部权重）。

## 7. 结论

方法学脚手架与 P0/P1 审稿项已落地；临床表待官方资产（`--paper` 路径硬失败直至权重可用）。

## 8. 局限

缺 EchoNet / hub；basic split conformal 固定宽度；golden bit-exact EchoCLIP 仍有 tokenizer/crop/dtype 差距（已文档化）。

## 9. 五轮协作

见 `reports/echoclip_tc_five_round_collab_20260816.md`。

## 10. 参考文献

见 manuscript References；环境锁定见 `requirements-lock.txt`；许可边界见 `ATTRIBUTION.md`（勿扩大 MIT 主张）。
