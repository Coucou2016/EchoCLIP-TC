# EchoCLIP-TC dual-agent continue — 2026-08-16

## ChatGPT Pro/Plus 协作记录

| # | 问题 | 链接 |
|---|------|------|
| 1 | B0 vs M1 时序聚合公平性、协议对比表、M4 静默失败风险 | https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6 |

- 对话数：1（文本粘贴，无附件上传）
- 过程中曾出现 “Too many requests”；稍后恢复并完成 CONTEXT 2 答复
- 浏览器标签曾短暂消失，已用保存的 URL 恢复

## 源码基线

| 项 | 值 |
|----|-----|
| Git | **无 `.git`**（非 git 仓库） |
| 分支 / commit | N/A |
| Working tree | 已有大量本地 EchoCLIP-TC 改动（先前会话） |
| 门禁 | `unittest` + `scripts/validate.py --skip-eval`（无 lint/typecheck/E2E CI） |

## 提供给 ChatGPT 的上下文

- 模块：`zeroshot.compute_regression_score`、`eval_clinical.predict_ef`、协议 B0/M1/M2/M4
- 代码：以文本粘贴关键路径与形状约定（无 ZIP / 无文件上传）
- 脱敏：无密钥 / 无患者数据
- 分阶段：CONTEXT 1/2 ACK → CONTEXT 2/2 + 正式问题

## ChatGPT 主要建议（摘要）

1. **B0 vs M1**：B0 在每帧上排序 prompt 再跨帧平均 EF 值；M1 先平均 embedding 再排序一次。因排序非线性，二者不等价。
2. **公平性**：若 M1 定位为「无参数视频向量消融」（对照 M2），则当前定义正确；勿写成“官方 EchoCLIP + mean pool”而不加限定。
3. **保留当前 M1**；标量级 per-frame EF 均值可作为可选 M1b，勿替换主 M1。
4. **需要 comparison 表**（mae/load_source/demo 守卫）——仓库已有 `write_protocol_comparison`。
5. **M4 风险**：校准必须真正作用到指标路径；校验 VAL/TEST 不相交、checkpoint hash、禁止静默 identity fallback。
6. **测试**：T=1 等价、相同帧等价、rank-crossing 反例、`(B,D)` 形状约定。

## 被否决或修正的问题

| ChatGPT 说法 | 本地证据 | 处理 |
|--------------|----------|------|
| `score(x_bd) == score(x_bd.unsqueeze(1))` | `_as_frame_batch` 将 2D 解释为 `(T,D)` 单视频，不是 `(B,D)`；实测不等价 | **否决字面测试**；改为文档化约定 + `test_bd_batch_requires_explicit_t_axis`。`zero_shot_ef_batch` 已对 `(B,D)` 做 `unsqueeze(1)`，评测路径正确 |

其余建议与本地源码核对后采纳。

## 实际本地修改（本轮）

| 文件 | 变化 |
|------|------|
| `PAPER.md` | 明确 B0 vs M1 非线性差异；指向 comparison 表 |
| `echoclip/zeroshot.py` | `_as_frame_batch` 文档：2D=`(T,D)`；`(B,D)` 须由调用方变为 `(B,1,D)` |
| `tests/test_zeroshot.py` | B0/M1 语义测试（T=1、同帧、rank-crossing、2D 约定） |
| （删除）`echoclip/protocol_table.py` | 与已有 `write_protocol_comparison` 重复，删除避免双实现 |

依赖 / Lockfile：**无变化**。

（先前会话已存在：`write_protocol_comparison`、`scripts/write_protocol_table.py`、M4 cal 硬失败等，本轮未重复发明。）

## 独立测试结果

| 命令 | 结果 |
|------|------|
| `python -m unittest discover -s tests -v` | **PASS**（先前整套 59；本轮 zeroshot+protocol 18 OK；新增语义用例通过） |
| `python scripts/validate.py --skip-eval` | **PASS**（含 smoke + manifest + unittest） |
| EchoNet 临床 EF MAE | **未运行**（无 EchoNet-Dynamic / 官方权重） |
| GPU `convnext` 训练 | **未运行** |

Demo / dry-run **≠** 临床验证。

## 尚未验证的风险

| 风险 | 状态 |
|------|------|
| 官方 hub EF MAE ≈ 7.1% | 需要真实环境 |
| M4 校准是否改变 TEST 指标（相对 M2） | 仅代码审查 + 既有 cal 测试；需 EchoNet |
| VAL/TEST ID 不相交 | 尚未做 ID 级校验 |
| Checkpoint SHA 绑定校准产物 | 尚未实现 |
| CAMUS / Pediatric / LVH | 构建器在；数据未下载 |

## Git / 发布状态

**仅本地修改，未提交、未推送、未创建 PR、未部署。**
（仓库甚至无 `.git`。）
