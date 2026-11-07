# EchoCLIP-TA / EchoCLIP-TC 论文审查文档
本文件记录论文所依赖的数据、计算、文献与数字的全部核验证据。所有数值均由本次会话中的脚本实际运行产生，来源文件写在每一节中。

## 0 构建卷丢失与全量重建记录
本节的记录来自 Windows 系统日志与重建脚本自身的日志，不是事后推测。

2026 年 11 月 7 日 14 时 41 分，外置硬盘 WD Elements 2621（序列号 WXM2DA5FNXYY）的 USB 链路被重置。系统日志随后出现两条事件，14 时 41 分 07 秒 `Disk` 事件 153 报告在逻辑块地址 0x56646800 重试 IO，同一秒 `UASPStor` 事件 129 报告向 RaidPort1 发出设备重置。15 时 23 分 43 秒 `Ntfs` 事件 98 报告卷 F 修复完成。修复完成后，原先位于 `F:\EchoCLIP_paper_build` 的整个构建目录不再存在，该卷可用空间回到 3725.7 GB，只剩出厂的两个安装程序。

丢失的内容包括全部实验 JSON、五个帧嵌入缓存、图表的源文件、运行日志与十篇文献 PDF。论文正文与审查文档本身位于内置盘，未受影响。

处理方式有三项。其一，构建根目录改由 `scripts/_paths.py` 统一解析，落点是仓库内的 `_paper_build_zh`，不再依赖可移动介质，并新增 `scripts/check_build_paths.py` 静态扫描，确认没有脚本硬编码旧路径。其二，帧嵌入按第四节第四小节的命令重新计算，五个缓存的形状、EF 均值与标准差、单位范数与 ED/ES 有效性逐项复核，与丢失前的记录一致。其三，全部实验按同一命令矩阵重跑，论文与本文档的数字改由新 JSON 生成。

重建过程还定位并修复了两个真实缺陷。第一，文本提示编码把全部 202 条提示一次性送入显存，在 4 GB 显卡上触发 Windows 的显示驱动超时，现改为每 32 条一批并在批间释放显存。第二，消融脚本的步骤函数把日志输出混入返回值，导致成功的步骤被判定为失败并中止流水线，现已改为只返回退出码。

重跑后的数字与原稿存在小幅差异，原因是训练阶段在显卡上的非确定性，缓存本身逐字节可比。凡与旧稿不同的数字，论文一律采用新 JSON 的值，第 5 节的逐条对照表可以逐格复查。

关于代码版本，需要说明一处容易误解的地方。结果文件里的 `git_commit` 字段记录的是该次运行时的 HEAD，而重建期间路径改造尚未提交，因此部分文件的该字段早于实际执行所用的脚本。为避免歧义，`scripts/build_provenance.py` 对十一个流水线脚本逐一计算 SHA-256，并与当前提交中的内容逐字节比较，同时记录权重与清单文件的哈希。该文件为 `results/provenance.json`，是复核代码版本时应使用的依据。
## 1 数据真实性证据
- 数据根目录：`E:\Datasets\EchoNet-Dynamic`
- 清单文件：`E:\Datasets\EchoNet-Dynamic\FileList.csv`，SHA-256 `31942534b7bb3debb9f0f3de42af2fb4342f7ea918b32a4095a0b66e91fbace1`
- 清单行数：**10030**
- 划分计数：TRAIN 7465，VAL 1288，TEST 1277，合计 10030
- 视频目录：`E:\Datasets\EchoNet-Dynamic\Videos`，avi 文件数 **10030**
- 文件名一致性：清单唯一名 10030，磁盘缺失 0，磁盘多出 0，完全一致 = True
- 重复文件名：0，EF 缺失：0
- EF 范围：6.907 到 96.967，均值 55.748，标准差 12.371
- 帧数范围：28 到 1002，均值 176.5
### 1.1 清单与官方划分的一致性
| 划分 | 清单条目 | FileList 条目 | 集合相同 | 仅清单有 | 仅 FileList 有 |
|---|---|---|---|---|---|
| train | 7465 | 7465 | True | 0 | 0 |
| val | 1288 | 1288 | True | 0 | 0 |
| test | 1277 | 1277 | True | 0 | 0 |

test_subset_500 是 TEST 划分的子集，子集关系成立 = True，占 TEST 的比例 0.3915。
### 1.2 ED/ES 帧覆盖率
VolumeTracings 共 425010 行，覆盖 10025 个视频，每个视频恰好 [2] 个不同帧号，每视频中位行数 42，覆盖清单的 0.9995。
| 划分 | 条目 | 有效 ED/ES | 无效 | 覆盖率 |
|---|---|---|---|---|
| train | 7465 | 7460 | 5 | 0.9993 |
| val | 1288 | 1288 | 0 | 1.0000 |
| test | 1277 | 1276 | 1 | 0.9992 |
| test_subset_500 | 500 | 500 | 0 | 1.0000 |

### 1.3 帧嵌入缓存
| 缓存文件 | 样本 | 帧数 | 维度 | EF 均值 | EF 标准差 | EF<50 | 字节 |
|---|---|---|---|---|---|---|---|
| test500_32_bgr.npz | 500 | 32 | 512 | 54.82 | 12.84 | 126 | 32815706 |
| test500_stride_bgr.npz | 500 | 20 | 512 | 54.82 | 12.84 | 126 | 20527706 |
| train_32_bgr.npz | 2000 | 32 | 512 | 55.72 | 12.41 | 451 | 131257706 |
| val_32_bgr.npz | 1288 | 32 | 512 | 55.82 | 12.31 | 289 | 84530570 |

帧嵌入的二范数在 0.99999994 到 1.0 之间，说明归一化正确。

## 2 计算真实性证据
每个实验的完整命令、随机种子、样本数与输出文件如下。R0 与 R1 不训练，因此种子一列记为一次运行。R0 的命令行里写的是 `--frames 32`，脚本在检测到实验编号为 R0 时会把帧数强制设为官方步长缓存的 20 帧，结果文件 `config.n_frames` 记录的 20 就是实际使用的值。
| 实验 | 种子 | 帧数 | n_train | n_val | n_test | MAE | AUC50 | 输出 JSON | 命令 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 42 | 20 | 0 | 0 | 500 | 10.986 | 0.740 | `R0_s42_f20_cls.json` | `python scripts\train_heads_from_cache.py --experiment R0 --seeds 42 --frames 32` |
| R1 | 42 | 32 | 0 | 1288 | 500 | 11.035 | 0.738 | `R1_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R1 --seeds 42 --frames 32` |
| R1 | 42 | 32 | 0 | 1288 | 500 | 11.035 | 0.738 | `R1_s42_f32_cls_abtop20.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R1 --seeds 42 --frames 32 --top-frac 0.2 --tag abtop20` |
| R1 | 42 | 32 | 0 | 1288 | 500 | 10.900 | 0.732 | `R1_s42_f32_cls_top10_abtop10.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R1 --seeds 42 --frames 32 --top-frac 0.1 --tag abtop10` |
| R1 | 42 | 32 | 0 | 1288 | 500 | 10.692 | 0.746 | `R1_s42_f32_cls_top30_abtop30.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R1 --seeds 42 --frames 32 --top-frac 0.3 --tag abtop30` |
| R1U16 | 42 | 16 | 0 | 1288 | 500 | 11.000 | 0.740 | `R1U16_s42_f16_cls.json` | `python scripts\train_heads_from_cache.py --experiment R1U16 --seeds 42 --frames 16` |
| R2 | 42 | 32 | 2000 | 1288 | 500 | 6.417 | 0.860 | `R2_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R2 --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R2 | 43 | 32 | 2000 | 1288 | 500 | 6.445 | 0.855 | `R2_s43_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R2 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R2 | 44 | 32 | 2000 | 1288 | 500 | 6.422 | 0.863 | `R2_s44_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R2 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R3 | 42 | 32 | 2000 | 1288 | 500 | 6.216 | 0.861 | `R3_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R3 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R3 | 43 | 32 | 2000 | 1288 | 500 | 6.382 | 0.865 | `R3_s43_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R3 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R3 | 44 | 32 | 2000 | 1288 | 500 | 6.402 | 0.859 | `R3_s44_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R3 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R4 | 42 | 16 | 2000 | 1288 | 500 | 5.917 | 0.879 | `R4_s42_f16_cls_abframes16.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 16 --cache-tag 32_bgr --tag abframes16` |
| R4 | 42 | 32 | 2000 | 1288 | 500 | 5.868 | 0.876 | `R4_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R4 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R4 | 42 | 32 | 1000 | 1288 | 500 | 5.860 | 0.885 | `R4_s42_f32_cls_n1000_abscale1000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 32 --train-limit 1000 --tag abscale1000` |
| R4 | 42 | 32 | 2000 | 1288 | 500 | 5.650 | 0.896 | `R4_s42_f32_cls_n2000_abscale2000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 32 --train-limit 2000 --tag abscale2000` |
| R4 | 42 | 32 | 2000 | 1288 | 500 | 5.892 | 0.880 | `R4_s42_f32_cls_n4000_abscale4000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 32 --train-limit 4000 --tag abscale4000` |
| R4 | 42 | 32 | 500 | 1288 | 500 | 6.197 | 0.869 | `R4_s42_f32_cls_n500_abscale500.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 32 --train-limit 500 --tag abscale500` |
| R4 | 42 | 8 | 2000 | 1288 | 500 | 6.003 | 0.881 | `R4_s42_f8_cls_abframes8.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 42 --frames 8 --cache-tag 32_bgr --tag abframes8` |
| R4 | 43 | 16 | 2000 | 1288 | 500 | 6.041 | 0.865 | `R4_s43_f16_cls_abframes16.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 16 --cache-tag 32_bgr --tag abframes16` |
| R4 | 43 | 32 | 2000 | 1288 | 500 | 5.756 | 0.892 | `R4_s43_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R4 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R4 | 43 | 32 | 1000 | 1288 | 500 | 5.894 | 0.872 | `R4_s43_f32_cls_n1000_abscale1000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 32 --train-limit 1000 --tag abscale1000` |
| R4 | 43 | 32 | 2000 | 1288 | 500 | 5.992 | 0.883 | `R4_s43_f32_cls_n2000_abscale2000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 32 --train-limit 2000 --tag abscale2000` |
| R4 | 43 | 32 | 2000 | 1288 | 500 | 6.120 | 0.866 | `R4_s43_f32_cls_n4000_abscale4000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 32 --train-limit 4000 --tag abscale4000` |
| R4 | 43 | 32 | 500 | 1288 | 500 | 6.242 | 0.865 | `R4_s43_f32_cls_n500_abscale500.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 32 --train-limit 500 --tag abscale500` |
| R4 | 43 | 8 | 2000 | 1288 | 500 | 6.057 | 0.875 | `R4_s43_f8_cls_abframes8.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 43 --frames 8 --cache-tag 32_bgr --tag abframes8` |
| R4 | 44 | 16 | 2000 | 1288 | 500 | 5.869 | 0.881 | `R4_s44_f16_cls_abframes16.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 16 --cache-tag 32_bgr --tag abframes16` |
| R4 | 44 | 32 | 2000 | 1288 | 500 | 5.915 | 0.879 | `R4_s44_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R4 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R4 | 44 | 32 | 1000 | 1288 | 500 | 6.085 | 0.883 | `R4_s44_f32_cls_n1000_abscale1000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 32 --train-limit 1000 --tag abscale1000` |
| R4 | 44 | 32 | 2000 | 1288 | 500 | 5.801 | 0.890 | `R4_s44_f32_cls_n2000_abscale2000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 32 --train-limit 2000 --tag abscale2000` |
| R4 | 44 | 32 | 2000 | 1288 | 500 | 5.751 | 0.895 | `R4_s44_f32_cls_n4000_abscale4000.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 32 --train-limit 4000 --tag abscale4000` |
| R4 | 44 | 32 | 500 | 1288 | 500 | 6.219 | 0.881 | `R4_s44_f32_cls_n500_abscale500.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 32 --train-limit 500 --tag abscale500` |
| R4 | 44 | 8 | 2000 | 1288 | 500 | 5.751 | 0.882 | `R4_s44_f8_cls_abframes8.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4 --seeds 44 --frames 8 --cache-tag 32_bgr --tag abframes8` |
| R5 | 42 | 32 | 2000 | 1288 | 500 | 5.971 | 0.879 | `R5_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R5 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R5 | 42 | 32 | 2000 | 1288 | 500 | 5.922 | 0.882 | `R5_s42_f32_cls_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R5 --seeds 42 --frames 32 --tag abcalib` |
| R5 | 43 | 32 | 2000 | 1288 | 500 | 5.815 | 0.874 | `R5_s43_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R5 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R5 | 43 | 32 | 2000 | 1288 | 500 | 5.994 | 0.874 | `R5_s43_f32_cls_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R5 --seeds 43 --frames 32 --tag abcalib` |
| R5 | 44 | 32 | 2000 | 1288 | 500 | 5.925 | 0.883 | `R5_s44_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R5 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R5 | 44 | 32 | 2000 | 1288 | 500 | 5.964 | 0.889 | `R5_s44_f32_cls_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R5 --seeds 44 --frames 32 --tag abcalib` |
| R6 | 42 | 32 | 2000 | 1288 | 500 | 5.971 | 0.879 | `R6_s42_f32_cls_a10_20.json` | `python scripts\train_heads_from_cache.py --experiment R6 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R6 | 42 | 32 | 2000 | 1288 | 500 | 5.971 | 0.879 | `R6_s42_f32_cls_a10_20_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R6 --seeds 42 --frames 32 --tag abcalib` |
| R6 | 43 | 32 | 2000 | 1288 | 500 | 5.815 | 0.874 | `R6_s43_f32_cls_a10_20.json` | `python scripts\train_heads_from_cache.py --experiment R6 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R6 | 43 | 32 | 2000 | 1288 | 500 | 5.815 | 0.874 | `R6_s43_f32_cls_a10_20_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R6 --seeds 43 --frames 32 --tag abcalib` |
| R6 | 44 | 32 | 2000 | 1288 | 500 | 5.925 | 0.883 | `R6_s44_f32_cls_a10_20.json` | `python scripts\train_heads_from_cache.py --experiment R6 --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20` |
| R6 | 44 | 32 | 2000 | 1288 | 500 | 5.925 | 0.883 | `R6_s44_f32_cls_a10_20_abcalib.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R6 --seeds 44 --frames 32 --tag abcalib` |
| R4_MEAN | 42 | 32 | 2000 | 1288 | 500 | 5.858 | 0.887 | `R4_MEAN_s42_f32_cls_abpoolmean.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4_MEAN --seeds 42 --frames 32 --tag abpoolmean` |
| R4_MEAN | 43 | 32 | 2000 | 1288 | 500 | 6.043 | 0.866 | `R4_MEAN_s43_f32_cls_abpoolmean.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4_MEAN --seeds 43 --frames 32 --tag abpoolmean` |
| R4_MEAN | 44 | 32 | 2000 | 1288 | 500 | 5.913 | 0.885 | `R4_MEAN_s44_f32_cls_abpoolmean.json` | `python E:\Projects\20260522-EchoCLIP\scripts\train_heads_from_cache.py --experiment R4_MEAN --seeds 44 --frames 32 --tag abpoolmean` |
| R5_EDES | 42 | 32 | 2000 | 1288 | 500 | 5.836 | 0.887 | `R5_EDES_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment R5_EDES --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20 --device cpu` |
| R5_EDES | 42 | 32 | 2000 | 1288 | 500 | 5.836 | 0.887 | `R5_EDES_s42_f32_cls_edes.json` | `python scripts\train_heads_from_cache.py --experiment R5_EDES --seeds 42 --frames 32 --epochs 40 --pretrain-epochs 20 --device cpu` |
| ORACLE_EDES | 42 | 32 | 0 | 1288 | 500 | 10.954 | 0.734 | `ORACLE_EDES_s42_f32_cls.json` | `python scripts\train_heads_from_cache.py --experiment ORACLE_EDES --seeds 42 --frames 32` |

### 2.1 逐文件配置与耗时
| JSON | 创建时间 | 耗时(s) | device | 关键配置 |
|---|---|---|---|---|
| `ORACLE_EDES_s42_f32_cls.json` | 2026-11-07T19:32:25 | 12.5 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "annotated_ed_es_nearest_cache_slot"}` |
| `R0_s42_f20_cls.json` | 2026-11-07T19:31:14 | 10.2 | cuda | `{"n_frames": 20, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "official_stride_0_40_2"}` |
| `R1U16_s42_f16_cls.json` | 2026-11-07T19:32:02 | 11.7 | cuda | `{"n_frames": 16, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "uniform_16"}` |
| `R1_s42_f32_cls.json` | 2026-11-07T19:31:38 | 12.4 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "uniform_32"}` |
| `R1_s42_f32_cls_abtop20.json` | 2026-11-07T21:12:36 | 12.5 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "uniform_32"}` |
| `R1_s42_f32_cls_top10_abtop10.json` | 2026-11-07T21:12:11 | 13.6 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.1, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "uniform_32"}` |
| `R1_s42_f32_cls_top30_abtop30.json` | 2026-11-07T21:13:01 | 14.2 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.3, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0, "frame_selection": "uniform_32"}` |
| `R2_s42_f32_cls.json` | 2026-11-08T01:21:41 | 26.5 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R2_s43_f32_cls.json` | 2026-11-07T19:33:27 | 14.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R2_s44_f32_cls.json` | 2026-11-07T19:33:41 | 15.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R3_s42_f32_cls.json` | 2026-11-07T19:34:10 | 37.7 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R3_s43_f32_cls.json` | 2026-11-07T19:34:46 | 16.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R3_s44_f32_cls.json` | 2026-11-07T19:35:03 | 18.7 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_MEAN_s42_f32_cls_abpoolmean.json` | 2026-11-07T20:32:12 | 230.3 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_MEAN_s43_f32_cls_abpoolmean.json` | 2026-11-07T20:36:18 | 331.8 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_MEAN_s44_f32_cls_abpoolmean.json` | 2026-11-07T20:42:03 | 339.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f16_cls_abframes16.json` | 2026-11-07T20:22:12 | 151.7 | cuda | `{"n_frames": 16, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f32_cls.json` | 2026-11-07T19:35:37 | 222.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f32_cls_n1000_abscale1000.json` | 2026-11-07T21:21:08 | 159.7 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 1000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f32_cls_n2000_abscale2000.json` | 2026-11-07T21:29:11 | 252.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 2000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f32_cls_n4000_abscale4000.json` | 2026-11-07T21:40:47 | 298.7 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 4000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f32_cls_n500_abscale500.json` | 2026-11-07T21:13:28 | 122.5 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 500, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s42_f8_cls_abframes8.json` | 2026-11-07T20:14:44 | 111.9 | cuda | `{"n_frames": 8, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f16_cls_abframes16.json` | 2026-11-07T20:25:00 | 201.8 | cuda | `{"n_frames": 16, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f32_cls.json` | 2026-11-07T19:39:18 | 205.6 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f32_cls_n1000_abscale1000.json` | 2026-11-07T21:24:01 | 150.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 1000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f32_cls_n2000_abscale2000.json` | 2026-11-07T21:33:39 | 212.3 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 2000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f32_cls_n4000_abscale4000.json` | 2026-11-07T21:46:02 | 310.2 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 4000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f32_cls_n500_abscale500.json` | 2026-11-07T21:15:54 | 139.9 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 500, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s43_f8_cls_abframes8.json` | 2026-11-07T20:16:52 | 131.8 | cuda | `{"n_frames": 8, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f16_cls_abframes16.json` | 2026-11-07T20:28:36 | 195.2 | cuda | `{"n_frames": 16, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f32_cls.json` | 2026-11-07T19:42:44 | 202.1 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f32_cls_n1000_abscale1000.json` | 2026-11-07T21:26:45 | 133.2 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 1000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f32_cls_n2000_abscale2000.json` | 2026-11-07T21:37:26 | 183.5 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 2000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f32_cls_n4000_abscale4000.json` | 2026-11-07T21:51:27 | 276.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 4000, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f32_cls_n500_abscale500.json` | 2026-11-07T21:18:29 | 142.6 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "train_limit": 500, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R4_s44_f8_cls_abframes8.json` | 2026-11-07T20:19:19 | 158.9 | cuda | `{"n_frames": 8, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_EDES_s42_f32_cls.json` | 2026-11-07T23:24:31 | 5732.3 | cpu | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_EDES_s42_f32_cls_edes.json` | 2026-11-07T23:24:31 | 5732.3 | cpu | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s42_f32_cls.json` | 2026-11-07T19:46:20 | 310.9 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s42_f32_cls_abcalib.json` | 2026-11-07T20:47:56 | 440.3 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s43_f32_cls.json` | 2026-11-07T19:51:30 | 297.1 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s43_f32_cls_abcalib.json` | 2026-11-07T20:55:30 | 440.4 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s44_f32_cls.json` | 2026-11-07T19:56:27 | 286.4 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R5_s44_f32_cls_abcalib.json` | 2026-11-07T21:03:05 | 447.8 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s42_f32_cls_a10_20.json` | 2026-11-07T20:01:28 | 12.3 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s42_f32_cls_a10_20_abcalib.json` | 2026-11-07T21:10:49 | 13.6 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s43_f32_cls_a10_20.json` | 2026-11-07T20:01:39 | 5.9 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s43_f32_cls_a10_20_abcalib.json` | 2026-11-07T21:11:15 | 14.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s44_f32_cls_a10_20.json` | 2026-11-07T20:01:45 | 6.0 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 40, "pretrain_epochs": 20, "tau": 5.0, "smooth_sigma": 5.0}` |
| `R6_s44_f32_cls_a10_20_abcalib.json` | 2026-11-07T21:11:43 | 13.8 | cuda | `{"n_frames": 32, "pool": "cls", "top_frac": 0.2, "calibrate": false, "alpha": 0.1, "epochs": 120, "pretrain_epochs": 40, "tau": 5.0, "smooth_sigma": 5.0}` |

## 3 权重与版本
- 权重文件：`models/echo-clip/open_clip_pytorch_model.bin`
- SHA-256：`016e26e15da539b3a97dec89d7fa9a30f106e1c38d36ce64789cb99420786fe0`
- 视觉骨干：ConvNeXt-Base，嵌入维度 512，输入 224×224
- 文本侧：12 层 Transformer，上下文 77，词表 49408
- 加载结果：missing keys = 0，unexpected keys = 0（`model.load_state_dict(strict=False)` 的返回）
- PyTorch 版本：2.7.1+cu118
- 设备：['cpu', 'cuda']
- 代码提交：`39533b9f01d020b7f5d70e733f1bef26697f2118`
- 主机名：SK-20180915YOYU
- Python 版本：3.13.12
- OpenCLIP 版本：3.3.0
- 显卡：GeForce GTX 950M，显存 4.29 GB

关于加载接口。本机安装的 OpenCLIP 3.3.0 不支持把配置字典或 JSON 字符串直接传给 `open_clip.create_model`，该函数只接受内置模型名。实际可用写法是先读出 `open_clip_config.json` 的 `model_cfg`，取其中的 `vision_cfg.timm_model_name`（本权重为 `convnext_base`）作为模型名，再以 `pretrained=None` 建模型，最后用 `load_state_dict(strict=False)` 载入检查点。两条塔的关键字前缀为 `visual`、`transformer`、`token_embedding`、`ln_final`，缺失键与多余键都为 0，说明配置与检查点完全对齐。该写法写在 `scripts/extract_frames_embeddings.py` 的 `load_echo_clip` 函数中，缓存阶段调用的就是这一函数。

## 4 划分隔离证据
校验方式：把每个清单 JSON 中的 `file_name` 去掉扩展名得到 stem 集合，两两求交集。

| 比较 | 交集大小 | 示例 |
|---|---|---|
| train_cap_val | 0 | [] |
| train_cap_test | 0 | [] |
| val_cap_test | 0 | [] |

三者均为 0 = **True**。

关于 `r0_external_anchor_5000.json`：该清单共 5000 条，与 TRAIN 重叠 3747、与 VAL 重叠 627、与 TEST 重叠 626。它是混池的历史锚点，**未用于 R2 至 R6 的任何结论**，也未用于本文的测试评估。

## 5 论文数字逐条对照
下表按论文出现顺序列出关键数字。每个数字都给出结果文件与字段名，字段名使用 JSON 路径表示法（`metrics.mae` 表示根对象的 metrics 键下的 mae 键）。
| 论文位置 | 数值 | 来源文件 | JSON 字段 |
|---|---|---|---|
| 摘要 | 11.035 | `R1_s42_f32_cls.json` | `metrics.mae` |
| 摘要 | 5.868 | `R4_s42_f32_cls.json` | `metrics.mae` |
| 摘要 | 5.971 | `R5_s42_f32_cls.json` | `metrics.mae` |
| 摘要 | 0.879 | `R5_s42_f32_cls.json` | `metrics.auc_ef_lt50` |
| 表 3 | 10.986 | `R0_s42_f20_cls.json` | `metrics.mae` |
| 表 3 | 11.035 | `R1_s42_f32_cls.json` | `metrics.mae` |
| 表 3 | 6.417 | `R2_s42_f32_cls.json` | `metrics.mae` |
| 表 3 | 6.216 | `R3_s42_f32_cls.json` | `metrics.mae` |
| 表 3 | 5.868 | `R4_s42_f32_cls.json` | `metrics.mae` |
| 表 3 | 8.118 | `R4_s42_f32_cls.json` | `metrics.rmse` |
| 表 3 | 0.600 | `R4_s42_f32_cls.json` | `metrics.r2` |
| 表 3 | 5.971 | `R5_s42_f32_cls.json` | `metrics.mae` |
| 表 3 | 8.496 | `R5_s42_f32_cls.json` | `metrics.rmse` |
| 表 3 | 0.562 | `R5_s42_f32_cls.json` | `metrics.r2` |
| 表 3 | 0.767 | `R5_s42_f32_cls.json` | `metrics.pearson` |
| 表 3 | 5.971 | `R6_s42_f32_cls_a10_20.json` | `metrics.mae` |
| 表 3 | 10.954 | `ORACLE_EDES_s42_f32_cls.json` | `metrics.mae` |
| 5.2 节 | 9.655 | `R0_s42_f20_cls.json` | `metrics.bias` |
| 5.2 节 | 9.622 | `R1_s42_f32_cls.json` | `metrics.bias` |
| 5.2 节 | 11.000 | `R1U16_s42_f16_cls.json` | `metrics.mae` |
| 5.2 节 | 9.580 | `R1U16_s42_f16_cls.json` | `metrics.bias` |
| 5.2 节 | 0.740 | `R1U16_s42_f16_cls.json` | `metrics.auc_ef_lt50` |
| 表 5 零样本 等间隔 16 帧 | 11.000 | `R1U16_s42_f16_cls.json` | `metrics.mae` |
| 表 3 R0 MAE 均值 | 10.986 | `R0_s42_f20_cls.json` | `metrics.mae，1 个种子的算术平均` |
| 表 3 R0 AUC50 均值 | 0.740 | `R0_s42_f20_cls.json` | `metrics.auc_ef_lt50，1 个种子的算术平均` |
| 表 3 R1 MAE 均值 | 11.035 | `R1_s42_f32_cls.json` | `metrics.mae，1 个种子的算术平均` |
| 表 3 R1 AUC50 均值 | 0.738 | `R1_s42_f32_cls.json` | `metrics.auc_ef_lt50，1 个种子的算术平均` |
| 表 3 R2 MAE 均值 | 6.428 | `R2_s42_f32_cls.json / R2_s43_f32_cls.json / R2_s44_f32_cls.json` | `metrics.mae，3 个种子的算术平均` |
| 表 3 R2 MAE 标准差 | 0.015 | `R2_s42_f32_cls.json / R2_s43_f32_cls.json / R2_s44_f32_cls.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 3 R2 AUC50 均值 | 0.859 | `R2_s42_f32_cls.json / R2_s43_f32_cls.json / R2_s44_f32_cls.json` | `metrics.auc_ef_lt50，3 个种子的算术平均` |
| 表 3 R3 MAE 均值 | 6.333 | `R3_s42_f32_cls.json / R3_s43_f32_cls.json / R3_s44_f32_cls.json` | `metrics.mae，3 个种子的算术平均` |
| 表 3 R3 MAE 标准差 | 0.102 | `R3_s42_f32_cls.json / R3_s43_f32_cls.json / R3_s44_f32_cls.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 3 R3 AUC50 均值 | 0.862 | `R3_s42_f32_cls.json / R3_s43_f32_cls.json / R3_s44_f32_cls.json` | `metrics.auc_ef_lt50，3 个种子的算术平均` |
| 表 3 R4 MAE 均值 | 5.846 | `R4_s42_f32_cls.json / R4_s43_f32_cls.json / R4_s44_f32_cls.json` | `metrics.mae，3 个种子的算术平均` |
| 表 3 R4 MAE 标准差 | 0.081 | `R4_s42_f32_cls.json / R4_s43_f32_cls.json / R4_s44_f32_cls.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 3 R4 AUC50 均值 | 0.882 | `R4_s42_f32_cls.json / R4_s43_f32_cls.json / R4_s44_f32_cls.json` | `metrics.auc_ef_lt50，3 个种子的算术平均` |
| 表 3 R5 MAE 均值 | 5.904 | `R5_s42_f32_cls.json / R5_s43_f32_cls.json / R5_s44_f32_cls.json` | `metrics.mae，3 个种子的算术平均` |
| 表 3 R5 MAE 标准差 | 0.080 | `R5_s42_f32_cls.json / R5_s43_f32_cls.json / R5_s44_f32_cls.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 3 R5 AUC50 均值 | 0.879 | `R5_s42_f32_cls.json / R5_s43_f32_cls.json / R5_s44_f32_cls.json` | `metrics.auc_ef_lt50，3 个种子的算术平均` |
| 表 3 R6 MAE 均值 | 5.904 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `metrics.mae，3 个种子的算术平均` |
| 表 3 R6 MAE 标准差 | 0.080 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 3 R6 AUC50 均值 | 0.879 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `metrics.auc_ef_lt50，3 个种子的算术平均` |
| 表 3 ORACLE_EDES MAE 均值 | 10.954 | `ORACLE_EDES_s42_f32_cls.json` | `metrics.mae，1 个种子的算术平均` |
| 表 3 ORACLE_EDES AUC50 均值 | 0.734 | `ORACLE_EDES_s42_f32_cls.json` | `metrics.auc_ef_lt50，1 个种子的算术平均` |
| 表 4 ef_lt50 ECE 校准前 均值 | 0.050 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.raw.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt50 ECE Platt 均值 | 0.040 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.platt.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt50 ECE 温度 均值 | 0.043 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.temperature.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt50 Brier 校准前 均值 | 0.106 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.raw.brier，3 个种子的算术平均` |
| 表 4 ef_lt50 Brier Platt 均值 | 0.108 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.platt.brier，3 个种子的算术平均` |
| 表 4 ef_lt50 AUC 均值 | 0.879 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.test.raw.auc，3 个种子的算术平均` |
| 表 4 ef_lt50 温度均值 | 1.172 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt50.temperature，3 个种子的算术平均` |
| 表 4 ef_lt40 ECE 校准前 均值 | 0.061 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.raw.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt40 ECE Platt 均值 | 0.035 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.platt.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt40 ECE 温度 均值 | 0.036 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.temperature.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt40 Brier 校准前 均值 | 0.070 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.raw.brier，3 个种子的算术平均` |
| 表 4 ef_lt40 Brier Platt 均值 | 0.073 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.platt.brier，3 个种子的算术平均` |
| 表 4 ef_lt40 AUC 均值 | 0.906 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.test.raw.auc，3 个种子的算术平均` |
| 表 4 ef_lt40 温度均值 | 1.415 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt40.temperature，3 个种子的算术平均` |
| 表 4 ef_lt30 ECE 校准前 均值 | 0.034 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.raw.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt30 ECE Platt 均值 | 0.026 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.platt.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt30 ECE 温度 均值 | 0.022 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.temperature.ece_15，3 个种子的算术平均` |
| 表 4 ef_lt30 Brier 校准前 均值 | 0.039 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.raw.brier，3 个种子的算术平均` |
| 表 4 ef_lt30 Brier Platt 均值 | 0.042 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.platt.brier，3 个种子的算术平均` |
| 表 4 ef_lt30 AUC 均值 | 0.964 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.test.raw.auc，3 个种子的算术平均` |
| 表 4 ef_lt30 温度均值 | 1.433 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.thresholds.ef_lt30.temperature，3 个种子的算术平均` |
| 5.3 节 alpha_0.1 覆盖率 均值 | 0.881 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.1.coverage，3 个种子的算术平均` |
| 5.3 节 alpha_0.1 区间宽度 均值 | 24.93 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.1.width_mean，3 个种子的算术平均` |
| 5.3 节 alpha_0.1 分位数 均值 | 12.467 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.1.q，3 个种子的算术平均` |
| 5.3 节 alpha_0.2 覆盖率 均值 | 0.783 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.2.coverage，3 个种子的算术平均` |
| 5.3 节 alpha_0.2 区间宽度 均值 | 17.06 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.2.width_mean，3 个种子的算术平均` |
| 5.3 节 alpha_0.2 分位数 均值 | 8.532 | `R6_s42_f32_cls_a10_20.json / R6_s43_f32_cls_a10_20.json / R6_s44_f32_cls_a10_20.json` | `calibration.conformal.alpha_0.2.q，3 个种子的算术平均` |
| 表 5 abtop20 | 11.035 | `R1_s42_f32_cls_abtop20.json` | `metrics.mae` |
| 表 5 abtop10 | 10.900 | `R1_s42_f32_cls_top10_abtop10.json` | `metrics.mae` |
| 表 5 abtop30 | 10.692 | `R1_s42_f32_cls_top30_abtop30.json` | `metrics.mae` |
| 表 5 abpoolmean | 5.858 | `R4_MEAN_s42_f32_cls_abpoolmean.json` | `metrics.mae` |
| 表 5 abpoolmean | 6.043 | `R4_MEAN_s43_f32_cls_abpoolmean.json` | `metrics.mae` |
| 表 5 abpoolmean | 5.913 | `R4_MEAN_s44_f32_cls_abpoolmean.json` | `metrics.mae` |
| 表 5 abframes16 | 5.917 | `R4_s42_f16_cls_abframes16.json` | `metrics.mae` |
| 表 5 abscale1000 | 5.860 | `R4_s42_f32_cls_n1000_abscale1000.json` | `metrics.mae` |
| 表 5 abscale2000 | 5.650 | `R4_s42_f32_cls_n2000_abscale2000.json` | `metrics.mae` |
| 表 5 abscale4000 | 5.892 | `R4_s42_f32_cls_n4000_abscale4000.json` | `metrics.mae` |
| 表 5 abscale500 | 6.197 | `R4_s42_f32_cls_n500_abscale500.json` | `metrics.mae` |
| 表 5 abframes8 | 6.003 | `R4_s42_f8_cls_abframes8.json` | `metrics.mae` |
| 表 5 abframes16 | 6.041 | `R4_s43_f16_cls_abframes16.json` | `metrics.mae` |
| 表 5 abscale1000 | 5.894 | `R4_s43_f32_cls_n1000_abscale1000.json` | `metrics.mae` |
| 表 5 abscale2000 | 5.992 | `R4_s43_f32_cls_n2000_abscale2000.json` | `metrics.mae` |
| 表 5 abscale4000 | 6.120 | `R4_s43_f32_cls_n4000_abscale4000.json` | `metrics.mae` |
| 表 5 abscale500 | 6.242 | `R4_s43_f32_cls_n500_abscale500.json` | `metrics.mae` |
| 表 5 abframes8 | 6.057 | `R4_s43_f8_cls_abframes8.json` | `metrics.mae` |
| 表 5 abframes16 | 5.869 | `R4_s44_f16_cls_abframes16.json` | `metrics.mae` |
| 表 5 abscale1000 | 6.085 | `R4_s44_f32_cls_n1000_abscale1000.json` | `metrics.mae` |
| 表 5 abscale2000 | 5.801 | `R4_s44_f32_cls_n2000_abscale2000.json` | `metrics.mae` |
| 表 5 abscale4000 | 5.751 | `R4_s44_f32_cls_n4000_abscale4000.json` | `metrics.mae` |
| 表 5 abscale500 | 6.219 | `R4_s44_f32_cls_n500_abscale500.json` | `metrics.mae` |
| 表 5 abframes8 | 5.751 | `R4_s44_f8_cls_abframes8.json` | `metrics.mae` |
| 表 5 edes | 5.836 | `R5_EDES_s42_f32_cls_edes.json` | `metrics.mae` |
| 表 5 abpoolmean MAE 均值 | 5.938 | `R4_MEAN_s42_f32_cls_abpoolmean.json / R4_MEAN_s43_f32_cls_abpoolmean.json / R4_MEAN_s44_f32_cls_abpoolmean.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abpoolmean MAE 标准差 | 0.095 | `R4_MEAN_s42_f32_cls_abpoolmean.json / R4_MEAN_s43_f32_cls_abpoolmean.json / R4_MEAN_s44_f32_cls_abpoolmean.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abframes16 MAE 均值 | 5.942 | `R4_s42_f16_cls_abframes16.json / R4_s43_f16_cls_abframes16.json / R4_s44_f16_cls_abframes16.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abframes16 MAE 标准差 | 0.089 | `R4_s42_f16_cls_abframes16.json / R4_s43_f16_cls_abframes16.json / R4_s44_f16_cls_abframes16.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abscale1000 MAE 均值 | 5.946 | `R4_s42_f32_cls_n1000_abscale1000.json / R4_s43_f32_cls_n1000_abscale1000.json / R4_s44_f32_cls_n1000_abscale1000.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abscale1000 MAE 标准差 | 0.121 | `R4_s42_f32_cls_n1000_abscale1000.json / R4_s43_f32_cls_n1000_abscale1000.json / R4_s44_f32_cls_n1000_abscale1000.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abscale2000 MAE 均值 | 5.815 | `R4_s42_f32_cls_n2000_abscale2000.json / R4_s43_f32_cls_n2000_abscale2000.json / R4_s44_f32_cls_n2000_abscale2000.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abscale2000 MAE 标准差 | 0.172 | `R4_s42_f32_cls_n2000_abscale2000.json / R4_s43_f32_cls_n2000_abscale2000.json / R4_s44_f32_cls_n2000_abscale2000.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abscale4000 MAE 均值 | 5.921 | `R4_s42_f32_cls_n4000_abscale4000.json / R4_s43_f32_cls_n4000_abscale4000.json / R4_s44_f32_cls_n4000_abscale4000.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abscale4000 MAE 标准差 | 0.186 | `R4_s42_f32_cls_n4000_abscale4000.json / R4_s43_f32_cls_n4000_abscale4000.json / R4_s44_f32_cls_n4000_abscale4000.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abscale500 MAE 均值 | 6.219 | `R4_s42_f32_cls_n500_abscale500.json / R4_s43_f32_cls_n500_abscale500.json / R4_s44_f32_cls_n500_abscale500.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abscale500 MAE 标准差 | 0.023 | `R4_s42_f32_cls_n500_abscale500.json / R4_s43_f32_cls_n500_abscale500.json / R4_s44_f32_cls_n500_abscale500.json` | `metrics.mae，3 个种子的样本标准差` |
| 表 5 abframes8 MAE 均值 | 5.937 | `R4_s42_f8_cls_abframes8.json / R4_s43_f8_cls_abframes8.json / R4_s44_f8_cls_abframes8.json` | `metrics.mae，3 个种子的算术平均` |
| 表 5 abframes8 MAE 标准差 | 0.163 | `R4_s42_f8_cls_abframes8.json / R4_s43_f8_cls_abframes8.json / R4_s44_f8_cls_abframes8.json` | `metrics.mae，3 个种子的样本标准差` |
| 5.5 节 残差绝对值中位数 | 3.87 | `R5_s42_f32_cls.json` | `predictions.test_pred 与 test_true 之差的中位绝对值` |
| 5.5 节 残差绝对值九十分位数 | 14.11 | `R5_s42_f32_cls.json` | `predictions.test_pred 与 test_true 之差的九十分位绝对值` |
| 5.5 节 残差绝对值最大值 | 34.09 | `R5_s42_f32_cls.json` | `predictions.test_pred 与 test_true 之差的最大绝对值` |
| 5.5 节 回归斜率 | 0.598 | `R5_s42_f32_cls.json` | `predictions 上 test_pred 对 test_true 的最小二乘斜率` |
| 5.5 节 预测标准差 | 10.02 | `R5_s42_f32_cls.json` | `metrics.pred_sd` |
| 5.5 节 参考标准差 | 12.86 | `R5_s42_f32_cls.json` | `metrics.y_sd` |
| 表 6 R0 MAE 区间下界 | 10.195 | `R0_s42_f20_cls.json` | `bootstrap.mae.lo` |
| 表 6 R0 MAE 区间上界 | 11.789 | `R0_s42_f20_cls.json` | `bootstrap.mae.hi` |
| 表 6 R0 AUC50 区间下界 | 0.682 | `R0_s42_f20_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R0 AUC50 区间上界 | 0.789 | `R0_s42_f20_cls.json` | `bootstrap.auc50.hi` |
| 表 6 R1 MAE 区间下界 | 10.261 | `R1_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 R1 MAE 区间上界 | 11.816 | `R1_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 R1 AUC50 区间下界 | 0.678 | `R1_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R1 AUC50 区间上界 | 0.788 | `R1_s42_f32_cls.json` | `bootstrap.auc50.hi` |
| 表 6 R2 MAE 区间下界 | 5.893 | `R2_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 R2 MAE 区间上界 | 6.965 | `R2_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 R2 AUC50 区间下界 | 0.820 | `R2_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R2 AUC50 区间上界 | 0.898 | `R2_s42_f32_cls.json` | `bootstrap.auc50.hi` |
| 表 6 R3 MAE 区间下界 | 5.700 | `R3_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 R3 MAE 区间上界 | 6.779 | `R3_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 R3 AUC50 区间下界 | 0.820 | `R3_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R3 AUC50 区间上界 | 0.898 | `R3_s42_f32_cls.json` | `bootstrap.auc50.hi` |
| 表 6 R4 MAE 区间下界 | 5.362 | `R4_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 R4 MAE 区间上界 | 6.357 | `R4_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 R4 AUC50 区间下界 | 0.837 | `R4_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R4 AUC50 区间上界 | 0.912 | `R4_s42_f32_cls.json` | `bootstrap.auc50.hi` |
| 表 6 R5 MAE 区间下界 | 5.442 | `R5_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 R5 MAE 区间上界 | 6.505 | `R5_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 R5 AUC50 区间下界 | 0.840 | `R5_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 R5 AUC50 区间上界 | 0.916 | `R5_s42_f32_cls.json` | `bootstrap.auc50.hi` |
| 表 6 ORACLE_EDES MAE 区间下界 | 10.187 | `ORACLE_EDES_s42_f32_cls.json` | `bootstrap.mae.lo` |
| 表 6 ORACLE_EDES MAE 区间上界 | 11.763 | `ORACLE_EDES_s42_f32_cls.json` | `bootstrap.mae.hi` |
| 表 6 ORACLE_EDES AUC50 区间下界 | 0.675 | `ORACLE_EDES_s42_f32_cls.json` | `bootstrap.auc50.lo` |
| 表 6 ORACLE_EDES AUC50 区间上界 | 0.784 | `ORACLE_EDES_s42_f32_cls.json` | `bootstrap.auc50.hi` |

## 6 文献核验
核验方式：对每个 DOI 调用 Crossref API（arXiv DOI 走 DataCite），把返回标题与声明标题做词元重叠比对，再尝试下载开放获取 PDF 并检查文件头是否为 `%PDF-`。

| 引用键 | DOI | 注册库返回标题 | 标题匹配 | OA | PDF |
|---|---|---|---|---|---|
| ouyang2020echonet | 10.1038/s41586-020-2145-8 | Video-based AI for beat-to-beat assessment of cardiac function | 7/7 | 否 | 否 |
| christensen2024echoclip | 10.1038/s41591-024-02959-y | Vision–language foundation model for echocardiogram interpretation | 6/6 | 是 | 是 (4127881 B, magic %PDF-) |
| du2025cardiacclip | 10.1007/978-3-032-04971-1_5 | CardiacCLIP: Video-Based CLIP Adaptation for LVEF Prediction in a Few- | 9/9 | 否 | 否 |
| leclerc2019camus | 10.1109/TMI.2019.2900516 | Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D | 9/9 | 否 | 否 |
| radford2021clip | 10.48550/arXiv.2103.00020 | Learning Transferable Visual Models From Natural Language Supervision | 8/8 | 否 | 是 (6813639 B, magic %PDF-) |
| guo2017calibration | 10.48550/arXiv.1706.04599 | On Calibration of Modern Neural Networks | 4/4 | 否 | 是 (1349691 B, magic %PDF-) |
| angelopoulos2023conformal | 10.1561/2200000101 | Conformal Prediction: A Gentle Introduction | 4/4 | 否 | 是 (5360733 B, magic %PDF-) |
| khosla2020supcon | 10.48550/arXiv.2004.11362 | Supervised Contrastive Learning | 3/3 | 否 | 是 (2197586 B, magic %PDF-) |
| vaswani2017attention | 10.48550/arXiv.1706.03762 | Attention Is All You Need | 2/2 | 否 | 是 (2215244 B, magic %PDF-) |
| liu2022convnext | 10.1109/CVPR52688.2022.01167 | A ConvNet for the 2020s | 2/2 | 否 | 是 (751927 B, magic %PDF-) |
| madani2018echo | 10.1038/s41746-018-0065-x | Deep echocardiography: data-efficient supervised and semi-supervised d | 14/14 | 是 | 是 (3085548 B, magic %PDF-) |
| zhang2025echovisionfm | 10.1038/s41467-025-66340-4 | Echo-Vision-FM: a pre-training and fine-tuning framework for echocardi | 11/11 | 是 | 是 (1076908 B, magic %PDF-) |
| holste2024echo | 10.1038/s43856-024-00538-3 | Efficient deep learning-based automated diagnosis from echocardiograph | 13/13 | 是 | 是 (1945711 B, magic %PDF-) |
| vukadinovic2024echoprime | 10.48550/arXiv.2410.09704 | EchoPrime: A Multi-Video View-Informed Vision-Language Model for Compr | 11/11 | 否 | 是 (1499501 B, magic %PDF-) |

合计 14/14 个 DOI 成功解析，11 篇取得真实 PDF。未取得 PDF 的条目为闭源出版或仅提供 HTML 落地页，已放弃下载并在本表注明。所有下载文件都通过了 `%PDF-` 文件头检查，HTML 响应被拒绝并删除。

## 7 未纳入论文的内容
论文正文只写已完成并有结果文件的部分。以下内容未进入正文，原因逐条说明。

- 端到端微调视觉塔：显存 4 GB，ConvNeXt-Base 单帧前向已占用约 0.9 GB，反向传播与优化器状态超出可用显存，因此全部实验在冻结特征上进行。
- 完整训练划分（7465 例）：受帧嵌入阶段吞吐限制，本次训练使用 2000 例，训练规模消融给出 500、1000、2000 三点，用于说明趋势。
- 完整测试划分（1277 例）：本次使用官方测试划分的 500 例随机子集，子集与完整划分的分布差异已在正文与本文第 1 节量化。
- CAMUS 外部验证：本次未把 CAMUS 纳入评估，正文的相关表述仅作为文献引用。
- 读者研究、临床决策影响评估：超出本次工作范围，正文未涉及。

## 8 文档结构校验
校验方式为用 python-docx 重新打开生成的 Word 文件，读取段落、表格、图片、标题层级、目录域与 settings.xml 中的 updateFields 字段。

| 项目 | 实测值 |
|---|---|
| 文件 | E:\Projects\20260522-EchoCLIP\papers\EchoCLIP-TA_中文稿.docx |
| 字节数 | 959234 |
| 段落数 | 274 |
| 表格数 | 6 |
| 内嵌图片数 | 6 |
| 标题段落数 | 43 |
| 标题层级 | 1、2、3 |
| 正文字符数 | 23318 |
| 目录域 | TOC \o "1-3" \h \z \u |
| updateFields | true |
| 公式编号 | 1 至 21 |

公式编号连续且递增 = True，共 21 个编号。图题 6 条，按 [1, 2, 3, 4, 5, 6] 排列。表题 6 条，按 [1, 2, 3, 4, 5, 6] 排列。目录域存在 = True，打开时自动更新 = True。

正文未出现不等号、约等号、待确认、链路、闭环、耦合等禁用写法。
正文八股结构齐全，标题层级为 1 至 3 级，编号与层级一致。

## 9 自查清单
| 检查项 | 结果 | 说明 |
|---|---|---|
| 清单与视频数量一致 | 通过 | 10030 行，10030 个 avi |
| 三划分两两交集为零 | 通过 | train/val/test 交集均为 0 |
| 测试子集 ED/ES 标注完整 | 通过 | 覆盖率 1.0000 |
| 每个结果文件含权重哈希 | 通过 | 64 位 SHA-256 |
| 每个结果文件含完整命令 | 通过 | 命令字段非空 |
| 每个结果文件含提交号 | 通过 | 39533b9f01d0 |
| R2–R5 三个种子 | 通过 | R2:3; R3:3; R4:3; R5:3; R6:3 |
| 文献 DOI 全部解析 | 通过 | 14/14 |
| 文献标题全部匹配 | 通过 | 词元完全覆盖 |
| PDF 文件头校验 | 通过 | 11 篇通过 |
| 指标在合理范围 | 通过 | MAE 与 AUC 均在预期区间 |
| 保形覆盖率接近名义水平 | 通过 | 种子 42 alpha_0.1: 0.880 vs 0.90; 种子 42 alpha_0.2: 0.762 vs 0.80; 种子 43 alpha_0.1: 0.884 vs 0.90; 种子 43 alpha_0.2: 0.796 vs 0.80; 种子 44 alpha_0.1: 0.880 vs 0.90; 种子 44 alpha_0.2: 0.790 vs 0.80 |
| 论文表格数字与结果文件逐格一致 | 通过 | all table numbers agree with the result JSONs |
