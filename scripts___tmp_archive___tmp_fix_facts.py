from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

EDITS = [
    # 3.5.1 epoch budget must match the command that was run (40)
    ('"学习率按余弦退火衰减，最多一百二十轮，验证集平均绝对误差连续二十五轮不下降则停止，"',
     '"学习率按余弦退火衰减，最多四十轮，验证集平均绝对误差连续二十五轮不下降则停止，"'),
    # 3.7 pretrain epochs must match the run (20)
    ('("本文取 σ 等于五，τ 等于五，预训练四十轮，随后接回归头微调。"',
     '("本文取 σ 等于五，τ 等于五，预训练二十轮，随后接回归头微调。"'),
    # 4.4 command list: real commands with the real arguments
    ('''    for para in (
        "全部实验由三个脚本完成。脚本一负责冻结塔的帧嵌入缓存，脚本二负责头部训练与指标计算，脚本三负责消融与统计检验。命令按执行顺序列出。",
    ):
        add_body(doc, para)
    _code_block(doc, [
        "python scripts/extract_frames_embeddings.py --split test500 --limit 500 --frames 32",
        "python scripts/extract_frames_embeddings.py --split val --limit 1288 --frames 32",
        "python scripts/extract_frames_embeddings.py --split train --limit 2000 --frames 32",
        "python scripts/train_heads_from_cache.py --experiment R2 --seeds 42 43 44",
        "python scripts/run_ablation_matrix.py --group all",
        "python scripts/audit_data.py",
        "python scripts/make_figures.py",
        "python scripts/build_manuscript.py",
    ])''',
     '''    for para in (
        "实验由五个脚本完成。脚本一负责冻结塔的帧嵌入缓存，脚本二负责头部训练与指标计算，"
        "脚本三负责消融矩阵，脚本四负责统计检验，脚本五负责审计与出图。命令按执行顺序列出，"
        "每条命令的完整字符串同时写在对应结果文件的 command 字段中。",
    ):
        add_body(doc, para)
    _code_block(doc, [
        "python scripts/extract_frames_embeddings.py --split test500 --limit 500 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split val --limit 1288 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split train --limit 2000 --frames 32 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split test500 --stride --frames 20 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split train --frames 8 --batch-size 16",
        "python scripts/extract_frames_embeddings.py --split train --frames 16 --batch-size 16",
        "python scripts/train_heads_from_cache.py --experiment R2 --seeds 42 43 44",
        "python scripts/train_heads_from_cache.py --experiment R3 --seeds 42 43 44",
        "python scripts/train_heads_from_cache.py --experiment R4 --seeds 42 43 44",
        "python scripts/train_heads_from_cache.py --experiment R5 --seeds 42 43 44",
        "python scripts/train_heads_from_cache.py --experiment R6 --seeds 42 43 44",
        "python scripts/train_heads_from_cache.py --experiment ORACLE_EDES --seeds 42",
        "python scripts/run_ablation_matrix.py --group all",
        "python scripts/run_stat_tests.py",
        "python scripts/audit_data.py",
        "python scripts/make_figures.py",
        "python scripts/build_manuscript.py",
    ])'''),
]
for old, new in EDITS:
    assert old in s, old[:60]
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("ok")
