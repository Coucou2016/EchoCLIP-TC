from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

EDITS = [
    ('"图 1 给出整体结构。输入是一段心尖四腔超声视频，先做等间隔帧采样，"',
     '"整体结构如图 1 所示。输入是一段心尖四腔超声视频，先做等间隔帧采样，"'),
    ('f"表 3 给出六类预测头在测试子集上的主要指标。"',
     'f"六类预测头在测试子集上的主要指标见表 3。"'),
    ('"表 4 汇总校准前后的概率指标。" + "；".join(parts) + "。"',
     '"校准前后的概率指标见表 4。" + "，".join(parts) + "。"'),
    ('f"图 6 展示 {EXP_LABEL.get(exp, exp)} 的预测与残差。"',
     'f"{EXP_LABEL.get(exp, exp)} 的预测值与残差见图 6。"'),
    ('f"表 6 给出自助重采样的置信区间。{EXP_LABEL.get(exp, exp)} 的平均绝对误差点估计为 "',
     'f"自助重采样的置信区间见表 6。{EXP_LABEL.get(exp, exp)} 的平均绝对误差点估计为 "'),
]
for old, new in EDITS:
    assert old in s, old[:50]
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("ok", len(EDITS))
