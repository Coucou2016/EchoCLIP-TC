"""Convert 'note-after-equation' paragraphs into the third tuple slot.

The methods builders return (lead_in, (formula, number)). A note that reads as an
explanation of the symbols belongs *after* the equation, so it moves to a third
slot: (lead_in, (formula, number, after_text)).
"""
from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

EDITS = [
    # (old text block, new text block)
    (
        '''        ("其中 f 表示视觉塔，输出再做一次二范数归一化，使余弦相似度等于内积。"
         "归一化后的向量落在单位球面上，因此不同帧之间的相似度可以直接比较。",
         ("z = f(x) / || f(x) ||_2,  z ∈ R^512", eq.next())),''',
        '''        ("", ("z = f(x) / || f(x) ||_2,  z ∈ R^512", eq.next(),
               "其中 f 表示视觉塔，输出再做一次二范数归一化，使余弦相似度等于内积。"
               "归一化后的向量落在单位球面上，因此不同帧之间的相似度可以直接比较。")),''',
    ),
    (
        '''        ("当 N 不小于 K 时该式给出严格递增的下标序列，覆盖从首帧到末帧的整段视频。"
         "当 N 小于 K 时下标重复取最后一帧。本文数据的最短视频为二十八帧，"
         "因此三十二帧采样时不会出现重复，八帧与十六帧采样同样不会。",
         ("i_j = round( j × (N − 1) / (K − 1) ),  j = 0, …, K − 1", eq.next())),''',
        '''        ("", ("i_j = round( j × (N − 1) / (K − 1) ),  j = 0, …, K − 1", eq.next(),
               "当 N 不小于 K 时该式给出严格递增的下标序列，覆盖从首帧到末帧的整段视频。"
               "当 N 小于 K 时下标重复取最后一帧。本文数据的最短视频为二十八帧，"
               "因此三十二帧采样时不会出现重复，八帧与十六帧采样同样不会。")),''',
    ),
    (
        '''        ("该权重在标签差为零时取一，随差值增大而衰减，批内归一化后作为软目标分布。"
         "归一化使每一行的权重之和为一，从而可以直接与交叉熵配合。",
         ("w_ij = exp( −(y_i − y_j)^2 / (2 σ^2) ) / Σ_k exp( −(y_i − y_k)^2 / (2 σ^2) )", eq.next())),''',
        '''        ("", ("w_ij = exp( −(y_i − y_j)^2 / (2 σ^2) ) / Σ_k exp( −(y_i − y_k)^2 / (2 σ^2) )", eq.next(),
               "该权重在标签差为零时取一，随差值增大而衰减，批内归一化后作为软目标分布。"
               "归一化使每一行的权重之和为一，从而可以直接与交叉熵配合。")),''',
    ),
    (
        '''        ("投影头把视频表示映射到一百二十八维并归一化，记为 u。"
         "对比损失为软目标交叉熵，温度记为 τ，其形式如下。"
         "批大小记为 B，内层求和遍历批内全部样本。",
         ("L_con = − (1 / B) Σ_i Σ_j w_ij log softmax_j( u_i · u_j / τ )", eq.next())),''',
        '''        ("投影头把视频表示映射到一百二十八维并归一化，记为 u。"
         "对比损失为软目标交叉熵，温度记为 τ，其形式如下。",
         ("L_con = − (1 / B) Σ_i Σ_j w_ij log softmax_j( u_i · u_j / τ )", eq.next(),
          "批大小记为 B，内层求和遍历批内全部样本。")),''',
    ),
    (
        '''        ("其中 σ_0 固定为四，表示四个百分点的尺度，θ 为阈值，ŷ 为回归头的输出。"
         "尺度取四是因为测试集上残差的标准差在这个量级，"
         "取四可以使初始概率落在零与一之间而不饱和。",
         ("p = 1 / (1 + exp( −(θ − ŷ) / σ_0 )),  σ_0 = 4", eq.next())),''',
        '''        ("", ("p = 1 / (1 + exp( −(θ − ŷ) / σ_0 )),  σ_0 = 4", eq.next(),
               "其中 σ_0 固定为四，表示四个百分点的尺度，θ 为阈值，ŷ 为回归头的输出。"
               "尺度取四是因为验证集上残差的标准差在这个量级，"
               "取四可以使初始概率落在零与一之间而不饱和。")),''',
    ),
]

for old, new in EDITS:
    assert old in s, old[:60]
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("edits applied:", len(EDITS))
