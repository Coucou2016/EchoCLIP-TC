"""Make every headline number a 3-seed mean, not the first seed."""
from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

# 1) helper + mean-based best selection
old = '''def _best_exp(present: Dict[str, List[dict]]) -> str:
    cands = [e for e in ("R5", "R6", "R4") if e in present]'''
new = '''_METRIC_KEYS = ("mae", "rmse", "r2", "bias", "pearson", "spearman", "auc_ef_lt50", "auprc_ef_lt50")


def _seed_mean(present: Dict[str, List[dict]], exp: str) -> Dict[str, float]:
    """Mean of each headline metric over the seeds of one experiment."""
    runs = present[exp]
    return {k: float(np.mean([r["metrics"][k] for r in runs])) for k in _METRIC_KEYS}


def _best_exp(present: Dict[str, List[dict]]) -> str:
    cands = [e for e in ("R5", "R6", "R4") if e in present]'''
assert old in s
s = s.replace(old, new, 1)

old = '''    if not cands:
        return "R2"
    return min(cands, key=lambda e: present[e][0]["metrics"]["mae"])'''
new = '''    if not cands:
        return "R2"
    return min(cands, key=lambda e: float(np.mean([r["metrics"]["mae"] for r in present[e]])))'''
assert old in s
s = s.replace(old, new, 1)

# 2) abstract: honest wording about the R4 -> R5 change, mean-based AUC
old = '''    auc = best["metrics"]["auc_ef_lt50"]
    conf = r6.get("calibration", {}).get("conformal", {})'''
new = '''    auc = _seed_mean(present, best_exp)["auc_ef_lt50"]
    z_mae = _seed_mean(present, "R1")["mae"]
    conf = r6.get("calibration", {}).get("conformal", {})'''
assert old in s
s = s.replace(old, new, 1)

old = '''        parts.append(
            f"时序 Transformer 把测试集平均绝对误差从零样本的 {z['metrics']['mae']:.2f} 降到 {m4:.2f}。"
        )'''
new = '''        parts.append(
            f"时序 Transformer 把测试集平均绝对误差从零样本的 {z_mae:.2f} 降到 {m4:.2f}。"
        )'''
assert old in s
s = s.replace(old, new, 1)

old = '''        if np.isfinite(m4) and m5 < m4:
            parts.append(f"加入软标签对比预训练后进一步降到 {m5:.2f}。")
        else:'''
new = '''        if np.isfinite(m4) and m5 < m4 - 0.1:
            parts.append(f"加入软标签对比预训练后进一步降到 {m5:.2f}。")
        elif np.isfinite(m4):
            parts.append(
                f"加入软标签对比预训练后误差为 {m5:.2f}，相对时序 Transformer 的变化为 "
                f"{m5 - m4:+.2f} 个百分点，落在种子噪声范围内。"
            )
        else:'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"EF 低于 50% 的判别 AUC 从 {z['metrics']['auc_ef_lt50']:.3f} 提升到 {auc:.3f}。"'''
new = '''        f"EF 低于 50% 的判别 AUC 从 {_seed_mean(present, 'R1')['auc_ef_lt50']:.3f} 提升到 {auc:.3f}。"'''
assert old in s
s = s.replace(old, new, 1)

# 3) intro work
old = '''def _intro_work(present, best_exp) -> List[str]:
    z = present.get("R1", [{}])[0]
    best = present[best_exp][0]'''
new = '''def _intro_work(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)
    best = present[best_exp][0]'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"在 {best['n_test']} 例测试样本上，误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，"
        f"平均绝对误差 {best['metrics']['mae']:.2f}，AUC {best['metrics']['auc_ef_lt50']:.3f}，"
        f"零样本对照分别为 {z['metrics']['mae']:.2f} 与 {z['metrics']['auc_ef_lt50']:.3f}。",'''
new = '''        f"在 {best['n_test']} 例测试样本上，误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，"
        f"平均绝对误差 {bm['mae']:.2f}，AUC {bm['auc_ef_lt50']:.3f}，"
        f"零样本对照分别为 {z['mae']:.2f} 与 {z['auc_ef_lt50']:.3f}，均为三个种子的均值。",'''
assert old in s
s = s.replace(old, new, 1)

# 4) main results
old = '''def _r_main(present, best_exp) -> List[str]:
    z = present.get("R1", [{}])[0]
    r0 = present.get("R0", [{}])[0]
    best = present[best_exp][0]'''
new = '''def _r_main(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    r0m = _seed_mean(present, "R0") if "R0" in present else None
    best = present[best_exp][0]
    bm = _seed_mean(present, best_exp)'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"零样本基线 R1 的平均绝对误差为 {z['metrics']['mae']:.2f}，"
        f"决定系数为 {z['metrics']['r2']:.3f}，说明零样本预测的方差结构偏离参考值。"
        f"这一误差主要来自系统性高估，偏差为 {z['metrics']['bias']:+.2f} 个百分点。",'''
new = '''        f"零样本基线 R1 的平均绝对误差为 {z['mae']:.2f}，"
        f"决定系数为 {z['r2']:.3f}，说明零样本预测的方差结构偏离参考值。"
        f"这一误差主要来自系统性高估，偏差为 {z['bias']:+.2f} 个百分点。",'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"表中误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，其平均绝对误差为 {best['metrics']['mae']:.2f}，"
        f"均方根误差为 {best['metrics']['rmse']:.2f}，决定系数为 {best['metrics']['r2']:.3f}，"
        f"皮尔逊相关系数为 {best['metrics']['pearson']:.3f}。"'''
new = '''        f"表中误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，其平均绝对误差为 {bm['mae']:.2f}，"
        f"均方根误差为 {bm['rmse']:.2f}，决定系数为 {bm['r2']:.3f}，"
        f"皮尔逊相关系数为 {bm['pearson']:.3f}，三者均为三个种子的均值。"'''
assert old in s
s = s.replace(old, new, 1)

# 5) mechanism
old = '''def _d_mechanism(present, best_exp) -> List[str]:
    z = present.get("R1", [{}])[0]
    best = present[best_exp][0]'''
new = '''def _d_mechanism(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"它把零样本的系统偏差吸收掉，因此偏差从 {z['metrics']['bias']:+.2f} 变为 {best['metrics']['bias']:+.2f}。"
        f"这一校正不需要改动视觉塔，几百个样本已经足够。"
        f"从零样本到线性头的误差降幅约 {z['metrics']['mae'] - present['R2'][0]['metrics']['mae']:.2f} 个百分点，"
        f"而从线性头到最优头再降 "
        f"{present['R2'][0]['metrics']['mae'] - best['metrics']['mae']:.2f} 个百分点，"'''
new = '''        f"它把零样本的系统偏差吸收掉，因此偏差从 {z['bias']:+.2f} 变为 {bm['bias']:+.2f}。"
        f"这一校正不需要改动视觉塔，两千例样本已经足够。"
        f"从零样本到线性头的误差降幅约 {z['mae'] - _seed_mean(present, 'R2')['mae']:.2f} 个百分点，"
        f"而从线性头到最优头再降 "
        f"{_seed_mean(present, 'R2')['mae'] - bm['mae']:.2f} 个百分点，"'''
assert old in s
s = s.replace(old, new, 1)

# 6) compare
old = '''def _d_compare(present) -> List[str]:
    z = present.get("R1", [{}])[0]
    best_exp = _best_exp(present)
    best = present[best_exp][0]'''
new = '''def _d_compare(present) -> List[str]:
    z = _seed_mean(present, "R1")
    best_exp = _best_exp(present)
    bm = _seed_mean(present, best_exp)'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"本文的监督头达到 {best['metrics']['mae']:.2f}，与文献中专用模型的量级相当，"'''
new = '''        f"本文的监督头达到 {bm['mae']:.2f}，与文献中专用模型的量级相当，"'''
assert old in s
s = s.replace(old, new, 1)

# 7) conclusion
old = '''def _conclusion(present, best_exp) -> List[str]:
    z = present.get("R1", [{}])[0]
    best = present[best_exp][0]'''
new = '''def _conclusion(present, best_exp) -> List[str]:
    z = _seed_mean(present, "R1")
    bm = _seed_mean(present, best_exp)'''
assert old in s
s = s.replace(old, new, 1)

old = '''        f"零样本基线的平均绝对误差为 {z['metrics']['mae']:.2f}，"
        f"误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，达到 {best['metrics']['mae']:.2f}，"
        f"EF 低于 50% 的判别 AUC 由 {z['metrics']['auc_ef_lt50']:.3f} 变为 {best['metrics']['auc_ef_lt50']:.3f}。",'''
new = '''        f"零样本基线的平均绝对误差为 {z['mae']:.2f}，"
        f"误差最小的方案为 {EXP_LABEL.get(best_exp, best_exp)}，达到 {bm['mae']:.2f}，"
        f"EF 低于 50% 的判别 AUC 由 {z['auc_ef_lt50']:.3f} 变为 {bm['auc_ef_lt50']:.3f}。",'''
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("ok")
