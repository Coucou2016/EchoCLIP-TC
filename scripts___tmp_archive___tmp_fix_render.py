"""Replace the ten inline render loops with one helper that honours after-notes."""
import re
from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

helper = '''
def _render(doc, blocks) -> None:
    """Emit a builder's blocks: lead-in prose, numbered equation, optional note."""
    for para, formula in blocks:
        if para:
            add_body(doc, para)
        if not formula:
            continue
        add_equation(doc, formula[0], formula[1])
        if len(formula) > 2 and formula[2]:
            add_body(doc, formula[2])


class _EqCounter:'''

s = s.replace("\nclass _EqCounter:", helper, 1)

BUILDERS = [
    "_m_problem", "_m_encoder", "_m_sampling", "_m_zeroshot", "_m_linear",
    "_m_mlp", "_m_temporal", "_m_contrastive", "_m_calibration", "_m_metrics",
]

count = 0
for name in BUILDERS:
    pat = re.compile(
        r"    for para, formula in " + name + r"\(eq\):\n"
        r"        if para:\n"
        r"            add_body\(doc, para\)\n"
        r"        if formula:\n"
        r"            add_equation\(doc, formula\[0\], formula\[1\]\)\n"
    )
    new = "    _render(doc, " + name + "(eq))\n"
    s, n = pat.subn(new, s)
    count += n

p.write_text(s, encoding="utf-8")
print("loops replaced:", count)
