"""Rewrite the manuscript build loops so a paired formula keeps its lead-in text.

Before:  if formula: add_equation(...) else: add_body(...)
After:   if para: add_body(...)  /  if formula: add_equation(...)

This preserves the explanation sentence that introduces every numbered
equation, which was being dropped.
"""
import re
from pathlib import Path

p = Path(r"E:\Projects\20260522-EchoCLIP\scripts\build_manuscript.py")
s = p.read_text(encoding="utf-8")

pat = re.compile(
    r"( +)for para, formula in (_m_\w+)\(eq\):\n"
    r" +if formula:\n"
    r" +add_equation\(doc, formula\[0\], formula\[1\]\)\n"
    r" +else:\n"
    r" +add_body\(doc, para\)\n"
)

def repl(m):
    ind, fn = m.group(1), m.group(2)
    return (
        f"{ind}for para, formula in {fn}(eq):\n"
        f"{ind}    if para:\n"
        f"{ind}        add_body(doc, para)\n"
        f"{ind}    if formula:\n"
        f"{ind}        add_equation(doc, formula[0], formula[1])\n"
    )

s2, n = pat.subn(repl, s)
print("rewritten loops:", n)
p.write_text(s2, encoding="utf-8")
