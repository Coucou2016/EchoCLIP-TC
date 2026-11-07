import re
from pathlib import Path

s = Path("scripts/build_manuscript.py").read_text(encoding="utf-8")
print("== captions ==")
for m in re.finditer(r'"((?:表|图) \d[^"]{0,50})', s):
    print(m.group(1))
print("== calls ==")
for m in re.finditer(r"_table_\w+\(doc|add_picture\(doc, \"([a-z0-9_]+)\"", s):
    print(m.group(0)[:70])
print("== section refs in prose ==")
for m in re.finditer(r"(表|图) (\d+)", s):
    pass
from collections import Counter
c = Counter(m.group(0) for m in re.finditer(r"(?:表|图) \d+", s))
print(sorted(c.items()))
