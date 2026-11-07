import re
from pathlib import Path

s = Path("scripts/build_manuscript.py").read_text(encoding="utf-8")
print("不是:", s.count("不是"), "而是:", s.count("而是"))
bad = ["链路", "回路", "闭环", "耦合", "待确定", "待确认", "TBD", "待补充"]
print({w: s.count(w) for w in bad})
for sym in ["≠", "≈"]:
    for m in re.finditer(re.escape(sym), s):
        print(sym, repr(s[max(0, m.start() - 40):m.start() + 20]))
