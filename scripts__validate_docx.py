"""Reopen the generated docx and report structure + style compliance."""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import exports_dir  # noqa: E402

PATHS = [
    Path(r"E:\Projects\20260522-EchoCLIP\papers\EchoCLIP-TA_中文稿.docx"),
    exports_dir() / "EchoCLIP-TA_中文稿.docx",
]

for p in PATHS:
    print("=" * 70)
    print(p, p.stat().st_size, "bytes")
    doc = Document(str(p))
    paras = doc.paragraphs
    print("paragraphs:", len(paras))
    print("tables:", len(doc.tables))
    imgs = [r for r in doc.part.rels.values() if "image" in r.reltype]
    print("images:", len(imgs))
    heads = [(x.style.name, x.text) for x in paras if x.style.name.startswith("Heading")]
    print("headings:", len(heads))
    for s, t in heads:
        print("   ", s, "|", t[:50])
    xml = doc.element.xml
    print("TOC field present:", 'TOC \\o "1-3"' in xml or 'TOC \\o &quot;1-3&quot;' in xml)
    with zipfile.ZipFile(p) as z:
        settings = z.read("word/settings.xml").decode("utf-8", "replace")
        print("updateFields:", "updateFields" in settings)
    body = "\n".join(x.text for x in paras)
    print("chars:", len(body))
    for sym in ["≠", "≈", "TBD", "待确认", "待确定", "链路", "闭环", "耦合"]:
        if sym in body:
            print("  !! contains", sym, body.count(sym))
    # equation numbering must be 1..N ascending in order of appearance
    nums = [int(m) for m in re.findall(r"\((\d{1,2})\)\s*$", body, re.M)]
    print("equation numbers found:", nums)
    print("dup/ordered:", nums == sorted(nums), "count", len(nums))
    print("figure captions:", len(re.findall(r"^图 \d", body, re.M)))
    print("table captions:", len(re.findall(r"^表 \d", body, re.M)))
    figs = [int(x) for x in re.findall(r"^图 (\d)", body, re.M)]
    tabs = [int(x) for x in re.findall(r"^表 (\d)", body, re.M)]
    print("fig order:", figs, "tab order:", tabs)
    # in-text references
    refs_f = sorted(set(int(x) for x in re.findall(r"图 (\d)", body)))
    refs_t = sorted(set(int(x) for x in re.findall(r"表 (\d)", body)))
    print("fig referenced:", refs_f, "tab referenced:", refs_t)
