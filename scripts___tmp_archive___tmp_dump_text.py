import sys
from pathlib import Path
from docx import Document

p = Path(r"E:\Projects\20260522-EchoCLIP\papers\EchoCLIP-TA_中文稿.docx")
doc = Document(str(p))
out = []
for para in doc.paragraphs:
    t = para.text.strip()
    if not t:
        continue
    st = para.style.name
    if st.startswith("Heading"):
        out.append(f"\n[{st}] {t}")
    else:
        out.append(t)
Path(r"F:\EchoCLIP_paper_build\_manuscript_text.txt").write_text("\n".join(out), encoding="utf-8")
print(len(out), "paragraphs written")
print("chars", sum(len(x) for x in out))
