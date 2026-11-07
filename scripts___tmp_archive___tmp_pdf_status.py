import json
from pathlib import Path

p = Path(r"F:\EchoCLIP_paper_build\results\zh\references_verified.json")
refs = json.loads(p.read_text(encoding="utf-8"))
for r in refs:
    pdf = r.get("pdf") or {}
    print(f"{r['key'][:30]:32s} oa={str(r.get('oa')):5s} "
          f"pdf_ok={str(pdf.get('ok')):5s} magic={pdf.get('magic')} "
          f"file={Path(str(pdf.get('path', ''))).name}")
print("total", len(refs), "pdf ok", sum(1 for r in refs if (r.get('pdf') or {}).get('ok')))
