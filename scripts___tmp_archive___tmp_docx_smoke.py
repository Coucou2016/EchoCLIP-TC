"""Smoke test the docx helpers without needing real results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docx import Document
from docx.shared import Cm
from docx_helpers import (
    add_body, add_caption, add_equation, add_heading, add_page_number_footer,
    add_three_line_table, add_toc, enable_update_fields, setup_styles,
)

doc = Document()
setup_styles(doc)
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
add_page_number_footer(sec)
add_heading(doc, "目录", 1)
add_toc(doc)
doc.add_page_break()
add_heading(doc, "1 测试", 1)
add_body(doc, "这是一个测试段落，包含中文与 Times New Roman 混排。", indent=True)
add_equation(doc, "EF = (EDV − ESV) / EDV × 100%", 1)
add_three_line_table(doc, ["编号", "MAE"], [["R2", "4.20"], ["R3", "4.10"]], "表 1  测试三线表。")
add_caption(doc, "图 1  测试图注。", above=False)
enable_update_fields(doc)
out = Path(r"F:\EchoCLIP_paper_build\_smoke_docx.docx")
doc.save(out)
print("saved", out, out.stat().st_size)
d2 = Document(str(out))
print("paras", len(d2.paragraphs), "tables", len(d2.tables))
