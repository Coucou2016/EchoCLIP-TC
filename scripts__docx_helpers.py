"""Build the Chinese manuscript as a Word document with a live TOC field.

Everything numeric is read from results/zh/*.json produced in this session. No
figure or table value is typed by hand: the builder aborts if a required JSON is
missing, so a number can never appear without a source file behind it.

Outputs
  E:\\Projects\\20260522-EchoCLIP\\papers\\EchoCLIP-TA_中文稿.docx
  <build root>\\exports\\EchoCLIP-TA_中文稿.docx (copy)

Layout rules implemented
  - body: SimSun 12 pt (小四), 1.5 line spacing, first-line indent 2 chars
  - headings: SimHei, numbered 1 / 1.1 / 1.1.1 through real Heading 1-3 styles
  - western text: Times New Roman
  - table captions above, figure captions below, three-line tables
  - equations as text with right-aligned (1) (2) ... numbering
  - TOC field ``TOC \\o "1-3" \\h \\z \\u`` plus <w:updateFields w:val="true"/>
  - page number field in the footer
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import cache_dir, exports_dir, figures_dir, results_dir  # noqa: E402

RES = results_dir()
FIGS = figures_dir()
PAPERS = ROOT / "papers"
EXPORTS = exports_dir()
CACHE = cache_dir()

BODY_FONT_CN = "宋体"
BODY_FONT_EN = "Times New Roman"
HEAD_FONT_CN = "黑体"


# --------------------------------------------------------------------------
# data loading helpers
# --------------------------------------------------------------------------
def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise FileNotFoundError(f"required result missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def gather(exp: str, frames: int = 32, tag: str = "", pool: str = "cls") -> List[dict]:
    pat = f"{exp}_s*_f{frames}_{pool}"
    if tag:
        pat += f"_{tag}"
    out = []
    for p in sorted(RES.glob(pat + "*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def mean_sd(vals: List[float]) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def fmt(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# docx low-level helpers
# --------------------------------------------------------------------------
def set_run_font(run, size: float = 12, bold: bool = False, cn: str = BODY_FONT_CN, en: str = BODY_FONT_EN):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:cs"), en)


def style_font(style, cn: str, en: str, size: float, bold: bool = False):
    style.font.name = en
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def add_body(doc, text: str, indent: bool = True, align=None, size: float = 12, bold: bool = False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_heading(doc, text: str, level: int):
    h = doc.add_heading(level=level)
    h.paragraph_format.space_before = Pt(12 if level == 1 else 9)
    h.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    h.paragraph_format.line_spacing = 1.5
    run = h.add_run(text)
    size = {1: 15, 2: 13, 3: 12}.get(level, 12)
    set_run_font(run, size=size, bold=True, cn=HEAD_FONT_CN, en=BODY_FONT_EN)
    return h


def add_equation(doc, text: str, number: int):
    """Centred equation text with a right-aligned tab-stop number."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(3)
    pf.space_after = Pt(3)
    # centre tab and right tab
    tabs = pf.tab_stops
    tabs.add_tab_stop(Cm(7.75), alignment=1)  # centre
    tabs.add_tab_stop(Cm(15.5), alignment=2)  # right
    run = p.add_run("\t" + text + "\t(" + str(number) + ")")
    set_run_font(run, size=11)
    run.font.italic = False
    return p


def add_caption(doc, text: str, above: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4 if above else 2)
    pf.space_after = Pt(6 if above else 10)
    pf.line_spacing = 1.15
    run = p.add_run(text)
    set_run_font(run, size=10)


def _set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tcpr = tc.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        if edge in kwargs:
            spec = kwargs[edge]
            el = borders.find(qn(f"w:{edge}"))
            if el is None:
                el = OxmlElement(f"w:{edge}")
                borders.append(el)
            for k, v in spec.items():
                el.set(qn(f"w:{k}"), str(v))


def add_three_line_table(doc, header: List[str], rows: List[List[str]], caption: str,
                         col_widths: Optional[List[float]] = None, font_size: float = 9,
                         note: str = "") -> None:
    add_caption(doc, caption, above=True)
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    hdr = table.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h)
        set_run_font(run, size=font_size, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(str(val))
            set_run_font(run, size=font_size)
    # three-line style: thick top, thin below header, thick bottom
    for cell in table.rows[0].cells:
        _set_cell_border(cell, top={"val": "single", "sz": 12, "color": "000000"},
                         bottom={"val": "single", "sz": 8, "color": "000000"},
                         left={"val": "nil"}, right={"val": "nil"})
    for row in table.rows[1:]:
        for cell in row.cells:
            _set_cell_border(cell, left={"val": "nil"}, right={"val": "nil"},
                             top={"val": "nil"}, bottom={"val": "nil"})
    for cell in table.rows[-1].cells:
        _set_cell_border(cell, bottom={"val": "single", "sz": 12, "color": "000000"})
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(note)
        set_run_font(run, size=8.5)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_picture(doc, stem: str, caption: str, width_cm: float = 15.5) -> bool:
    path = FIGS / f"{stem}.png"
    if not path.exists():
        print(f"  WARNING figure missing: {path}", flush=True)
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption, above=False)
    return True


def add_toc(doc) -> None:
    p = doc.add_paragraph()
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t")
    t.text = "右键此处选择“更新域”以生成目录与页码。"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(t)
    run._r.append(fld_end)


def add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)


def enable_update_fields(doc) -> None:
    settings = doc.settings.element
    el = settings.find(qn("w:updateFields"))
    if el is None:
        el = OxmlElement("w:updateFields")
        settings.append(el)
    el.set(qn("w:val"), "true")


def setup_styles(doc) -> None:
    normal = doc.styles["Normal"]
    style_font(normal, BODY_FONT_CN, BODY_FONT_EN, 12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    for lvl, size in ((1, 15), (2, 13), (3, 12)):
        st = doc.styles[f"Heading {lvl}"]
        style_font(st, HEAD_FONT_CN, BODY_FONT_EN, size, bold=True)
        st.paragraph_format.space_before = Pt(12 if lvl == 1 else 9)
        st.paragraph_format.space_after = Pt(6 if lvl == 1 else 4)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.keep_with_next = True
