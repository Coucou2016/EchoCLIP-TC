import ast
import re
from pathlib import Path

src = Path("scripts/build_manuscript.py").read_text(encoding="utf-8")
tree = ast.parse(src)
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef) and fn.name.startswith("_m_"):
        for ret in [n for n in ast.walk(fn) if isinstance(n, ast.Return)]:
            if not isinstance(ret.value, ast.List):
                continue
            for el in ret.value.elts:
                if not isinstance(el, ast.Tuple) or len(el.elts) != 2:
                    continue
                para, form = el.elts
                if not (isinstance(form, ast.Tuple) and len(form.elts) == 2):
                    continue
                pv = ast.literal_eval(para) if isinstance(para, ast.Constant) else "<fstring>"
                fv = ast.literal_eval(form.elts[0])
                print(f"--- {fn.name}")
                print("   P:", (pv if isinstance(pv, str) else "<expr>")[:70])
                print("   F:", fv)
