"""Ensure every build script can import ``_paths`` regardless of cwd."""

from __future__ import annotations

from pathlib import Path

TARGETS = {
    "scripts/train_heads_from_cache.py": "from _paths import cache_dir, results_dir, logs_dir",
    "scripts/build_manuscript.py": "from _paths import exports_dir, figures_dir, results_dir",
    "scripts/make_figures.py": "from _paths import cache_dir, exports_dir, figures_dir, results_dir",
    "scripts/crosscheck_numbers.py": "from _paths import results_dir",
    "scripts/build_review_doc.py": "from _paths import build_root, logs_dir, results_dir",
    "scripts/audit_data.py": "from _paths import cache_dir, results_dir",
}

SHIM = (
    "_SCRIPTS = Path(__file__).resolve().parent\n"
    "if str(_SCRIPTS) not in sys.path:\n"
    "    sys.path.insert(0, str(_SCRIPTS))\n"
)

ROOT = Path(__file__).resolve().parents[1]

for rel, imp in TARGETS.items():
    p = ROOT / rel
    text = p.read_text(encoding="utf-8")
    if "_SCRIPTS = Path(__file__)" in text:
        print(f"skip (already shimmed) {rel}")
        continue
    if not any(line.startswith("import sys") for line in text.splitlines()):
        text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport sys\n", 1)
    if imp not in text:
        print(f"  WARN import line not found in {rel}")
        continue
    text = text.replace(imp, SHIM + "\n" + imp, 1)
    p.write_text(text, encoding="utf-8")
    print(f"shimmed {rel}")
