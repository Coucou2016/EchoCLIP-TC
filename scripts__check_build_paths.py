"""Static check that no pipeline script still hard-codes the old F: build tree.

Importing the modules would execute them (make_figures draws, the weight
verifier loads 600 MB), so this walks the source text instead.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

PATTERNS = [
    re.compile(r"F:\\\\EchoCLIP_paper_build"),
    re.compile(r"F:/EchoCLIP_paper_build"),
    re.compile(r"F:\\EchoCLIP_paper_build"),
]

SKIP = {"_paths.py", "_patch_paths.py", "check_build_paths.py"}

# A match only counts as stale when the path is actually *used* (as an argument
# to Path/glob/open or similar). Prose in the audit narrative deliberately
# names the lost directory, and that must not be flagged.
USE_PREFIX = re.compile(
    r"(Path\s*\(\s*r?[\"']|=\s*r?[\"']|glob\s*\(\s*r?[\"']|open\s*\(\s*r?[\"'])"
)

stale: list[tuple[str, int, str]] = []
for p in sorted(SCRIPTS.glob("*.py")):
    if p.name in SKIP:
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if not any(pat.search(line) for pat in PATTERNS):
            continue
        if not USE_PREFIX.search(line):
            continue
        stale.append((p.name, i, line.strip()[:110]))

print(f"scanned {len(list(SCRIPTS.glob('*.py')))} scripts")
if stale:
    print("STALE absolute build paths:")
    for name, line_no, text in stale:
        print(f"  {name}:{line_no}  {text}")
    raise SystemExit(1)
print("no script hard-codes the old F: build tree")

# every script that needs the build root should reference the resolver
needs = ["train_heads_from_cache", "build_manuscript", "make_figures",
         "crosscheck_numbers", "extract_frames_embeddings", "build_review_doc",
         "run_ablation_matrix", "audit_data", "verify_references"]
missing = []
for name in needs:
    src = (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
    if "_paths" not in src:
        missing.append(name)
    else:
        ast.parse(src)  # syntax sanity
print("uses _paths:", "all" if not missing else "missing " + ", ".join(missing))
raise SystemExit(1 if missing else 0)
