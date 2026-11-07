"""One-off: repoint every hard-coded build path at the shared resolver."""

from __future__ import annotations

from pathlib import Path

EDITS = {
    "scripts/train_heads_from_cache.py": [
        (
            'CACHE = Path(r"F:\\EchoCLIP_paper_build\\cache")\n'
            'RESULTS = Path(r"F:\\EchoCLIP_paper_build\\results\\zh")\n'
            'LOGS = Path(r"F:\\EchoCLIP_paper_build\\logs")',
            "from _paths import cache_dir, results_dir, logs_dir\n\n"
            "CACHE = cache_dir()\n"
            "RESULTS = results_dir()\n"
            "LOGS = logs_dir()",
        )
    ],
    "scripts/build_manuscript.py": [
        (
            'RES = Path(r"F:\\EchoCLIP_paper_build\\results\\zh")\n'
            'FIGS = Path(r"F:\\EchoCLIP_paper_build\\figures")',
            "from _paths import results_dir, figures_dir\n\n"
            "RES = results_dir()\n"
            "FIGS = figures_dir()",
        ),
        (
            'EXPORTS = Path(r"F:\\EchoCLIP_paper_build\\exports")',
            "EXPORTS = exports_dir()",
        ),
    ],
    "scripts/make_figures.py": [
        (
            'RES = Path(r"F:\\EchoCLIP_paper_build\\results\\zh")\n'
            'FIGS = Path(r"F:\\EchoCLIP_paper_build\\figures")',
            "from _paths import results_dir, figures_dir\n\n"
            "RES = results_dir()\n"
            "FIGS = figures_dir()",
        ),
        (
            'CACHE = Path(r"F:\\EchoCLIP_paper_build\\cache")',
            "CACHE = cache_dir()",
        ),
    ],
    "scripts/crosscheck_numbers.py": [
        (
            'RES = Path(r"F:\\EchoCLIP_paper_build\\results\\zh")',
            "from _paths import results_dir\n\nRES = results_dir()",
        )
    ],
    "scripts/extract_frames_embeddings.py": [
        (
            'DEFAULT_OUT = Path(r"F:\\EpochCLIP_placeholder")  # replaced below\n'
            'DEFAULT_OUT = Path(r"F:\\EchoCLIP_paper_build\\cache")',
            "DEFAULT_OUT = cache_dir()",
        )
    ],
}

ROOT = Path(__file__).resolve().parents[1]

for rel, subs in EDITS.items():
    p = ROOT / rel
    text = p.read_text(encoding="utf-8")
    for old, new in subs:
        if old not in text:
            print(f"  MISS {rel}: {old.splitlines()[0][:70]!r}")
            continue
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print(f"patched {rel}")

# build_manuscript / make_figures need the symbol imported explicitly
for rel in ("scripts/build_manuscript.py", "scripts/make_figures.py"):
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    if "exports_dir" in t and "from _paths import" in t and "exports_dir," not in t:
        t = t.replace(
            "from _paths import results_dir, figures_dir",
            "from _paths import exports_dir, figures_dir, results_dir",
        )
        p.write_text(t, encoding="utf-8")
        print(f"  import fixed in {rel}")

    if rel.endswith("make_figures.py") and "cache_dir" in p.read_text(encoding="utf-8"):
        t = p.read_text(encoding="utf-8")
        if "cache_dir," not in t and "import cache_dir" not in t:
            t = t.replace(
                "from _paths import exports_dir, figures_dir, results_dir",
                "from _paths import cache_dir, exports_dir, figures_dir, results_dir",
            )
            p.write_text(t, encoding="utf-8")
            print(f"  cache_dir import fixed in {rel}")

print("done")
