#!/usr/bin/env python3
"""P0-7: repository integrity audit.

Fails (non-zero exit) when:

  * a path in :data:`REQUIRED_PATHS` is missing, or
  * a repo-relative path referenced in ``README.md`` / ``DATA.md`` does not exist.

Referenced paths are collected from:
  * backticked inline code spans (e.g. `` `scripts/run_protocol.py` ``), and
  * fenced project-layout listings (indentation tracked as directory context).

Genuinely-planned-but-absent files must be listed under a heading containing
``Planned`` (the audit exempts those) rather than in a layout listing.

This audits the **tree-layout** repository only. It is not aware of, and must
not be pointed at, any separate flattened audit mirror.

Usage
-----
::

    python scripts/audit_repo_integrity.py            # human report, exit 0/1
    python scripts/audit_repo_integrity.py --json     # machine-readable verdict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Paths that must exist for the documented workflows to run.
REQUIRED_PATHS: Tuple[str, ...] = (
    # Packaging / metadata
    "README.md",
    "DATA.md",
    "PAPER.md",
    "LICENSE",
    "NOTICE",
    "ATTRIBUTION.md",
    "PROVENANCE.md",
    "requirements.txt",
    "requirements-minimal.txt",
    "docs/OBTAIN_DATA_AND_WEIGHTS.md",
    # Core package
    "echoclip/__init__.py",
    "echoclip/model.py",
    "echoclip/config.py",
    "echoclip/data.py",
    "echoclip/protocol.py",
    "echoclip/clinical.py",
    "echoclip/checkpoint.py",
    "echoclip/supervised.py",
    "echoclip/supervised_checkpoint.py",
    "echoclip/temporal.py",
    "echoclip/calibrate.py",
    "echoclip/loss.py",
    "echoclip/zeroshot.py",
    "echoclip/prompts.py",
    "echoclip/official_parity.py",
    "echoclip/official_r0.py",
    "echoclip/config_io.py",
    # Scripts referenced by the documented protocol
    "scripts/run_protocol.py",
    "scripts/run_seeds.py",
    "scripts/train.py",
    "scripts/train_supervised.py",
    "scripts/eval_clinical.py",
    "scripts/eval_official_r0.py",
    "scripts/build_echonet_manifest.py",
    "scripts/make_demo_data.py",
    "scripts/validate.py",
    "scripts/audit_repo_integrity.py",
    # Configs
    "configs/default.yaml",
    "configs/echonet_dynamic.yaml",
    # Tests
    "tests/test_protocol.py",
    "tests/test_p1_protocol.py",
    "tests/test_split_integrity.py",
    "tests/test_prediction_mode.py",
)

#: Filenames whose inline/backticked repo paths are audited.
DOC_FILES: Tuple[str, ...] = ("README.md", "DATA.md")

#: Sections whose listed paths are explicitly *planned* and may be absent.
_PLANNED_RE = re.compile(r"^\s*#{1,6}\s*.*planned|^\s*planned\s*:", re.IGNORECASE)

_KNOWN_EXTS = (
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".txt",
    ".sh",
    ".ps1",
    ".cff",
    ".lock",
    ".ini",
)
# Inline code spans only (never cross a line boundary — avoids pairing a lone
# backtick with one several lines away).
_PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`")

#: Top-level repo directories whose referenced paths we audit. Anything else
#: (owner/repo, ``hf-hub:...``, ``.venv/...``, package names) is ignored.
AUDITED_PREFIXES = frozenset(
    {
        ".github",
        "checkpoints",
        "configs",
        "data",
        "docs",
        "echoclip",
        "figures",
        "LICENSES",
        "models",
        "notes",
        "papers",
        "reports",
        "scripts",
        "tests",
    }
)

#: Known root-level files that are always audited when referenced.
AUDITED_ROOT_FILES = frozenset(
    {
        "ATTRIBUTION.md",
        "CITATION.cff",
        "DATA.md",
        "LICENSE",
        "NOTICE",
        "PAPER.md",
        "PROVENANCE.md",
        "README.md",
        "pytest.ini",
        "requirements-lock.txt",
        "requirements-minimal.txt",
        "requirements-paper.lock",
        "requirements.txt",
    }
)

_SKIP_PREFIXES = (
    "python",
    "pip",
    "git",
    "cd ",
    "conda",
    "set ",
    "$",
    "#",
    "<",
    ">",
    "--",
    "export",
    "make ",
    "http",
    "https",
)


def _looks_like_repo_path(tok: str) -> bool:
    t = tok.strip().strip("'\"")
    if not t or " " in t:
        return False
    if any(ch in t for ch in ("*", "?", "{", "}", "$", "(", ")", ",", ";", "|", "=")):
        return False
    if t.lower().startswith(_SKIP_PREFIXES):
        return False
    if t.startswith("/") or t.startswith("./") or "://" in t or ":" in t:
        return False

    path = Path(t.rstrip("/"))
    parts = path.parts
    if not parts:
        return False

    if len(parts) == 1:
        # A bare filename must look like a real file we know about.
        return t in AUDITED_ROOT_FILES or t.lower().endswith(_KNOWN_EXTS) and t in AUDITED_ROOT_FILES
    # Multi-segment: only audit under a known top-level directory.
    return parts[0] in AUDITED_PREFIXES


def _normalize(tok: str) -> str:
    t = tok.strip().strip("'\"").replace("\\", "/")
    return t.rstrip("/")


def extract_inline_paths(text: str) -> Set[str]:
    """Backticked repo-relative paths referenced inline in the doc."""
    out: Set[str] = set()
    for tok in _PATH_TOKEN_RE.findall(text):
        if _looks_like_repo_path(tok):
            out.add(_normalize(tok))
    return out


def extract_layout_paths(text: str) -> Tuple[Set[str], Set[str]]:
    """Paths from fenced layout blocks, plus paths under a ``Planned`` heading.

    Returns ``(layout_paths, planned_paths)``. Directory context is tracked by
    zero-indentation entries ending in ``/``.
    """
    layout: Set[str] = set()
    planned: Set[str] = set()
    in_fence = False
    in_planned = False
    context = ""

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            context = ""
            continue
        if not in_fence:
            if _PLANNED_RE.match(raw):
                in_planned = True
            elif stripped.startswith("#"):
                in_planned = False
            continue

        if not stripped or stripped.startswith("#"):
            continue
        candidate = stripped.split("#", 1)[0].strip()
        if not candidate:
            continue

        is_dir_entry = candidate.endswith("/")
        normalized = _normalize(candidate)
        if not normalized or " " in normalized:
            continue

        indented = raw[:1].isspace()
        if indented and context:
            rel = f"{context}/{normalized}"
        else:
            rel = normalized
            if is_dir_entry and not indented:
                context = normalized
            elif indented:
                context = ""

        if not (_looks_like_repo_path(rel) or is_dir_entry):
            continue
        if is_dir_entry and rel not in AUDITED_PREFIXES:
            # Only track directory context we actually audit.
            context = rel
            continue

        (planned if in_planned else layout).add(rel)
    return layout, planned


def collect_referenced_paths(
    root: Path,
    doc_files: Tuple[str, ...] = DOC_FILES,
) -> Dict[str, Dict[str, Set[str]]]:
    refs: Dict[str, Dict[str, Set[str]]] = {}
    for name in doc_files:
        path = root / name
        if not path.is_file():
            refs[name] = {"inline": set(), "layout": set(), "planned": set()}
            continue
        text = path.read_text(encoding="utf-8")
        layout, planned = extract_layout_paths(text)
        refs[name] = {
            "inline": extract_inline_paths(text),
            "layout": layout,
            "planned": planned,
        }
    return refs


def audit(root: Path = ROOT) -> Dict[str, object]:
    """Return a structured verdict dict (never raises)."""
    missing_required = [
        p for p in REQUIRED_PATHS if not (root / p).exists()
    ]

    refs = collect_referenced_paths(root)
    missing_refs: Dict[str, List[str]] = {}
    for name, buckets in refs.items():
        planned = buckets["planned"]
        referenced = set(buckets["inline"]) | set(buckets["layout"])
        missing = sorted(
            p for p in referenced if p not in planned and not (root / p).exists()
        )
        if missing:
            missing_refs[name] = missing

    ok = not missing_required and not missing_refs
    return {
        "ok": bool(ok),
        "root": str(root),
        "required_paths_n": len(REQUIRED_PATHS),
        "missing_required": missing_required,
        "missing_referenced": missing_refs,
        "referenced_counts": {
            name: {
                "inline": len(b["inline"]),
                "layout": len(b["layout"]),
                "planned_exempt": len(b["planned"]),
            }
            for name, b in refs.items()
        },
    }


def _print_report(verdict: Dict[str, object]) -> None:
    print("Repository integrity audit (tree layout)")
    print(f"  root: {verdict['root']}")
    print(f"  required paths checked: {verdict['required_paths_n']}")
    for name, counts in verdict["referenced_counts"].items():  # type: ignore[union-attr]
        print(
            f"  {name}: {counts['inline']} inline + {counts['layout']} layout refs "
            f"({counts['planned_exempt']} planned-exempt)"
        )

    missing_required = verdict["missing_required"]  # type: ignore[assignment]
    if missing_required:
        print("\nMISSING REQUIRED PATHS:")
        for p in missing_required:  # type: ignore[union-attr]
            print(f"  - {p}")

    missing_refs = verdict["missing_referenced"]  # type: ignore[assignment]
    if missing_refs:
        print("\nBROKEN DOC REFERENCES (path listed but does not exist):")
        for doc, paths in missing_refs.items():  # type: ignore[union-attr]
            for p in paths:  # type: ignore[union-attr]
                print(f"  - {doc}: {p}")
        print(
            "\nFix: create the file, correct the reference, or move it under a "
            "`Planned:` heading if it is intentionally not yet present."
        )

    print("\nRESULT:", "OK" if verdict["ok"] else "FAILED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Emit JSON verdict")
    args = parser.parse_args()

    verdict = audit(args.root)
    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        _print_report(verdict)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
