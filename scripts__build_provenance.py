"""Record the exact code, weights and data revision behind the result JSONs.

The ``git_commit`` field inside each result JSON is whatever HEAD happened to
be while the run executed. During the rebuild the path refactor was still
uncommitted, so that field points at an earlier commit than the one holding
the scripts that actually ran. Rather than rewrite result files after the
fact, this script pins the real provenance by hashing the scripts and
comparing them against the committed tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _paths import results_dir, weights_dir, data_root, manifest_dir  # noqa: E402

PIPELINE = [
    "scripts/_paths.py",
    "scripts/extract_frames_embeddings.py",
    "scripts/train_heads_from_cache.py",
    "scripts/run_ablation_matrix.py",
    "scripts/audit_data.py",
    "scripts/make_figures.py",
    "scripts/build_manuscript.py",
    "scripts/build_review_doc.py",
    "scripts/crosscheck_numbers.py",
    "scripts/verify_references.py",
    "scripts/verify_rebuild_vs_paper.py",
]


def sha256(path: Path, normalise: bool = True) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(_norm(chunk) if normalise else chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _norm(raw: bytes) -> bytes:
    """Normalise line endings so a CRLF checkout can be compared with git blobs."""
    return raw.replace(b"\r\n", b"\n")


def blob_at(ref: str, rel: str) -> str | None:
    """Hash of the file content as stored in the given commit, if present."""
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{rel}"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return hashlib.sha256(_norm(raw)).hexdigest()


def main() -> int:
    head = git("rev-parse", "HEAD")
    stamped = set()
    rst = results_dir()
    for p in sorted(rst.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "git_commit" in d:
            stamped.add(d["git_commit"])

    scripts = {}
    same_as_head = 0
    for rel in PIPELINE:
        p = ROOT / rel
        if not p.exists():
            continue
        local = sha256(p)
        in_head = blob_at(head, rel)
        scripts[rel] = {
            "sha256": local,
            "matches_head": bool(in_head and in_head == local),
        }
        if in_head and in_head == local:
            same_as_head += 1

    w = weights_dir() / "open_clip_pytorch_model.bin"
    fl = data_root() / "FileList.csv"

    out = {
        "head_at_write": head,
        "stamped_in_results": sorted(stamped),
        "note": (
            "结果文件内的 git_commit 字段记录的是该次运行时的 HEAD。重建期间路径改造"
            "尚未提交，因此该字段早于实际运行所用的脚本版本。下面的 sha256 是逐脚本"
            "实测值，matches_head 表示与当前提交中的内容逐字节一致，可据此复核代码版本。"
        ),
        "scripts": scripts,
        "scripts_matching_head": f"{same_as_head}/{len(scripts)}",
        "weights": {
            "file": str(w),
            "sha256": sha256(w, normalise=False) if w.exists() else "missing",
            "bytes": w.stat().st_size if w.exists() else 0,
        },
        "data": {
            "filelist": str(fl),
            "sha256": sha256(fl, normalise=False) if fl.exists() else "missing",
            "manifest_sha256": {
                p.name: sha256(p, normalise=False)
                for p in sorted(manifest_dir().glob("*.json"))
            },
        },
        "torch": _torch(),
        "python": sys.version.split()[0],
    }

    dst = rst / "provenance.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst}")
    print(f"head {head[:12]}, scripts matching head {same_as_head}/{len(scripts)}")
    print(f"weights {out['weights']['sha256'][:16]}...")
    return 0


def _torch() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
