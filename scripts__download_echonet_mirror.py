#!/usr/bin/env python3
"""Download EchoNet-Dynamic from a Kaggle / Hugging Face *mirror*.

IMPORTANT (read before using)
-----------------------------
EchoNet-Dynamic is distributed by Stanford AIMI under a Research Use Agreement
that forbids redistribution. Every Kaggle/HF copy of the real dataset is an
*unauthorized* third-party mirror (many are mislabelled "MIT"/"Apache-2.0").
Downloading from them means you are NOT getting the data through the AIMI
agreement, and you must NOT describe the data as "obtained under the Stanford
AIMI RUA" in DATA.md / PAPER.md. The upstream-provided legal route is:

    https://echonet.github.io/dynamic/  ->  https://stanford.redivis.com/datasets/66s1-2hsmzj5rn

This script exists because the operator explicitly asked for the mirror route.
It never uploads, reshards, or republishes anything.

Sources
-------
kaggle : needs a Kaggle API token (~/.kaggle/kaggle.json, or KAGGLE_USERNAME +
         KAGGLE_KEY / KAGGLE_API_TOKEN). Anonymous download returns 404 + captcha.
         Candidate mirrors (all ~7.9 GB, real videos, MIT/Apache/CC0 mislabels):
             snikhilrao/echonet-dynamic-data
             mahnurrahman/echonet-dynamic
             nazmulxdxd/echonet-dynamic-full-dataset-curated
hf     : needs a HF token for gated repos. AISport/EchoNetDataset is `gated: manual`
         (~1.57 GB, 2000 videos + FileList.csv). Public-but-useless repos such as
         zeahub/echonet-dynamic only hold frame-index YAMLs, not videos.
         Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable.

Layout produced:  <dest>/FileList.csv, <dest>/Videos/*.avi   (what
`scripts/build_echonet_manifest.py --echonet-root <dest>` expects).

Usage
-----
  # show what would happen, without touching the network
  python scripts/download_echonet_mirror.py --source kaggle --dry-run

  # Kaggle mirror (needs token)
  python scripts/download_echonet_mirror.py --source kaggle \
      --dataset-ref snikhilrao/echonet-dynamic-data

  # HF gated mirror (needs token + accepted terms on the repo page)
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python scripts/download_echonet_mirror.py --source hf \
      --repo AISport/EchoNetDataset --hf-token $env:HF_TOKEN

Then:
  $env:ECHONET_ROOT = "E:\\Datasets\\EchoNet-Dynamic"
  python scripts/build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT \
      --subset-5000 --test-subset 500 --allow-missing-videos
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DEST = Path(r"E:\Datasets\EchoNet-Dynamic")

KAGGLE_MIRRORS = {
    "snikhilrao/echonet-dynamic-data": "MIT (mislabel) - 7.9 GB, full videos",
    "mahnurrahman/echonet-dynamic": "Apache-2.0 (mislabel) - 7.9 GB, full videos",
    "nazmulxdxd/echonet-dynamic-full-dataset-curated": "Other - 7.9 GB, full videos",
    "kartikraikar012/echonet": "MIT (mislabel) - 7.9 GB, full videos",
}

HF_MIRRORS = {
    "AISport/EchoNetDataset": "MIT (mislabel) - gated, ~1.57 GB, 2000 videos",
    "HReynaud/EchoNet-Synthetic-v1": "apache-2.0 - SYNTHETIC videos, not real patients",
}

LICENSE_WARNING = """\
[!] This is a third-party mirror of an AIMI research-use dataset.
    - Do NOT redistribute what you download.
    - Do NOT claim AIMI/RUA authorization for these files in DATA.md / PAPER.md.
    - Prefer registering at https://stanford.redivis.com/datasets/66s1-2hsmzj5rn
      so the dataset is citable and lawfully obtained.
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def looks_like_root(root: Path) -> bool:
    if (root / "FileList.csv").is_file() and (root / "Videos").is_dir():
        return True
    nested = root / "EchoNet-Dynamic"
    return (nested / "FileList.csv").is_file() and (nested / "Videos").is_dir()


def resolve_root(dest: Path) -> Path:
    if (dest / "FileList.csv").is_file():
        return dest
    nested = dest / "EchoNet-Dynamic"
    if (nested / "FileList.csv").is_file():
        return nested
    if (dest / "EchoNetDataset" / "FileList.csv").is_file():
        return dest / "EchoNetDataset"
    return dest


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    for p in (
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
    ):
        if p.is_file():
            return True
    return False


def hf_token_present(cli_token: str | None) -> bool:
    if cli_token:
        return True
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    for p in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if p.is_file():
            return True
    return False


def download_kaggle(ref: str, dest: Path, dry_run: bool, keep_zip: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    _log(f"[kaggle] ref={ref}")
    if not kaggle_credentials_present():
        _log(
            "[kaggle] no credentials found.\n"
            "  1. https://www.kaggle.com/settings/account -> Create New API Token\n"
            "  2. save it to %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "  3. re-run this command\n"
            "  (kaggle.json is a secret: never commit it into the repo)"
        )
        if dry_run:
            _log("[kaggle] would run: kaggle datasets download -d %s -p %s --unzip" % (ref, dest))
        return 3

    # Preferred: kagglehub (resumable, cache-managed).
    try:
        import kagglehub  # type: ignore

        if dry_run:
            _log(f"[kaggle] would run: kagglehub.dataset_download('{ref}')")
            return 0
        _log("[kaggle] kagglehub.dataset_download ...")
        got = Path(kagglehub.dataset_download(ref))
        _log(f"[kaggle] cached at {got}")
        if got != dest and not looks_like_root(dest):
            return _link_or_copy(got, dest)
        return 0
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        _log(f"[kaggle] kagglehub failed: {exc}")

    if dry_run:
        _log(f"[kaggle] would run: kaggle datasets download -d {ref} -p {dest} --unzip")
        return 0

    exe = shutil.which("kaggle")
    if not exe:
        _log("[kaggle] `kaggle` CLI not found. Install with: python -m pip install kaggle")
        return 4
    cmd = [exe, "datasets", "download", "-d", ref, "-p", str(dest), "--unzip"]
    _log("[kaggle] " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        _log(f"[kaggle] CLI exited {rc}. Delete the partial cache and retry (network flaky).")
    return rc


def download_hf(repo: str, dest: Path, dry_run: bool, token: str | None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    _log(f"[hf] repo={repo} endpoint={endpoint}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log("[hf] huggingface_hub missing: python -m pip install huggingface_hub")
        return 4

    if dry_run:
        _log(
            f"[hf] would run: snapshot_download('{repo}', repo_type='dataset', "
            f"local_dir=r'{dest}')  # endpoint={endpoint}"
        )
        return 0

    if not hf_token_present(token):
        _log(
            "[hf] no token found and this repo is likely gated.\n"
            "  1. accept the terms on https://huggingface.co/datasets/"
            f"{repo}\n"
            "  2. create a read token at https://huggingface.co/settings/tokens\n"
            "  3. retry with --hf-token <token> or set $env:HF_TOKEN"
        )
        return 3

    try:
        got = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(dest),
            token=token,
            max_workers=4,
        )
        _log(f"[hf] downloaded to {got}")
        return 0
    except Exception as exc:  # noqa: BLE001
        _log(f"[hf] snapshot_download failed: {exc}")
        _log("[hf] if this is a 401/403 the repo is gated or your token lacks access.")
        return 5


def _link_or_copy(src: Path, dest: Path) -> int:
    """kagglehub keeps files in its own cache; mirror the layout into dest."""
    files = [p for p in src.rglob("*") if p.is_file()]
    _log(f"[kaggle] relocating {len(files)} files from {src} -> {dest}")
    for f in files:
        rel = f.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            try:
                os.link(f, out)  # same-volume hardlink, free on NTFS
            except OSError:
                shutil.copy2(f, out)
    return 0


def verify(dest: Path) -> int:
    root = resolve_root(dest)
    _log("\n[verify] checking layout ...")
    if not looks_like_root(root):
        _log(f"[verify] FAIL: no FileList.csv + Videos/ under {dest} (resolved {root})")
        _log("[verify] inspect the mirror's internal folder names; some mirrors nest deeper.")
        return 1

    videos = list((root / "Videos").glob("*.avi"))
    total = sum(v.stat().st_size for v in videos)
    _log(f"[verify] OK root      : {root}")
    _log(f"[verify] FileList.csv : {(root / 'FileList.csv').stat().st_size} bytes")
    _log(f"[verify] Videos/*.avi : {len(videos)} files, {total / 2**30:.2f} GiB")
    for extra in ("VolumeTracings.csv", "VolumeTracings2.csv"):
        p = root / extra
        _log(f"[verify] {extra:<20}: {'present' if p.is_file() else 'absent'}")
    _log(
        "\n[next]\n"
        f'  $env:ECHONET_ROOT = "{root}"\n'
        "  python scripts/build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT "
        "--subset-5000 --test-subset 500 --allow-missing-videos"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download EchoNet-Dynamic from a Kaggle/HF mirror (see module docstring for caveats).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", choices=["kaggle", "hf"], default=None)
    p.add_argument("--dataset-ref", default=None, help="Kaggle dataset ref, e.g. snikhilrao/echonet-dynamic-data")
    p.add_argument("--repo", default=None, help="HF dataset repo id, e.g. AISport/EchoNetDataset")
    p.add_argument("--hf-token", default=None, help="HF read token (else $HF_TOKEN / cached token)")
    p.add_argument("--dest", type=Path, default=Path(os.environ.get("ECHONET_ROOT", str(DEFAULT_DEST))))
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded archive (default: cache only)")
    p.add_argument("--dry-run", action="store_true", help="Print the plan; do not download")
    p.add_argument("--list-mirrors", action="store_true", help="Show known mirrors and exit")
    args = p.parse_args()

    if args.list_mirrors:
        _log("Kaggle mirrors (need kaggle.json):")
        for k, v in KAGGLE_MIRRORS.items():
            _log(f"  {k:<52} {v}")
        _log("HF mirrors (need HF token for gated):")
        for k, v in HF_MIRRORS.items():
            _log(f"  {k:<52} {v}")
        return 0

    if args.source is None:
        p.error("--source is required unless --list-mirrors is used")

    _log(LICENSE_WARNING)

    if looks_like_root(resolve_root(args.dest)) and not args.dry_run:
        _log(f"[skip] already present: {resolve_root(args.dest)}")
        return verify(args.dest)

    if args.source == "kaggle":
        ref = args.dataset_ref or "snikhilrao/echonet-dynamic-data"
        if ref not in KAGGLE_MIRRORS:
            _log(f"[warn] {ref} is not in the known-mirror list; proceeding anyway.")
        rc = download_kaggle(ref, args.dest, args.dry_run, args.keep_zip)
    else:
        repo = args.repo or "AISport/EchoNetDataset"
        if repo not in HF_MIRRORS:
            _log(f"[warn] {repo} is not in the known-mirror list; proceeding anyway.")
        rc = download_hf(repo, args.dest, args.dry_run, args.hf_token)

    if rc != 0 or args.dry_run:
        return rc
    return verify(args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
