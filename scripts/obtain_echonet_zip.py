#!/usr/bin/env python3
"""Unpack or fetch EchoNet-Dynamic into ECHONET_ROOT (license-gated).

EchoNet-Dynamic is Stanford AIMI non-commercial research data (~7 GB zip).
This script does NOT scrape Redivis/AIMI. It only:

  1) Unpacks a local zip you already obtained under your AIMI account, or
  2) Downloads from a URL you explicitly provide (e.g. a personal signed link
     stored in AIMI_DOWNLOAD_URL), or
  3) Prints registration steps if nothing is available.

Usage (PowerShell):
  # After placing the zip somewhere:
  python scripts/obtain_echonet_zip.py --zip E:\\Downloads\\EchoNet-Dynamic.zip

  # Or with a one-time signed URL from AIMI/Redivis (do not share):
  $env:AIMI_DOWNLOAD_URL = "https://..."
  python scripts/obtain_echonet_zip.py --url-env AIMI_DOWNLOAD_URL

  # Then build manifests:
  $env:ECHONET_ROOT = "E:\\Datasets\\EchoNet-Dynamic"
  python scripts/build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

AIMI_PORTAL = "https://stanfordaimi.azurewebsites.net/"
AIMI_REDIVIS = "https://stanford.redivis.com/datasets/66s1-2hsmzj5rn"
PROJECT_PAGE = "https://echonet.github.io/dynamic/"
DEFAULT_DEST = Path(r"E:\Datasets\EchoNet-Dynamic")


def _looks_like_root(root: Path) -> bool:
    if (root / "FileList.csv").is_file() and (root / "Videos").is_dir():
        return True
    nested = root / "EchoNet-Dynamic"
    return (nested / "FileList.csv").is_file() and (nested / "Videos").is_dir()


def _resolve_root(dest: Path) -> Path:
    if (dest / "FileList.csv").is_file():
        return dest
    nested = dest / "EchoNet-Dynamic"
    if (nested / "FileList.csv").is_file():
        return nested
    return dest


def _print_register() -> None:
    print(
        "EchoNet-Dynamic is license-gated (Stanford AIMI Research Use Agreement).\n"
        "This repo cannot download it without your credentials.\n\n"
        f"1. Read: {PROJECT_PAGE}\n"
        f"2. Apply / download (~7 GB zip): {AIMI_PORTAL}\n"
        f"   or Redivis: {AIMI_REDIVIS}\n"
        "3. Place the zip locally, then run:\n"
        "   python scripts/obtain_echonet_zip.py --zip <path-to-EchoNet-Dynamic.zip>\n"
        "4. Or set a personal signed URL (do not share):\n"
        "   $env:AIMI_DOWNLOAD_URL = '<your-url>'\n"
        "   python scripts/obtain_echonet_zip.py --url-env AIMI_DOWNLOAD_URL\n",
        file=sys.stderr,
    )


def download_url(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {zip_path} ...")
    req = Request(url, headers={"User-Agent": "EchoCLIP-obtain/1.0"})
    with urlopen(req, timeout=600) as resp, open(zip_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Saved {zip_path} ({zip_path.stat().st_size} bytes)")


def unpack_zip(zip_path: Path, dest: Path) -> Path:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unpacking {zip_path} -> {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    root = _resolve_root(dest)
    if not _looks_like_root(root) and not _looks_like_root(dest):
        print(
            f"WARNING: expected FileList.csv + Videos/ under {dest} "
            "(or nested EchoNet-Dynamic/). Check zip layout.",
            file=sys.stderr,
        )
    else:
        print(f"OK: EchoNet root appears at {_resolve_root(dest)}")
    return _resolve_root(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Local EchoNet-Dynamic.zip already obtained via AIMI",
    )
    p.add_argument(
        "--url",
        type=str,
        default=None,
        help="Personal signed download URL (do not commit/share)",
    )
    p.add_argument(
        "--url-env",
        type=str,
        default=None,
        help="Env var name holding a signed download URL (e.g. AIMI_DOWNLOAD_URL)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("ECHONET_ROOT", str(DEFAULT_DEST))),
        help="Unpack destination (default: ECHONET_ROOT or E:\\Datasets\\EchoNet-Dynamic)",
    )
    p.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip under dest parent (default keeps --zip path as-is)",
    )
    args = p.parse_args()

    dest: Path = args.dest
    if _looks_like_root(dest) or _looks_like_root(_resolve_root(dest)):
        root = _resolve_root(dest)
        print(f"Already present: {root}")
        print(f"Set: $env:ECHONET_ROOT = '{root}'")
        return 0

    url = args.url
    if args.url_env:
        url = os.environ.get(args.url_env) or url
    if not url:
        url = os.environ.get("AIMI_DOWNLOAD_URL") or os.environ.get("AIMI_TOKEN")
        # AIMI_TOKEN is not used as a URL; ignore non-http values
        if url and not str(url).startswith("http"):
            url = None

    zip_path = args.zip
    if zip_path is None:
        # Common drop locations
        candidates = [
            Path(r"E:\Downloads\EchoNet-Dynamic.zip"),
            Path(r"E:\Datasets\EchoNet-Dynamic.zip"),
            Path.cwd() / "EchoNet-Dynamic.zip",
            Path.cwd() / "data" / "EchoNet-Dynamic.zip",
        ]
        for c in candidates:
            if c.is_file():
                zip_path = c
                print(f"Found local zip: {zip_path}")
                break

    if url and zip_path is None:
        zip_path = dest.parent / "EchoNet-Dynamic.zip"
        try:
            download_url(url, zip_path)
        except Exception as exc:  # noqa: BLE001 — surface network/auth errors clearly
            print(f"Download failed: {exc}", file=sys.stderr)
            _print_register()
            return 2

    if zip_path is None:
        _print_register()
        return 1

    root = unpack_zip(zip_path, dest)
    print(f"\nNext:\n  $env:ECHONET_ROOT = '{root}'\n"
          f"  python scripts/build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
