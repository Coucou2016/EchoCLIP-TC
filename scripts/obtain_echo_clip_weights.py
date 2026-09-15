#!/usr/bin/env python3
"""Download official EchoCLIP weights into models/echo-clip/.

Hub id: hf-hub:mkaichristensen/echo-clip
Main file: open_clip_pytorch_model.bin (~606 MB)

If huggingface.co is unreachable (common in some networks), set:
  $env:HF_ENDPOINT = "https://hf-mirror.com"

Usage:
  python scripts/obtain_echo_clip_weights.py
  python scripts/obtain_echo_clip_weights.py --dest models/echo-clip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        default="mkaichristensen/echo-clip",
        help="Hugging Face repo id",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "echo-clip",
        help="Local directory for weights + tokenizer",
    )
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"repo={args.repo}")
    print(f"dest={args.dest}")
    print(f"HF_ENDPOINT={endpoint or '(default https://huggingface.co)'}")
    print(f"token={'set' if token else 'unset (public repo OK)'}")

    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(args.dest),
        token=token,
    )
    bin_path = Path(path) / "open_clip_pytorch_model.bin"
    if not bin_path.is_file():
        print(f"ERROR: missing {bin_path}", file=sys.stderr)
        return 1
    print(f"OK {bin_path} ({bin_path.stat().st_size} bytes)")
    print(
        "Use with:\n"
        f"  python scripts/run_protocol.py --paper --experiments R0 "
        f"--official-checkpoint {bin_path}"
    )
    print("Or let open_clip load hf-hub:mkaichristensen/echo-clip (same files / HF cache).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
