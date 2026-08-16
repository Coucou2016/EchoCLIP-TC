"""Run project validation: manifest, imports, smoke forward, optional eval."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EchoCLIP installation and data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "demo" / "manifest.json")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    failed = []

    print("== 1/4 Smoke test ==")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py")], cwd=ROOT)
    if r.returncode != 0:
        failed.append("smoke_test")

    print("\n== 2/4 Manifest validation ==")
    from echoclip.data import load_manifest, validate_manifest

    if not args.manifest.exists():
        print(f"  manifest missing: {args.manifest}")
        print("  run: python scripts/make_demo_data.py")
        failed.append("manifest")
    else:
        pairs = load_manifest(args.manifest)
        errors = validate_manifest(pairs, args.manifest_dir)
        if errors:
            print(f"  {len(errors)} error(s):")
            for e in errors[:10]:
                print(f"    - {e}")
            failed.append("manifest_errors")
        else:
            print(f"  OK: {len(pairs)} pairs, files exist")

    print("\n== 3/4 Unit tests ==")
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        failed.append("unittest")

    if not args.skip_eval and args.checkpoint.exists():
        print("\n== 4/4 Checkpoint eval ==")
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "eval.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--manifest",
                str(args.manifest),
                "--manifest-dir",
                str(args.manifest_dir),
                "--skip-zeroshot",
            ],
            cwd=ROOT,
        )
        if r.returncode != 0:
            failed.append("eval")
    else:
        print("\n== 4/4 Checkpoint eval (skipped) ==")
        if not args.checkpoint.exists():
            print(f"  no checkpoint at {args.checkpoint}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nAll validation steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
