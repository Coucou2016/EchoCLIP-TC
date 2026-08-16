"""Write a B0/M1/M2/M4 comparison table from protocol metrics.json files.

Reads ``checkpoints/protocol/<ID>/metrics.json`` and writes:

- ``checkpoints/protocol/comparison.json``
- ``checkpoints/protocol/comparison.md``

Demo / scratch_fallback numbers must never be reported as clinical EF MAE.

Examples
--------
::

  python scripts/write_protocol_table.py
  python scripts/write_protocol_table.py --protocol-root checkpoints/protocol
  python scripts/write_protocol_table.py --experiments B0,M1,M2,M4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.protocol import (  # noqa: E402
    EXPERIMENT_IDS,
    get_experiment,
    write_protocol_comparison,
)


def _parse_ids(raw: str | None) -> list[str]:
    if not raw or raw.strip().lower() in ("all", "*"):
        return list(EXPERIMENT_IDS)
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(get_experiment(part).id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate protocol metrics.json into a comparison table"
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "checkpoints" / "protocol",
        help="Directory containing B0/M1/M2/M4 subfolders",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated IDs or 'all'",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print comparison.md to stdout",
    )
    args = parser.parse_args()

    ids = _parse_ids(args.experiments)
    root = args.protocol_root
    if not root.exists():
        print(f"Protocol root not found: {root}")
        print("Run experiments first: python scripts/run_protocol.py --experiments B0,M1")
        return 1

    paths = write_protocol_comparison(root, exp_ids=ids)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['md']}")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    print(
        f"Rows: {payload['n_experiments']} "
        f"({', '.join(payload['experiment_ids']) or 'none'})"
    )
    if payload.get("any_demo"):
        print(
            "WARNING: at least one row is demo_is_not_clinical — "
            "do not report as EchoNet / paper EF MAE."
        )
    if args.do_print:
        print()
        print(paths["md"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
