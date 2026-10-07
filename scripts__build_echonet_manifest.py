"""Build a DATA.md-compatible JSON manifest from EchoNet-Dynamic.

EchoNet-Dynamic is distributed by Stanford AIMI under a **non-commercial**
research license. This script does not download the videos.

Expected layout (any one of these roots):
  <root>/FileList.csv
  <root>/Videos/*.avi
  <root>/VolumeTracings.csv   (optional; used for ED/ES frame indices)

Download:
  https://echonet.github.io/dynamic/
  Stanford AIMI (request access, non-commercial): https://stanfordaimi.azurewebsites.net/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.structured_text import pair_record
from echoclip.utils import set_seed

ECHONET_DOWNLOAD_HELP = """
EchoNet-Dynamic files were not found.

This dataset is NOT bundled with the repo. Request it from Stanford AIMI
(non-commercial research license):

  1. https://echonet.github.io/dynamic/
  2. https://stanfordaimi.azurewebsites.net/

Place at least FileList.csv and the Videos/ directory under --echonet-root, e.g.

  <root>/FileList.csv
  <root>/Videos/0X....avi
  <root>/VolumeTracings.csv   (optional, ED/ES indices)

Then re-run:

  python scripts/build_echonet_manifest.py --echonet-root <root>
""".strip()


def _find_filelist(root: Path) -> Optional[Path]:
    candidates = [
        root / "FileList.csv",
        root / "filelist.csv",
        root / "EchoNet-Dynamic" / "FileList.csv",
        root / "echonet-dynamic" / "FileList.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_videos_dir(root: Path, filelist: Path) -> Optional[Path]:
    candidates = [
        root / "Videos",
        root / "videos",
        filelist.parent / "Videos",
        filelist.parent / "videos",
        root / "EchoNet-Dynamic" / "Videos",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _find_tracings(root: Path, filelist: Path) -> Optional[Path]:
    candidates = [
        root / "VolumeTracings.csv",
        filelist.parent / "VolumeTracings.csv",
        root / "EchoNet-Dynamic" / "VolumeTracings.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _stem(name: str) -> str:
    p = Path(str(name).strip())
    return p.stem if p.suffix.lower() in {".avi", ".mp4", ".mov"} else p.name


def _video_filename(name: str) -> str:
    raw = str(name).strip()
    if Path(raw).suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
        return Path(raw).name
    return f"{_stem(raw)}.avi"


def _float(row: dict, *keys: str) -> Optional[float]:
    for key in keys:
        if key in row and str(row[key]).strip() not in ("", "NA", "nan", "None"):
            try:
                return float(row[key])
            except ValueError:
                continue
    return None


def _segment_length(x1, y1, x2, y2) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


def ed_es_from_tracings(tracing_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Unique frames per video; larger tracing-length proxy = ED, smaller = ES.

    EchoNet VolumeTracings rows are LV diameter segments at ED and ES.
    """
    by_file_frame: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with tracing_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fname = _stem(row.get("FileName") or row.get("Filename") or "")
            if not fname:
                continue
            try:
                frame = int(float(row.get("Frame") or row.get("frame")))
            except (TypeError, ValueError):
                continue
            try:
                length = _segment_length(row["X1"], row["Y1"], row["X2"], row["Y2"])
            except (KeyError, TypeError, ValueError):
                continue
            by_file_frame[fname][frame] += length

    out: Dict[str, Tuple[int, int]] = {}
    for fname, frames in by_file_frame.items():
        if not frames:
            continue
        ordered = sorted(frames.items(), key=lambda kv: kv[1], reverse=True)
        ed = int(ordered[0][0])
        es = int(ordered[-1][0]) if len(ordered) > 1 else ed
        out[fname] = (ed, es)
    return out


def load_filelist(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise ValueError(f"FileList.csv is empty: {path}")
    return rows


def build_pairs(
    filelist_rows: List[dict],
    videos_dir: Path,
    tracings: Optional[Dict[str, Tuple[int, int]]] = None,
    include_dilation: bool = False,
    require_video: bool = True,
) -> Tuple[List[dict], List[str]]:
    pairs: List[dict] = []
    skipped: List[str] = []
    tracings = tracings or {}
    for row in filelist_rows:
        raw_name = row.get("FileName") or row.get("Filename") or row.get("file_name")
        if not raw_name:
            skipped.append("row missing FileName")
            continue
        video_name = _video_filename(raw_name)
        stem = _stem(raw_name)
        video_path = videos_dir / video_name
        if not video_path.exists():
            # some dumps omit the extension in FileList but store .avi
            alt = videos_dir / f"{stem}.avi"
            if alt.exists():
                video_path = alt
                video_name = alt.name
        rel = f"Videos/{video_name}"
        if require_video and not video_path.exists():
            skipped.append(f"missing video: {video_path}")
            continue
        ef = _float(row, "EF", "ef", "LVEF")
        edv = _float(row, "EDV", "edv")
        esv = _float(row, "ESV", "esv")
        if ef is None:
            skipped.append(f"no EF: {video_name}")
            continue
        extra = {
            "file_name": video_name,
            "split": (row.get("Split") or row.get("split") or "").upper(),
        }
        if stem in tracings:
            extra["ed_frame"], extra["es_frame"] = tracings[stem]
        nframes = _float(row, "NumberOfFrames", "Frames")
        if nframes is not None:
            extra["n_frames"] = int(nframes)
        try:
            rec = pair_record(
                rel,
                ef=ef,
                edv=edv,
                esv=esv,
                extra=extra,
                include_dilation=include_dilation,
            )
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        pairs.append(rec)
    return pairs, skipped


def write_json(path: Path, pairs: List[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "pairs": pairs}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def subset_n(pairs: List[dict], n: int, seed: int) -> List[dict]:
    import random

    if n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    return [pairs[i] for i in idx[:n]]


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoNet-Dynamic → EchoCLIP manifest")
    parser.add_argument("--echonet-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "echonet_dynamic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help=(
            "Build a metadata-only manifest even when Videos/ is absent or some "
            "videos are missing. DANGER: such a manifest cannot run clinical "
            "evaluation; it is written with require_video=false and a loud warning."
        ),
    )
    parser.add_argument("--subset-5000", action="store_true",
                        help="Historical mixed-pool 5000 subset (R0 external anchor)")
    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Also write a TEST-only subset of N studies (test_subset_N.json + ids) "
            "for adapted-model evaluation (R2–R6). Disjointness is asserted."
        ),
    )
    parser.add_argument(
        "--include-dilation",
        action="store_true",
        help="Optional ablation: append EDV dilation captions (primary is EF-only)",
    )
    parser.add_argument(
        "--no-dilation",
        action="store_true",
        help="Deprecated alias: primary path is already EF-only",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    # Default: clinical manifests MUST have real videos. Metadata-only builds
    # require the explicit --allow-missing-videos opt-in and warn loudly.
    require_video = not bool(args.allow_missing_videos)
    root = args.echonet_root
    filelist = _find_filelist(root)
    if filelist is None:
        print(ECHONET_DOWNLOAD_HELP)
        print(f"\nLooked under: {root.resolve()}")
        return 1

    videos_dir = _find_videos_dir(root, filelist)
    if videos_dir is None and require_video:
        print(ECHONET_DOWNLOAD_HELP)
        print(f"\nFileList found at {filelist}, but no Videos/ directory.")
        return 1
    if videos_dir is None:
        videos_dir = root / "Videos"
        print(
            "WARNING: no Videos/ directory found and --allow-missing-videos was "
            "passed. Building a METADATA-ONLY manifest that CANNOT run clinical "
            "evaluation."
        )

    tracing_path = _find_tracings(root, filelist)
    tracings = ed_es_from_tracings(tracing_path) if tracing_path else {}

    rows = load_filelist(filelist)
    pairs, skipped = build_pairs(
        rows,
        videos_dir,
        tracings=tracings,
        include_dilation=bool(args.include_dilation) and not args.no_dilation,
        require_video=require_video,
    )
    if not pairs:
        print("No pairs could be built from FileList.csv.")
        for line in skipped[:20]:
            print(f"  - {line}")
        if skipped:
            print(f"  ... {len(skipped)} skipped")
        return 1

    meta = {
        "source": "EchoNet-Dynamic",
        "license": "Stanford AIMI non-commercial research license",
        "filelist": str(filelist),
        "videos_dir": str(videos_dir),
        "volume_tracings": str(tracing_path) if tracing_path else None,
        "n_pairs": len(pairs),
        "n_skipped": len(skipped),
        "seed": args.seed,
        "require_video": bool(require_video),
        "metadata_only": not bool(require_video),
        "text": "Official EchoCLIP prompt templates filled from EF/EDV (see echoclip.structured_text)",
        "note": "Demo retrieval metrics are not a substitute for this clinical split.",
    }
    if not require_video:
        meta["WARNING"] = (
            "METADATA-ONLY manifest built with --allow-missing-videos: one or more "
            "videos are absent on disk. This manifest CANNOT produce clinical EF "
            "metrics; it is for schema/wiring checks only. Do not report MAE."
        )
        print("\n" + "!" * 72)
        print("WARNING: METADATA-ONLY manifest (--allow-missing-videos).")
        print("         Videos are absent; clinical evaluation is IMPOSSIBLE.")
        print("         Do not report any MAE/AUC from this manifest.")
        print("!" * 72 + "\n")
    out_dir = args.output_dir
    write_json(out_dir / "manifest.json", pairs, meta)

    # --- Explicit per-split artifacts (P0-2) -----------------------------
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for rec in pairs:
        split = rec.get("split") or "UNKNOWN"
        by_split[split].append(rec)
    split_names = {"TRAIN": "train.json", "VAL": "val.json", "TEST": "test.json"}
    for split, items in by_split.items():
        name = split_names.get(split, f"{split.lower()}.json")
        write_json(out_dir / name, items, {**meta, "split": split, "n_pairs": len(items)})

    train_items = by_split.get("TRAIN", [])
    test_items = by_split.get("TEST", [])

    # Guard: TRAIN and TEST must be disjoint when both are present.
    if train_items and test_items:
        from echoclip.protocol import assert_disjoint

        assert_disjoint(
            train_items,
            test_items,
            id_key="file_name",
            label_train="TRAIN",
            label_eval="TEST",
        )

    if args.subset_5000:
        from echoclip.protocol import write_subset_ids

        # Historical mixed-pool subset (draws from TRAIN+VAL+TEST).
        # Kept ONLY as the R0 external/historical anchor — never for R2–R6.
        sub = subset_n(pairs, 5000, args.seed)
        anchor_name = "r0_external_anchor_5000"
        write_json(
            out_dir / f"{anchor_name}.json",
            sub,
            {
                **meta,
                "subset": 5000,
                "n_pairs": len(sub),
                "protocol": "EchoCLIP-style random 5000 (historical external anchor)",
                "ids_file": str(out_dir / f"{anchor_name}_ids.json"),
                "provenance": {
                    "role": "R0 external/historical anchor",
                    "sampled_from": "combined TRAIN+VAL+TEST pool",
                    "mixed_splits": True,
                    "may_overlap_train": True,
                    "seed": args.seed,
                    "note": (
                        "Drawn from the combined pool, so it MAY overlap TRAIN. "
                        "Use ONLY for R0 zero-shot external-anchor comparisons; "
                        "never for R2–R6 adapted models (use test_subset_N.json)."
                    ),
                },
            },
        )
        ids_path = write_subset_ids(
            sub,
            out_dir / f"{anchor_name}_ids.json",
            seed=args.seed,
            n=5000,
            source="EchoNet-Dynamic",
            already_sampled=True,
        )
        print(f"  Locked R0 anchor IDs → {ids_path} (+ .txt)")

    if args.test_subset is not None:
        from echoclip.protocol import assert_disjoint, write_subset_ids

        n_test = int(args.test_subset)
        if not test_items:
            print(
                "  WARNING: --test-subset requested but no TEST split rows found; "
                "skipping TEST-only subset."
            )
        else:
            sub_test = subset_n(test_items, n_test, args.seed)
            # TEST-only subset is by construction disjoint from TRAIN.
            if train_items:
                assert_disjoint(
                    train_items,
                    sub_test,
                    id_key="file_name",
                    label_train="TRAIN",
                    label_eval="TEST_SUBSET",
                )
            name = f"test_subset_{n_test}"
            write_json(
                out_dir / f"{name}.json",
                sub_test,
                {
                    **meta,
                    "split": "TEST",
                    "subset": n_test,
                    "n_pairs": len(sub_test),
                    "protocol": "TEST-only subset for adapted-model evaluation (R2–R6)",
                    "ids_file": str(out_dir / f"{name}_ids.json"),
                    "provenance": {
                        "role": "TEST-only subset (R2–R6)",
                        "sampled_from": "TEST split only",
                        "mixed_splits": False,
                        "may_overlap_train": False,
                        "seed": args.seed,
                    },
                },
            )
            ids_path = write_subset_ids(
                sub_test,
                out_dir / f"{name}_ids.json",
                seed=args.seed,
                n=n_test,
                source="EchoNet-Dynamic",
                already_sampled=True,
            )
            print(f"  Locked TEST-only subset IDs → {ids_path} (+ .txt)")

    print(f"Wrote {len(pairs)} pairs to {out_dir / 'manifest.json'}")
    for split, items in sorted(by_split.items()):
        print(f"  {split}: {len(items)}")
    if tracing_path:
        print(f"  ED/ES indices from VolumeTracings: {len(tracings)} videos")
    else:
        print("  VolumeTracings.csv not found; ed_frame/es_frame omitted (uniform/random sampling still works)")
    if skipped:
        print(f"  skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
