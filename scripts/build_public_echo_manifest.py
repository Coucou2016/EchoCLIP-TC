"""Build DATA.md manifests for public echo datasets (CAMUS, Pediatric, LVH, …).

Datasets are **not** bundled. Missing roots exit with download instructions.

Adapters
--------
- ``echonet_dynamic`` — delegates to the same FileList/Videos layout as
  ``build_echonet_manifest.py``
- ``camus`` — patient folders with Info_*.cfg (EF) + A2C/A4C sequences
- ``echonet_pediatric`` / ``echonet_lvh`` — EchoNet-style FileList.csv + Videos/

Examples
--------
::

  python scripts/build_public_echo_manifest.py --dataset camus --root E:\\data\\CAMUS
  python scripts/build_public_echo_manifest.py --dataset echonet_pediatric --root E:\\data\\EchoNet-Pediatric
  python scripts/build_public_echo_manifest.py --dataset echonet_lvh --root E:\\data\\EchoNet-LVH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.structured_text import pair_record
from echoclip.utils import set_seed

DOWNLOAD_HELP = {
    "echonet_dynamic": """
EchoNet-Dynamic not found under --root.

  https://echonet.github.io/dynamic/
  Stanford AIMI (non-commercial): https://stanfordaimi.azurewebsites.net/

Expected: <root>/FileList.csv and <root>/Videos/
""".strip(),
    "camus": """
CAMUS not found under --root.

  Challenge / download: https://www.creatis.insa-lyon.fr/Challenge/camus/
  (also mirrored on various research archives; you must obtain it yourself)

Expected layout (common)::

  <root>/training/patient0001/Info_2CH.cfg
  <root>/training/patient0001/*2CH*.nii  (or .mhd / .avi)
  <root>/testing/...

Info_*.cfg should contain an EF= or EjectionFraction= field.
""".strip(),
    "echonet_pediatric": """
EchoNet-Pediatric not found under --root.

  https://echonet.github.io/pediatric/
  Stanford AIMI: https://stanfordaimi.azurewebsites.net/

Expected: <root>/FileList.csv and <root>/Videos/ (EchoNet-style)
""".strip(),
    "echonet_lvh": """
EchoNet-LVH not found under --root.

  https://echonet.github.io/lvh/
  Stanford AIMI: https://stanfordaimi.azurewebsites.net/

Expected: <root>/FileList.csv and <root>/Videos/ (EchoNet-style).
EF may be absent on some releases — rows without EF are skipped.
""".strip(),
}


def _load_echonet_builder():
    script = ROOT / "scripts" / "build_echonet_manifest.py"
    spec = importlib.util.spec_from_file_location("build_echonet_manifest", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, pairs: List[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "pairs": pairs}, indent=2), encoding="utf-8")


def write_splits(out_dir: Path, pairs: List[dict], meta: dict) -> Dict[str, int]:
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for rec in pairs:
        split = (rec.get("split") or "UNKNOWN").upper()
        by_split[split].append(rec)
    counts = {}
    for split, items in by_split.items():
        name = {"TRAIN": "train.json", "VAL": "val.json", "TEST": "test.json"}.get(
            split, f"{split.lower()}.json"
        )
        write_json(out_dir / name, items, {**meta, "split": split, "n_pairs": len(items)})
        counts[split] = len(items)
    write_json(out_dir / "manifest.json", pairs, {**meta, "n_pairs": len(pairs)})
    return counts


# ---------------------------------------------------------------------------
# EchoNet-family (Dynamic / Pediatric / LVH)
# ---------------------------------------------------------------------------


def build_echonet_family(
    root: Path,
    *,
    dataset_key: str,
    require_video: bool = True,
    include_dilation: bool = True,
) -> Tuple[List[dict], dict, List[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            DOWNLOAD_HELP[dataset_key] + f"\n\nLooked under: {root.resolve()}"
        )
    mod = _load_echonet_builder()
    filelist = mod._find_filelist(root)
    if filelist is None:
        raise FileNotFoundError(DOWNLOAD_HELP[dataset_key])
    videos_dir = mod._find_videos_dir(root, filelist)
    if videos_dir is None and require_video:
        raise FileNotFoundError(
            DOWNLOAD_HELP[dataset_key] + f"\n\nFileList at {filelist} but no Videos/."
        )
    if videos_dir is None:
        videos_dir = root / "Videos"
    tracing_path = mod._find_tracings(root, filelist)
    tracings = mod.ed_es_from_tracings(tracing_path) if tracing_path else {}
    rows = mod.load_filelist(filelist)
    pairs, skipped = mod.build_pairs(
        rows,
        videos_dir,
        tracings=tracings,
        include_dilation=include_dilation,
        require_video=require_video,
    )
    meta = {
        "source": dataset_key,
        "license": "Stanford AIMI non-commercial research license (verify per dataset page)",
        "filelist": str(filelist),
        "videos_dir": str(videos_dir),
        "volume_tracings": str(tracing_path) if tracing_path else None,
        "n_pairs": len(pairs),
        "n_skipped": len(skipped),
        "text": "Official EchoCLIP prompt templates from EF/EDV (echoclip.structured_text)",
    }
    return pairs, meta, skipped


# ---------------------------------------------------------------------------
# CAMUS
# ---------------------------------------------------------------------------

_EF_CFG = re.compile(
    r"^(?:EF|EjectionFraction|LV_EF)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_VIEW_CFG = re.compile(r"Info_(2CH|4CH|2ch|4ch)\.cfg$", re.IGNORECASE)


def parse_camus_info_cfg(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
        elif ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _camus_ef(info: Dict[str, str], raw_text: str) -> Optional[float]:
    for key in ("EF", "EjectionFraction", "LV_EF", "ef"):
        if key in info and str(info[key]).strip() not in ("", "NA"):
            try:
                return float(info[key])
            except ValueError:
                continue
    match = _EF_CFG.search(raw_text)
    if match:
        return float(match.group(1))
    return None


def _camus_view(info_name: str) -> str:
    m = _VIEW_CFG.search(info_name)
    if not m:
        return "unknown"
    tag = m.group(1).upper()
    return "A2C" if tag.startswith("2") else "A4C"


def _find_camus_media(patient_dir: Path, view_tag: str) -> Optional[Path]:
    """Prefer a sequence file over single ED/ES stills when present."""
    patterns = [
        f"*_{view_tag}_half_sequence*",
        f"*_{view_tag}_sequence*",
        f"*_{view_tag}.*",
        f"*_{view_tag}_ED.*",
    ]
    # Normalize 2CH/4CH
    alts = [view_tag, view_tag.replace("A2C", "2CH").replace("A4C", "4CH")]
    if view_tag == "A2C":
        alts += ["2CH", "2ch"]
    if view_tag == "A4C":
        alts += ["4CH", "4ch"]
    exts = {".avi", ".mp4", ".nii", ".nii.gz", ".mhd", ".png", ".jpg", ".jpeg"}
    candidates: List[Path] = []
    for alt in alts:
        for path in patient_dir.glob(f"*_{alt}*"):
            if path.suffix.lower() in exts or path.name.lower().endswith(".nii.gz"):
                # Skip gt / masks when obvious
                low = path.name.lower()
                if any(x in low for x in ("_gt", "mask", "seg")):
                    continue
                candidates.append(path)
    if not candidates:
        return None
    # Prefer sequences / videos
    def score(p: Path) -> tuple:
        low = p.name.lower()
        return (
            0 if "sequence" in low else 1,
            0 if p.suffix.lower() in {".avi", ".mp4"} else 1,
            len(p.name),
        )

    candidates.sort(key=score)
    return candidates[0]


def discover_camus_patients(root: Path) -> List[Path]:
    patients: List[Path] = []
    for split_name in ("training", "train", "testing", "test", "val", "validation"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        for child in sorted(split_dir.iterdir()):
            if child.is_dir() and child.name.lower().startswith("patient"):
                patients.append(child)
    # Flat layout: root/patientXXXX
    if not patients:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.lower().startswith("patient"):
                patients.append(child)
    return patients


def build_camus(
    root: Path,
    *,
    require_media: bool = True,
    include_dilation: bool = False,
) -> Tuple[List[dict], dict, List[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            DOWNLOAD_HELP["camus"] + f"\n\nLooked under: {root.resolve()}"
        )
    patients = discover_camus_patients(root)
    if not patients:
        # Also accept nested database/
        nested = root / "database"
        if nested.is_dir():
            patients = discover_camus_patients(nested)
            root = nested
    if not patients:
        raise FileNotFoundError(DOWNLOAD_HELP["camus"] + f"\n\nLooked under: {root.resolve()}")

    pairs: List[dict] = []
    skipped: List[str] = []
    for patient_dir in patients:
        infos_raw = list(patient_dir.glob("Info_*.cfg")) + list(
            patient_dir.glob("info_*.cfg")
        )
        # Windows is case-insensitive; dedupe by resolved path
        seen = set()
        infos = []
        for path in infos_raw:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            infos.append(path)
        if not infos:
            skipped.append(f"no Info_*.cfg in {patient_dir}")
            continue
        # Infer split from parent folder name
        parent = patient_dir.parent.name.lower()
        if parent in ("training", "train"):
            split = "TRAIN"
        elif parent in ("testing", "test"):
            split = "TEST"
        elif parent in ("val", "validation"):
            split = "VAL"
        else:
            split = "UNKNOWN"
        for info_path in infos:
            raw = info_path.read_text(encoding="utf-8", errors="ignore")
            info = parse_camus_info_cfg(info_path)
            ef = _camus_ef(info, raw)
            if ef is None:
                skipped.append(f"no EF in {info_path}")
                continue
            view = _camus_view(info_path.name)
            view_tag = "2CH" if view == "A2C" else "4CH"
            media = _find_camus_media(patient_dir, view_tag)
            if media is None and require_media:
                skipped.append(f"no media for {patient_dir.name} {view}")
                continue
            if media is None:
                rel = f"{patient_dir.relative_to(root).as_posix()}/{view_tag}"
            else:
                try:
                    rel = media.relative_to(root).as_posix()
                except ValueError:
                    rel = str(media)
            try:
                rec = pair_record(
                    rel,
                    ef=ef,
                    edv=None,
                    esv=None,
                    extra={
                        "split": split,
                        "view": view,
                        "patient": patient_dir.name,
                        "file_name": media.name if media else f"{patient_dir.name}_{view}",
                        "dataset": "camus",
                    },
                    include_dilation=include_dilation,
                )
            except ValueError as exc:
                skipped.append(str(exc))
                continue
            pairs.append(rec)

    meta = {
        "source": "CAMUS",
        "license": "CAMUS challenge terms (verify before redistribution)",
        "root": str(root.resolve()),
        "n_pairs": len(pairs),
        "n_skipped": len(skipped),
        "n_patients": len(patients),
        "views": "A2C/A4C from Info_2CH / Info_4CH",
        "text": "Official EchoCLIP EF prompt templates (no dilation without EDV)",
        "note": "CAMUS volumes may be NIfTI/MHD; ensure echoclip.preprocess can read them or convert to AVI/PNG.",
    }
    return pairs, meta, skipped


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AdapterFn = Callable[..., Tuple[List[dict], dict, List[str]]]

ADAPTERS: Dict[str, AdapterFn] = {
    "echonet_dynamic": lambda root, **kw: build_echonet_family(
        root, dataset_key="echonet_dynamic", **kw
    ),
    "echonet_pediatric": lambda root, **kw: build_echonet_family(
        root, dataset_key="echonet_pediatric", **kw
    ),
    "echonet_lvh": lambda root, **kw: build_echonet_family(
        root, dataset_key="echonet_lvh", **kw
    ),
    "camus": lambda root, **kw: build_camus(
        root,
        require_media=kw.get("require_media", kw.get("require_video", True)),
        include_dilation=kw.get("include_dilation", False),
    ),
}

DEFAULT_OUTPUT = {
    "echonet_dynamic": "echonet_dynamic",
    "echonet_pediatric": "echonet_pediatric",
    "echonet_lvh": "echonet_lvh",
    "camus": "camus",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Public echo dataset → EchoCLIP manifest")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(ADAPTERS.keys()),
        help="Dataset adapter",
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-media", action="store_true")
    parser.add_argument("--no-dilation", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = args.dataset
    out_dir = args.output_dir or (ROOT / "data" / DEFAULT_OUTPUT[dataset])
    adapter = ADAPTERS[dataset]

    try:
        if dataset == "camus":
            pairs, meta, skipped = adapter(
                args.root,
                require_media=not args.allow_missing_media,
                include_dilation=not args.no_dilation,
            )
        else:
            pairs, meta, skipped = adapter(
                args.root,
                require_video=not args.allow_missing_media,
                include_dilation=not args.no_dilation,
            )
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if not pairs:
        print(f"No pairs built for {dataset}.")
        for line in skipped[:25]:
            print(f"  - {line}")
        if not skipped:
            print(DOWNLOAD_HELP.get(dataset, "Check --root layout."))
        return 1

    meta["seed"] = args.seed
    counts = write_splits(out_dir, pairs, meta)
    print(f"Wrote {len(pairs)} pairs → {out_dir / 'manifest.json'}")
    for split, n in sorted(counts.items()):
        print(f"  {split}: {n}")
    if skipped:
        print(f"  skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
