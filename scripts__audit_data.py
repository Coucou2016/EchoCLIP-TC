"""Data authenticity audit for the EchoCLIP-TA/TC paper.

Checks, with values written to results/zh/data_audit.json:
  1. FileList.csv row count, column set, split counts, EF/volume range
  2. Videos/ directory file count and name agreement with FileList
  3. Manifest pair counts vs FileList split counts, id agreement
  4. Pairwise split intersections (train/val/test) must all be empty
  5. VolumeTracings coverage and ED/ES frame sanity
  6. test_subset_500 relationship to the full TEST split
  7. The external anchor manifest composition (mixed pool, not used for R2-R6)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import cache_dir, results_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\Datasets\EchoNet-Dynamic")
MANI = ROOT / "data" / "echonet_dynamic"
OUT = results_dir()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stems(rows) -> set:
    return {Path(str(p["file_name"])).stem for p in rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    fl = pd.read_csv(DATA / "FileList.csv")
    fl["stem"] = fl["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    audit["filelist"] = {
        "path": str(DATA / "FileList.csv"),
        "sha256": sha256(DATA / "FileList.csv"),
        "rows": int(len(fl)),
        "columns": list(fl.columns),
        "split_counts": {k: int(v) for k, v in fl["Split"].value_counts().items()},
        "split_sum_check": int(fl["Split"].value_counts().sum()) == int(len(fl)),
        "ef": {
            "min": float(fl["EF"].min()),
            "max": float(fl["EF"].max()),
            "mean": float(fl["EF"].mean()),
            "sd": float(fl["EF"].std()),
        },
        "frames": {
            "min": int(fl["NumberOfFrames"].min()),
            "max": int(fl["NumberOfFrames"].max()),
            "mean": float(fl["NumberOfFrames"].mean()),
        },
        "duplicate_file_names": int(fl["FileName"].duplicated().sum()),
        "nan_ef": int(fl["EF"].isna().sum()),
    }

    videos = sorted((DATA / "Videos").glob("*.avi"))
    video_stems = {p.stem for p in videos}
    audit["videos"] = {
        "dir": str(DATA / "Videos"),
        "n_avi": len(videos),
        "filelist_unique": int(fl["stem"].nunique()),
        "missing_on_disk": len(set(fl["stem"]) - video_stems),
        "extra_on_disk": len(video_stems - set(fl["stem"])),
        "name_agreement": set(fl["stem"]) == video_stems,
    }

    # ---- manifests -------------------------------------------------------
    man = {}
    ids_by_split = {}
    for name in ("train", "val", "test", "test_subset_500", "r0_external_anchor_5000"):
        d = json.loads((MANI / f"{name}.json").read_text())
        pairs = d["pairs"]
        man[name] = {
            "path": str(MANI / f"{name}.json"),
            "sha256": sha256(MANI / f"{name}.json"),
            "n_pairs": len(pairs),
            "meta_n_pairs": d["meta"].get("n_pairs"),
            "split_field": d["meta"].get("split"),
            "n_skipped": d["meta"].get("n_skipped"),
            "n_unique_ids": len(stems(pairs)),
            "duplicate_ids": len(pairs) - len(stems(pairs)),
            "ef_mean": float(np.mean([p["ef"] for p in pairs])),
            "with_ed_es": int(sum(1 for p in pairs if p.get("ed_frame", -1) is not None and p.get("ed_frame", -1) >= 0)),
        }
        ids_by_split[name] = stems(pairs)
    audit["manifests"] = man

    # manifest vs FileList split agreement
    fl_split = {
        "train": set(fl.loc[fl["Split"] == "TRAIN", "stem"]),
        "val": set(fl.loc[fl["Split"] == "VAL", "stem"]),
        "test": set(fl.loc[fl["Split"] == "TEST", "stem"]),
    }
    audit["manifest_vs_filelist"] = {}
    for name, ref in (("train", "train"), ("val", "val"), ("test", "test")):
        m = ids_by_split[name]
        f = fl_split[ref]
        audit["manifest_vs_filelist"][name] = {
            "manifest_n": len(m),
            "filelist_n": len(f),
            "equal": m == f,
            "only_manifest": len(m - f),
            "only_filelist": len(f - m),
        }
    audit["test_subset_500"] = {
        "n": len(ids_by_split["test_subset_500"]),
        "subset_of_test": ids_by_split["test_subset_500"] <= fl_split["test"],
        "fraction_of_test": len(ids_by_split["test_subset_500"]) / max(len(fl_split["test"]), 1),
    }
    anchor = ids_by_split["r0_external_anchor_5000"]
    audit["r0_external_anchor"] = {
        "n": len(anchor),
        "overlap_with_train": len(anchor & fl_split["train"]),
        "overlap_with_val": len(anchor & fl_split["val"]),
        "overlap_with_test": len(anchor & fl_split["test"]),
        "note": (
            "Mixed historical pool of EchoNet-Dynamic ids. Used only as an external "
            "anchor for R0 scale checks, never for any R2-R6 number in the paper."
        ),
    }

    # ---- split isolation -------------------------------------------------
    iso = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = ids_by_split[a] & ids_by_split[b]
        iso[f"{a}_cap_{b}"] = {"n": len(inter), "examples": sorted(inter)[:3]}
    iso["all_zero"] = all(v["n"] == 0 for v in iso.values() if isinstance(v, dict))
    audit["split_isolation"] = iso

    # ---- volume tracings -------------------------------------------------
    vt = pd.read_csv(DATA / "VolumeTracings.csv")
    vt["stem"] = vt["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    per_file = vt.groupby("stem")["Frame"].agg(["min", "max", "nunique", "count"])
    audit["volume_tracings"] = {
        "path": str(DATA / "VolumeTracings.csv"),
        "sha256": sha256(DATA / "VolumeTracings.csv"),
        "rows": int(len(vt)),
        "columns": list(vt.columns),
        "n_files": int(per_file.shape[0]),
        "frames_per_file_values": [int(x) for x in sorted(per_file["nunique"].unique())],
        "rows_per_file_median": float(per_file["count"].median()),
        "files_with_two_distinct_frames": int((per_file["nunique"] == 2).sum()),
        "coverage_of_filelist": float(per_file.shape[0] / len(fl)),
    }
    # how many manifest pairs have a valid ED/ES frame inside the decoded length
    fr = dict(zip(fl["stem"], fl["NumberOfFrames"]))
    cov = {}
    for name in ("train", "val", "test", "test_subset_500"):
        pairs = json.loads((MANI / f"{name}.json").read_text())["pairs"]
        ok, bad = 0, 0
        for p in pairs:
            ed, es = p.get("ed_frame", -1), p.get("es_frame", -1)
            n = int(fr.get(Path(p["file_name"]).stem, 0))
            if ed is not None and es is not None and ed >= 0 and es >= 0 and ed < n and es < n and ed != es:
                ok += 1
            else:
                bad += 1
        cov[name] = {
            "n": len(pairs),
            "valid_ed_es": ok,
            "invalid": bad,
            "coverage": ok / max(len(pairs), 1),
        }
    audit["ed_es_coverage"] = cov

    # ---- labels used by the cache ---------------------------------------
    audit["cache_labels"] = {}
    for npz in sorted(cache_dir().glob("*.npz")):
        if npz.name.startswith("text_"):
            continue
        z = np.load(npz, allow_pickle=False)
        ef = z["ef"]
        audit["cache_labels"][npz.name] = {
            "n": int(len(ef)),
            "frames_per_video": int(z["emb"].shape[1]),
            "dim": int(z["emb"].shape[2]),
            "ef_mean": float(np.mean(ef)),
            "ef_sd": float(np.std(ef)),
            "n_ef_lt50": int((ef < 50).sum()),
            "n_ef_lt40": int((ef < 40).sum()),
            "n_ef_lt30": int((ef < 30).sum()),
            "norm_min": float(np.linalg.norm(z["emb"][0], axis=-1).min()),
            "norm_max": float(np.linalg.norm(z["emb"][0], axis=-1).max()),
            "bytes": int(npz.stat().st_size),
        }

    # ---- label distribution of the cached TRAIN split --------------------
    tr_npz = cache_dir() / "train_32_bgr.npz"
    if tr_npz.exists():
        zt = np.load(tr_npz, allow_pickle=False)
        eft = zt["ef"]
        audit["train_distribution"] = {
            "n": int(len(eft)),
            "ef_mean": float(np.mean(eft)),
            "ef_sd": float(np.std(eft)),
            "frac_50_65": float(np.mean((eft >= 50) & (eft < 65))),
            "frac_lt30": float(np.mean(eft < 30)),
            "frac_lt40": float(np.mean(eft < 40)),
            "frac_gt70": float(np.mean(eft > 70)),
            "lt50": float(np.mean(eft < 50)),
        }

    # ---- TEST distribution vs TEST-500 ----------------------------------
    test_all = fl[fl["Split"] == "TEST"]
    sub = test_all[test_all["stem"].isin(ids_by_split["test_subset_500"])]
    audit["test_distribution"] = {
        "full_test_n": int(len(test_all)),
        "full_test_ef_mean": float(test_all["EF"].mean()),
        "full_test_ef_sd": float(test_all["EF"].std()),
        "full_test_lt50": float((test_all["EF"] < 50).mean()),
        "full_test_lt40": float((test_all["EF"] < 40).mean()),
        "full_test_lt30": float((test_all["EF"] < 30).mean()),
        "subset_n": int(len(sub)),
        "subset_ef_mean": float(sub["EF"].mean()),
        "subset_ef_sd": float(sub["EF"].std()),
        "subset_lt50": float((sub["EF"] < 50).mean()),
        "subset_lt40": float((sub["EF"] < 40).mean()),
        "subset_lt30": float((sub["EF"] < 30).mean()),
    }

    (OUT / "data_audit.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT / 'data_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
