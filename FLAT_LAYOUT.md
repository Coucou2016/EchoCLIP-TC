# FLAT_LAYOUT.md — why this repository has no folders

## 1. Purpose

`Coucou2016/EchoCLIP-TC` is the **audit surface** for the EchoCLIP-TC project. Its single
design constraint is:

> Any agent or human, using only a repository listing and a bulk download, must be able to
> obtain and read **100 % of the project's code, documentation, manuscript, reports and
> core result data** without knowing anything about the project's internal structure.

A conventional tree (`src/`, `docs/`, `experiments/`, `results/`) fails this constraint in
practice: LLM agents routinely fetch a subset of directories, truncate deep trees, or skip
subtrees they consider irrelevant. A flat root removes that failure mode entirely.

## 2. Consequences of the flat layout

| Property | Effect |
|---|---|
| One namespace, no recursion | A single listing returns every file |
| Name = origin | The `__` prefix encodes the original folder, so provenance is never lost |
| Unique top-level names | Any file can be cited by a stable, unambiguous path |
| No orphans | Files cannot be "hidden" inside a folder that a crawler skips |
| Verifiable completeness | `FLAT_LAYOUT_MANIFEST.json` gives an expected file count + hashes |

## 3. Naming convention

```
<original_dir_1>__<original_dir_2>__...__<original_filename>
```

* Root-level files keep their name unchanged.
* The original root `README.md` is preserved as `README_project_original.md`; the flat
  repository's own `README.md` is this navigation document.
* Files added by this repository, which did not exist in the original tree:
  `README.md` (navigation), `FLAT_LAYOUT.md` (rationale), `FLAT_LAYOUT_MANIFEST.json`
  (machine-readable index), `restore_flat_to_tree.ps1` (rehydration helper),
  `_flat_map.json` (build-time `flat_name -> original_path` map), and a regenerated
  `.gitignore`.

## 4. Inclusion policy

**Included**

* All library source (`echoclip__*.py`)
* All CLI entry points (`scripts__*.py`)
* All tests (`tests__*.py`)
* All configuration (`configs__*.yaml`)
* All documentation (`*.md`, `docs__*.md`, `DATA.md`, `PAPER.md`, `ATTRIBUTION.md`, `NOTICE`)
* Manuscript, in editable and rendered form (`papers__*`)
* Reports, review responses, collaboration logs (`reports__*`)
* Publication figures as PNG **and** vector PDF (`figures__*`, `reports__figures__*`)
* **Core result data as text/JSON** — metrics, summaries, splits, comparison tables
  (`checkpoints__*.json`, `reports__attention_edes__*.csv/json`)
* Demo manifests and templates (`data__*`)
* CI workflow (`.github__workflows__ci.yml`)

**Excluded**

| Excluded | Reason |
|---|---|
| `*.pt`, `*.pth`, `*.ckpt` model weights (254–280 MB) | GitHub hard limit 100 MB/file; reproducible |
| `open_clip_pytorch_model.bin` (606 MB) | Same; public download documented in `docs__OBTAIN_DATA_AND_WEIGHTS.md` |
| `*.safetensors`, `*.onnx`, `*.h5`, `*.hdf5` | Large binaries, no review value |
| `*.zip` staging bundles | Archived duplicates of tracked sources |
| EchoNet-Dynamic `*.avi` / any clinical video | License-gated (AIMI, non-commercial) — **never redistributed** |
| `data/demo/images/*.png` | Regenerable via `scripts__make_demo_data.py` |
| `__pycache__/`, `*.pyc`, caches | Build artifacts |
| `.git/`, env files, credentials | Security |

Everything excluded is either **trivially reproducible** or **legally restricted**; nothing
excluded is needed to audit a claim, because every reported number is present as JSON.

## 5. What a reviewer should be able to do here

1. Read the manuscript end-to-end (`papers__echoclip_tc_manuscript.md`).
2. Check each numeric claim against the raw JSON in `checkpoints__*` / `reports__*`.
3. Read the exact code path that produced each number (`scripts__run_*`, `echoclip__*`).
4. Read the tests that pin the protocol (`tests__test_*`).
5. Read the review history and confirm each P0/P1 item was actually closed
   (`reports__review_response_*`, `reports__ZERO_TAIL_COMPLETE_20260915.md`).
6. Verify completeness by diffing the file list against `FLAT_LAYOUT_MANIFEST.json`.

## 6. Reproducing the original structure

```powershell
pwsh -File restore_flat_to_tree.ps1 -Destination ..\EchoCLIP-TC-tree
```

The script rewrites `<a>__<b>__<file>` back to `<a>/<b>/<file>` and restores the original
`README.md`. Resulting tree is directly runnable (`pytest tests -q`, `python scripts/...`).

## 7. Maintenance

When the main working copy changes, regenerate this repository rather than hand-editing it,
so that the flat names and the manifest stay in sync:

```powershell
# from the main working copy
pwsh -File _dual_agent_staging\build_flat_repo.ps1
```

Then re-commit the flat repository and push.
