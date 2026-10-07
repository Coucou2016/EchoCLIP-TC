# EchoCLIP-TA — Strict-review closure (P0/P1), 2026-10-07

Working copy: `E:\Projects\20260522-EchoCLIP` (tree layout, runnable).
Scope: P0-1…P0-8 + selected P1 items from the strict peer review. **No** remote
commits, **no** pushes, **no** edits to the flattened audit mirror, and **no**
invented clinical/parity numbers.

Environment note (not a repo defect): on this Windows + Python 3.13 Miniconda
box, `pytest` crashed at import with
`ImportError: cannot import name 'LZMA' from 'numcodecs.lzma'` (import chain
`zarr` → `numcodecs`, pulled in by the `zarr.testing` entry-point plugin).
Fix: `pytest.ini` with `addopts = -p no:zarr`. This is a documented
environment workaround, not a change to project code.

---

## Status table

| Item | What changed | Files | Tests | Status |
|------|--------------|-------|-------|--------|
| **P0-1** | CLI flag + explicit `require_video` + loud warnings | `scripts/build_echonet_manifest.py` | `tests/test_build_echonet_manifest_cli.py` | **Done** |
| **P0-2** | Per-split artifacts, R0 anchor rename, TEST-only subset, `assert_disjoint`, provenance hashes | `scripts/build_echonet_manifest.py`, `echoclip/protocol.py`, `scripts/run_protocol.py`, `DATA.md` | `tests/test_split_integrity.py` | **Done** |
| **P0-3** | Unified supervised EF schema + `direct_ef` prediction mode | `echoclip/supervised_checkpoint.py`, `echoclip/checkpoint.py`, `scripts/train_supervised.py`, `scripts/eval_clinical.py`, `scripts/run_protocol.py` | `tests/test_supervised_roundtrip.py`, `tests/test_prediction_mode.py` | **Done** |
| **P0-4** | `--seed`/`--determinism` propagation to both trainers + seed provenance in checkpoints | `scripts/train.py`, `scripts/train_supervised.py`, `echoclip/utils.py`, `echoclip/checkpoint.py`, `scripts/run_seeds.py` (existing forwarding) | `tests/test_seed_propagation.py` | **Done** |
| **P0-5** | Standalone official R0 evaluator (clean-room, bypasses `EchoCLIPDataset`) | `echoclip/official_r0.py`, `scripts/eval_official_r0.py` | `tests/test_official_r0_standalone.py` | **Done** (bit-exact AVI+hub parity **blocked** without gated assets) |
| **P0-6** | Honest metadata gating + graceful golden-parity skip | `echoclip/official_r0.py`, `scripts/eval_official_r0.py`, `echoclip/protocol.py` (`merge_metrics_meta`) | `tests/test_official_r0_standalone.py`, updated `tests/test_protocol.py` | **Done** |
| **P0-7** | Repo integrity audit + CI wiring + doc path fixes | `scripts/audit_repo_integrity.py`, `.github/workflows/ci.yml`, `DATA.md` | `tests/test_audit_repo_integrity.py` | **Done** |
| **P0-8** | `${ENV}`/`~` expansion in config loading | `echoclip/config_io.py`, `scripts/run_protocol.py`, `scripts/train.py`, `scripts/train_supervised.py`, `scripts/eval_clinical.py`, `scripts/eval.py` | `tests/test_config.py` | **Done** |
| **P1-2** | Finite-EF subset for the soft term (no whole-batch drop) | `echoclip/loss.py` | `tests/test_loss.py` | **Done** |
| **P1-3** | Real hard/soft mixture `lambda_soft` (0 == exact hard InfoNCE) | `echoclip/loss.py`, `scripts/train.py` | `tests/test_loss.py` | **Done** |
| **P1-6** | `affine_logistic` is the paper-path calibration default | `scripts/run_protocol.py`, `scripts/eval_clinical.py` | `tests/test_eval_clinical.py` | **Done** |
| **P1-7** | VAL-scale / VAL-cal conformal separation + weakened wording | `echoclip/clinical.py` (existing), tests assert wording | `tests/test_adaptive_bootstrap.py` | **Done** (already implemented; now regression-pinned) |
| **P1-9** | Stratified bootstrap for binary AUC | `echoclip/clinical.py` | `tests/test_adaptive_bootstrap.py` | **Done** |

---

## Item detail

### P0-1 — manifest CLI bug
`args.allow_missing_videos` is now a real `--allow-missing-videos` flag, and the
default is explicit: `require_video = not bool(args.allow_missing_videos)`.
Metadata-only builds print a loud banner and stamp `metadata_only: true`,
`require_video: false`, plus a `WARNING` key in the manifest payload. The
standard build path no longer raises `AttributeError`.

### P0-2 — split leakage
* `train.json` / `val.json` / `test.json` (+ `*_ids.json`) are written per split.
* The mixed 5000 subset is now `r0_external_anchor_5000.json` with embedded
  provenance (`experiment: R0_external_anchor`, `pool: mixed_train_val_test`,
  `disjoint_from_train: false`).
* `--test-subset N` writes a TEST-only `test_subset_N.json`.
* `assert_disjoint(...)` raises `RuntimeError` listing the first overlapping IDs
  and guards R2–R6 via `_guard_adapted_split_integrity`.
* Metrics/manifest JSON carry `train_manifest_sha256`, `test_manifest_sha256`,
  `train_n`, `test_n`, `train_test_overlap_n`.
* VAL/TEST `mixed`/`ed_es` hard guards were **not** weakened.

### P0-3 — supervised direct-EF evaluation
Single on-disk schema (`echoclip/supervised_checkpoint.py`, `SCHEMA_VERSION=2`,
`task="ef_regression"`) capturing backbone config + load source, `head_kind`,
`head_state_dict`, `temporal_state_dict`, `train_cfg`, `train_seed`. A fresh
Python process can load `best.pt` and run TEST inference with no in-process head.
`eval_clinical.py` gained `--prediction-mode {zeroshot,direct_ef}`
(`direct_regression` accepted as a legacy alias); `run_protocol.py` routes
R2/R3/R4 to `direct_ef` while R0/R1/R5 stay `zeroshot`.

### P0-4 — real multi-seed training
`train.py` and `train_supervised.py` accept `--seed` (overrides `cfg["seed"]`)
and `--determinism {fast,strict}`. Checkpoints and `train_meta.json` persist
`train_seed`, `split_seed`, `sampler_seed`, `determinism_mode`, and the full
`determinism` provenance dict (cuDNN flags + CUDA presence). `set_seed()` seeds
`random`, `numpy`, `torch`, and all CUDA devices. `run_seeds.py` already forwards
`--seed` into `run_protocol.py`, which forwards it to both trainers.

### P0-5 / P0-6 — standalone official R0 + honest metadata
`echoclip/official_r0.py` implements the documented behavior independently of
`EchoCLIPDataset(video_frames=16)`: OpenCV read → `crop_and_scale` → stride
`0:min(40,T):2` → official OpenCLIP `preprocess_val` → hub model/tokenizer →
0..100 EF grid (× 2 templates = 202 prompts) → official regression aggregation.
It records frame indices, prompt-grid hash, input hash, tokenizer/hub names,
upstream repo + commit.

Honest metadata: `paper_mode`, `official_reproduction_requested`,
`official_reproduction_verified`, and the legacy `official_reproduction` alias
(only ever true when **verified**). `parity_report()` returns a *skipped* report
when the official example AVI or hub weights are missing — no fabricated numbers,
no failing CI. `--paper` alone never sets `verified=true`.

### P0-7 — repo integrity audit
`scripts/audit_repo_integrity.py` checks `REQUIRED_PATHS` plus backticked /
layout-listed repo paths in `README.md` and `DATA.md`, exits non-zero with a
per-file report, and exempts paths under a `Planned` heading. Wired into
`.github/workflows/ci.yml`. The audit targets the tree-layout repo only.

### P0-8 — config env expansion
`echoclip/config_io.py::load_yaml_config()` recursively expands
`os.path.expanduser(os.path.expandvars(...))` over str/dict/list, with optional
`strict=True` (raises on unset `${VAR}`) and an optional provenance stamp. All
YAML loaders (`run_protocol`, `train`, `train_supervised`, `eval_clinical`,
`eval`) route through it, so `${ECHONET_ROOT}` in `configs/echonet_dynamic.yaml`
resolves.

### P1 items
* **P1-2** — the soft term now computes on the finite-EF subset only; `<2` valid
  labels (or all-NaN) falls back to hard InfoNCE.
* **P1-3** — `lambda_soft` is a real mix ratio: `0` is *exactly* hard InfoNCE
  (soft branch short-circuited), `1` is pure soft (legacy). The old
  `soft_weight` kwarg is a deprecated alias that warns.
* **P1-6** — paper path defaults to `affine_logistic`; `temperature` remains the
  non-paper default and can always be forced explicitly.
* **P1-7** — VAL-scale / VAL-cal separation was already present; the wording
  ("empirical heuristic") and split flags are now regression-pinned.
* **P1-9** — `stratified_bootstrap_auc()` resamples positives/negatives
  separately so small minority classes cannot produce NaN CIs.

---

## Full-suite result

```
python -m pytest tests -q
194 passed in 611.22s (0:10:11)
```

No failures, no skips, no xfails. Tests that exercise GPU training were pinned
to `--device cpu` in the new tests to avoid VRAM contention on the 4 GB dev GPU.

## Blocked / not fully verifiable here

* **Bit-exact official AVI + hub parity (P0-5/P0-6):** EchoNet-Dynamic clinical
  data is license-gated and the official example AVI is not in the repo. The
  golden-parity harness therefore *skips gracefully*; it will never fabricate a
  parity value or a clinical MAE. `official_reproduction_verified` stays `false`
  until a real parity report passes.
* No clinical number, parity value, or MAE was invented anywhere in this change.

## Prompt action

None required. No remote was contacted; all changes are local to the tree-layout
working copy.
