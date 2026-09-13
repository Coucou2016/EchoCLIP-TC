# Review response — complete checklist (2026-09-14)

Peer-review **Major Concerns 1–8** + P0/P1/P2 mapped to
https://github.com/Coucou2016/EchoCLIP-TC. **No clinical MAE numbers invented.**

## Push status

| Field | Value |
|-------|-------|
| **Push** | Succeeded (via GitHub git Data API; local `git push` HTTPS to :443 was unreachable) |
| **Remote** | `https://github.com/Coucou2016/EchoCLIP-TC.git` |
| **Branch** | `main` |
| **Remote tip SHA** | `b288214704dea8eeb2d3117ef938f326fa2351e5` |
| **Feature commit SHA** | `7b7712db3778653cf5f627bb3d4ca5be5e705c45` |
| **When** | 2026-09-14 |

## Major Concerns 1–8

| # | Concern | Status | Where / notes |
|---|---------|--------|----------------|
| **1** | Annotation-assisted sampling contaminates primary VAL/TEST | **Done** | `assert_primary_eval_sampling` / `resolve_eval_sample_strategy`; R0–R6 force uniform; eval/protocol hard-fail |
| **2** | Missing Oracle / annotation-assisted upper bound | **Done** | `ORACLE_EDES`; `annotation_assisted=True`; comparison table labels Oracle |
| **3** | B0 not proven official EchoCLIP parity | **Partial** | `--paper`: EF `0..100`, `official_stride` `0:min(40,T):2`, no scratch; golden aggregation tests vs official utils; `scripts/compare_official_b0.py`. **Blocked** on hub weights + AVI for end-to-end bit-exact; `PARITY_GAPS` documented |
| **4** | M2 / R5 misframed as zero-shot temporal | **Done** | R5 = EF-supervised PEFT temporal adaptation; not “first temporal EchoCLIP” |
| **5** | Primary captions should be EF-only | **Done** | `include_dilation=False` default; `--use-edv-captions` / `--include-dilation` opt-in |
| **6** | Missing supervised baselines S0–S2 | **Done** | R2–R4 + `scripts/train_supervised.py` (`--demo` / `--paper`) |
| **7** | Calibration / conformal honesty (pseudo-logit, fixed width) | **Done** | Affine logistic preferred; adaptive conformal + risk–coverage / AURC; `scripts/plot_risk_coverage.py` |
| **8** | Provenance / LICENSE / hard-coded machine paths | **Done** | ATTRIBUTION + LICENSE footnote; env placeholders; remaining `E:\` stripped from scripts/DATA.md |

## P0 / P1 / P2 table

| Item | Priority | Status | Notes |
|------|----------|--------|-------|
| Uniform VAL/TEST + Oracle gate | P0 | **Done** | |
| R0–R6 matrix + legacy aliases | P0 | **Done** | |
| `--paper` hard-fail (train/eval/protocol/seeds/supervised) | P0 | **Done** | |
| Temporal mask + uniform subsample | P0 | **Done** | |
| Dataset `set_epoch` | P0 | **Done** | |
| EF soft contrastive (R5 default ON) | P1 | **Done** | |
| Calibration @50/40/30 + affine | P1 | **Done** | |
| Adaptive conformal + AURC | P1 | **Done** | |
| Paired bootstrap ΔMAE / CIs | P1 | **Done** | |
| Multi-seed helper (`run_seeds.py`) | P1 | **Done** | Paper 5-seed **Blocked** (needs GPU + data) |
| requirements-lock + path hygiene | P1 | **Done** | |
| Manuscript / research_report sync | P1 | **Done** | DEMO / 待补充 retained |
| Golden B0 aggregation parity tests | P1 | **Done** | Tensor-level; not end-to-end MAE |
| `compare_official_b0.py` | P1 | **Done** | Runs when AVI + hub available; else exit 2 |
| GitHub Actions CI (unittest + validate --skip-eval) | P1 | **Done** | `.github/workflows/ci.yml` (CPU) |
| Risk–coverage SciencePlots helper | P1 | **Done** | `--demo-curve` labeled DEMO |
| CardiacCLIP comparison numbers | P2 | **Blocked** | Stub only; external weights |
| Attention / ED–ES analysis on EchoNet | P2 | **Partial** | Toy script; needs VolumeTracings |
| EchoNet clinical MAE table | P2 | **Blocked** | No AIMI data on disk (search below) |
| 5-seed paper runs | P2 | **Blocked** | Helper ready; needs assets + GPU |

## EchoNet-Dynamic disk search (2026-09-14)

**Result: not found.** `ECHONET_ROOT` unset. Paths tried (all missing):

- `D:\data\EchoNet-Dynamic`, `D:\EchoNet-Dynamic`, `D:\Datasets\EchoNet-Dynamic`
- `E:\data\EchoNet-Dynamic`, `E:\Datasets\EchoNet-Dynamic`, `E:\EchoNet-Dynamic`, `E:\Projects\data\EchoNet-Dynamic`
- `<repo>\data\EchoNet-Dynamic` (only `data/demo` + `data/examples` present)
- `C:\data\EchoNet-Dynamic`, `C:\Datasets\EchoNet-Dynamic`
- `%USERPROFILE%\data\EchoNet-Dynamic`, `%USERPROFILE%\Datasets\EchoNet-Dynamic`, `%USERPROFILE%\EchoNet-Dynamic`
- Downloads/Documents quick name filter for EchoNet/FileList: none

**Action:** obtain AIMI EchoNet-Dynamic → set `ECHONET_ROOT` → `scripts/build_echonet_manifest.py --subset-5000` → `run_protocol.py --paper --experiments R0` (requires hub weights).

## Commands verified

```text
python -m unittest discover -s tests -v
python scripts/validate.py --skip-eval
python scripts/compare_official_b0.py --dry-run
python scripts/plot_risk_coverage.py --demo-curve
```

## Still blocked on external assets

1. **EchoNet-Dynamic** (FileList + Videos + optional VolumeTracings)
2. **Official EchoCLIP hub weights** (`hf-hub:mkaichristensen/echo-clip` or local `.pt`)
3. **CardiacCLIP** weights for comparison numbers
4. **GPU + 5-seed paper runs** for Table-1 mean±SD

Clinical EF MAE/AUC remain **待补充**. Demo metrics must never be reported as clinical.
