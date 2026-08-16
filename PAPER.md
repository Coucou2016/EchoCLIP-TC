# EchoCLIP-TC paper protocol

This document locks the experiment IDs, commands, and honesty rules for the
EchoCLIP-TC (Temporal, Calibrated) paper path in this repository.

**Demo ≠ clinical.** Numbers from `data/demo/` or `--demo` must never be
reported as EchoNet or Nature Medicine EF MAE.

## Cite

- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for echocardiogram interpretation.* Nature Medicine (2024).
- Official inference / prompts: [echonet/echo_CLIP](https://github.com/echonet/echo_CLIP)
- EchoNet-Dynamic (and related AIMI sets): Stanford AIMI **non-commercial** terms — obtain separately.

## Experiment matrix (Table 1 path)

| ID | What | Train | Pool | Calibrate |
|----|------|-------|------|-----------|
| **B0** | Official EchoCLIP zero-shot (hub or local fallback) | No | `frames` (per-frame → top-20% median EF) | No |
| **M1** | Cycle / uniform sampling + **mean** pool (no extra params) | No | `mean` | No |
| **M2** | Temporal aggregator on frozen towers | Yes | `temporal` | No |
| **M4** | M2 + val-fit temperature & split-conformal | Yes (reuse M2) | `temporal` | Yes (VAL only) |

**B0 vs M1 (intentional):** B0 keeps frame embeddings `(T,D)` and averages *ranked EF values* across frames inside the official top-20% median aggregator. M1 mean-pools embeddings to one `z_v` *before* that aggregator (same path as M2’s single video vector). These are **not** identical; M1 is the no-parameter video-vector ablation, not “mean of per-frame EF scalars.” Optional scalar-mean ablation can be added later as M1b if needed — do not silently redefine M1.

Definitions live in `echoclip/protocol.py`. Runner: `scripts/run_protocol.py`.  
Table aggregate: `scripts/write_protocol_table.py` → `checkpoints/protocol/comparison.{json,md}`.

### Commands (EchoNet-Dynamic)

```powershell
# 1. Build manifests + lock seed-42 subset IDs
python scripts\build_echonet_manifest.py `
  --echonet-root E:\data\EchoNet-Dynamic `
  --subset-5000

# 2. List modes
python scripts\run_protocol.py --list

# 3. B0 baseline (official hub when available)
python scripts\run_protocol.py --experiments B0

# 4. M1 mean-pool ablation
python scripts\run_protocol.py --experiments M1

# 5. Train temporal (M2) then calibrated eval (M4)
python scripts\run_protocol.py --experiments M2,M4

# Or full matrix
python scripts\run_protocol.py --experiments B0,M1,M2,M4
```

Comparable outputs:

```
checkpoints/protocol/B0/metrics.json
checkpoints/protocol/M1/metrics.json
checkpoints/protocol/M2/metrics.json   # also M2/best.pt
checkpoints/protocol/M4/metrics.json
checkpoints/protocol/summary.json
checkpoints/protocol/comparison.json   # cross-ID table
checkpoints/protocol/comparison.md
```

Rebuild the comparison table without re-running eval:

```powershell
python scripts\write_protocol_table.py --print
```

### B0 → reproduce EchoCLIP external ~7.1% EF MAE

Paper-style external protocol (you must have real data + official weights):

1. Weights: `hf-hub:mkaichristensen/echo-clip` via open_clip (`load_source` in metrics must show the hub id, **not** `scratch_fallback` / random `simple_cnn`).
2. Subset: `data/echonet_dynamic/subset_5000.json` drawn with **seed 42**; locked IDs in `subset_5000_ids.json` / `.txt`.
3. Also report **full TEST** (`test.json`) — do not substitute demo or VAL-tuned numbers.
4. Eval:

```powershell
python scripts\eval_clinical.py `
  --config configs\echonet_dynamic.yaml `
  --init-official `
  --manifest data\echonet_dynamic\subset_5000.json `
  --manifest-dir E:\data\EchoNet-Dynamic `
  --pool frames --video-frames 16 --sample-strategy uniform `
  --experiment-id B0 `
  --output checkpoints\protocol\B0\metrics_subset5000.json

python scripts\eval_clinical.py `
  --config configs\echonet_dynamic.yaml `
  --init-official `
  --manifest data\echonet_dynamic\test.json `
  --manifest-dir E:\data\EchoNet-Dynamic `
  --pool frames --video-frames 16 --sample-strategy uniform `
  --experiment-id B0 `
  --output checkpoints\protocol\B0\metrics_test.json
```

This repo does **not** invent or hard-code 7.1%. That figure is from Christensen et al.; your run may differ slightly with frame sampling / software stack.

### Demo / Windows CPU plumbing (not clinical)

```powershell
python scripts\make_demo_data.py
python scripts\run_protocol.py --demo --experiments B0,M1 `
  --vision-backbone simple_cnn --video-frames 4 --batch-size 4
# Optional M2/M4 smoke (1 epoch):
python scripts\run_protocol.py --demo --experiments M2,M4 `
  --vision-backbone simple_cnn --video-frames 4 --epochs 1 --batch-size 4
```

## Cross-dataset wiring

Builders fail clearly when data is absent:

```powershell
python scripts\build_public_echo_manifest.py --dataset camus --root E:\data\CAMUS
python scripts\build_public_echo_manifest.py --dataset echonet_pediatric --root E:\data\EchoNet-Pediatric
python scripts\build_public_echo_manifest.py --dataset echonet_lvh --root E:\data\EchoNet-LVH
python scripts\build_public_echo_manifest.py --dataset echonet_dynamic --root E:\data\EchoNet-Dynamic
```

Config stubs: `configs/camus.yaml`, `configs/echonet_pediatric.yaml`, `configs/echonet_lvh.yaml`.

After manifests exist, point `run_protocol.py --config` / `--test-manifest` at those paths (same B0–M4 IDs).

## Honesty rules

1. Never report demo MAE/AUC as clinical results.
2. Fit temperature / conformal **only** on VAL (`cal_manifest`); never retune on TEST.
   Non-demo M4 hard-fails if `cal_manifest` resolves to the same path as the test
   manifest (`run_protocol.py`).
3. State `load_source` (hub vs local vs scratch) next to any table number.
4. State which split (TEST vs subset_5000) and seed.
5. `simple_cnn` / missing hub = plumbing only on this Windows CPU path.
6. `--pool temporal` hard-fails when the loaded model has no temporal aggregator
   (avoids silently mean-pooling under an M2/M4 label).

## Comparison table (after runs)

Aggregate existing `checkpoints/protocol/<ID>/metrics.json` into one table:

```powershell
python scripts\write_protocol_table.py
python scripts\write_protocol_table.py --print
```

Also written automatically at the end of a non-dry-run `run_protocol.py` pass:

```
checkpoints/protocol/comparison.json
checkpoints/protocol/comparison.md
checkpoints/protocol/summary.json
```

`--dry-run` does **not** overwrite `summary.json` or comparison files.

## Remaining gaps (need real assets)

| Gap | Needed |
|-----|--------|
| EchoNet-Dynamic videos + FileList | AIMI download |
| Official EchoCLIP weights | open_clip hub or local `.pt` |
| GPU + `convnext_base` | paper-scale B0/M2 |
| CAMUS / Pediatric / LVH on disk | cross-dataset tables |
| NIfTI/MHD readers for some CAMUS dumps | convert or extend preprocess |

## Primary metric script

`scripts/eval_clinical.py` (EF MAE/RMSE/R², AUC@50/40/30, ECE, Brier, conformal, abstention).  
`scripts/eval.py` retrieval R@k is diagnostic only — not Table 1.
