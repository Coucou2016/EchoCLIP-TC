# Review response — round 2 (2026-09-14)

Peer review P0 items after prior VAL/TEST ED/ES / framing fixes.

| P0 | Issue | Fix | Status |
|----|-------|-----|--------|
| **P0-1** | R2–R4 supervised eval used `--pool mean` (zero-shot prompts); R2/R3 ckpts lacked `model_state` → `load_checkpoint` KeyError; R4 head in `train_cfg["head_state"]` never rebuilt | Unified schema (`format_version`, `task=ef_regression`, `head_kind`, `head_state_dict`, `temporal_state_dict`, `model_config`, `train_seed`, …); `save_supervised_ef_checkpoint` / `load_supervised_ef_checkpoint` / `predict_direct_ef`; `eval_clinical.py --prediction-mode zeroshot\|direct_regression\|auto`; protocol R2/R3/R4 → `direct_regression`, R5 → `zeroshot`; E2E tests in `tests/test_supervised_e2e.py` (train-style save→reload→finite preds; assert no zero-shot prompt pack) | **Done** |
| **P0-2** | Multi-seed not real (seed not passed into train) | `--seed` CLI on `train.py` / `train_supervised.py`; `run_protocol` / `run_seeds` pass seed; `train_seed` in ckpt + metrics; test two seeds differ; docs: R0/R1 deterministic once weights+sampling fixed | **Done** |
| **P0-3** | Official R0 blocked: protocol always forced uniform; official stride padded to 16 | `paper_eval_sample_strategy=official_stride` for R0; `resolve_eval_sample_strategy(..., paper=)`; dataset skips pad-or-trim for official_stride; `R0U16` ablation; `scripts/eval_official_r0.py`; `official_reproduction_verified` only after parity (`ECHOCLIP_OFFICIAL_PARITY_OK` / compare script); golden + compare improvements | **Done** (bit-exact AVI+hub still **Blocked** without assets) |
| **P0-4** | Manuscript ChatGPT/MCP/imitation/private URLs; cycle-aware zero-shot framing | Cleaned `papers/echoclip_tc_manuscript.md` (+ HTML sync); process notes → `notes/`; retitled toward **EchoCLIP-TA**; PAPER.md/README naming | **Done** |

## P1 (same pass)

| Item | Status |
|------|--------|
| Primary train sampling uniform (mixed = R5-EDEStrain ablation) | Done |
| EFSoft: finite EF mask (don’t drop batch); clarify `soft_weight` | Done |
| `view_weight=0` default (unchanged; two_views only if >0) | Done |
| Paper default calibration `affine_logistic` | Done (`run_protocol --paper`) |
| Adaptive conformal labeled empirical heuristic when same-VAL fit | Done (metrics + calibrate docs) |
| Bootstrap AUC: stratified + filter NaN replicates | Done |
| `requirements-paper.lock` + transformers lock vs requirements.txt | Done |
| LICENSE SPDX-style + ATTRIBUTION boundary | Done |
| Cite EchoJEPA (arXiv:2602.02603) + JACC Asia / multiview line | Done (verified via web) |

## Honesty / remaining gaps

- **No invented clinical MAE.**
- Official R0 end-to-end bit-exact vs echonet/echo_CLIP on real EchoNet AVI + hub: **Blocked** (needs AIMI data + hub).
- Paper 5-seed clinical tables: **Blocked** (GPU + EchoNet).
- `official_reproduction_verified` defaults **false** under `--paper` until parity gate.

## Tests / gates

```text
python -m unittest discover -s tests -v
python scripts/validate.py --skip-eval
```

## Key files

- `echoclip/checkpoint.py` — supervised schema + load/predict
- `scripts/train_supervised.py`, `scripts/eval_clinical.py`, `scripts/run_protocol.py`, `scripts/train.py`
- `echoclip/protocol.py` — R0U16, prediction_mode, paper official_stride
- `echoclip/data.py` — no pad-to-16 for official_stride; variable-T collate
- `scripts/eval_official_r0.py`, `scripts/compare_official_b0.py`
- `tests/test_supervised_e2e.py`
- `papers/echoclip_tc_manuscript.md`, `PAPER.md`, `requirements-paper.lock`, `LICENSE`
