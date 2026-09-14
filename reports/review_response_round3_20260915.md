# Review response — round 3 (2026-09-15)

Peer-review **new P0 closure** + high-ROI P1 after re-review of public `main`.
**No clinical MAE invented.**

## Push status

| Field | Value |
|-------|-------|
| **Push** | Succeeded via GitHub git Data API (`git push` HTTPS :443 unreachable) |
| **Remote** | https://github.com/Coucou2016/EchoCLIP-TC |
| **Branch** | `main` |
| **Content commit SHA** | `5b47c36e332aef9f51314acc40961679fbd026fe` |
| **When** | 2026-09-15 |

After checklist-only follow-up commits, read the live tip with:
`gh api repos/Coucou2016/EchoCLIP-TC/commits/main --jq .sha`

## New P0 → Done

| P0 | Issue | Fix | Status |
|----|-------|-----|--------|
| **P0-1** | R2–R4 eval used zero-shot prompts via `--pool mean`; supervised ckpts lacked usable `model_state` | Unified supervised schema (`format_version`, `task=ef_regression`, `head_kind`, `head_state_dict`, optional `temporal_state_dict`, `model_config`, `train_seed`); `load_supervised_ef_checkpoint` / `predict_direct_ef`; `eval_clinical --prediction-mode zeroshot\|direct_regression\|auto`; R2–R4 → `direct_regression`, R5 → `zeroshot`; E2E tests assert finite EF preds + **no** zero-shot prompt pack | **Done** |
| **P0-2** | Multi-seed not wired into train | `--seed` on `train.py` / `train_supervised.py`; `run_protocol` `_train_m2` / `_train_supervised` pass `--seed`; `train_seed` + `eval_seed` in ckpt / `metrics.json`; toy test: different seeds → different `train_seed` | **Done** |
| **P0-3** | Official R0 forced uniform / pad-to-16 | `R0` / `R0_OFFICIAL` vs `R0U16`; paper R0 → `official_stride` without pad-to-16; `scripts/eval_official_r0.py` open_clip + 224 AVI + `0:min(40,T):2` + EF 0–100 + top-20% median; `run_protocol --paper R0` routes to official script (bypasses generic Dataset when hub+AVI work) | **Done** (end-to-end bit-exact still **Blocked** without hub+EchoNet) |
| **P0-4** | `official_reproduction_verified` auto-true under `--paper` | `paper_mode` separate; verified only after parity gate (`ECHOCLIP_OFFICIAL_PARITY_OK` / formula+hub) | **Done** |
| **P0-5** | Manuscript process noise / wrong framing | Stripped ChatGPT / dual-agent / MCP / imitation / Nature-family follow-on notes from `papers/echoclip_tc_manuscript.md` (+ HTML); process notes in `notes/` / `reports/`; retitle toward **EchoCLIP-TA** / EF-aware PEFT; PAPER.md/README naming | **Done** |

## P1 (same pass)

| Item | Status |
|------|--------|
| Primary TRAIN uniform/random; optional `R5_EDESTRAIN` ablation | **Done** (`view_weight` default 0.0 in `echonet_dynamic.yaml`) |
| EFSoft finite-EF mask; clarify `soft_weight` | **Done** |
| Paper default calibration `affine_logistic` | **Done** |
| Adaptive conformal labeled empirical heuristic / VAL-scale note | **Done** |
| Stratified + NaN-filter bootstrap for AUC | **Done** |
| `requirements-lock` transformers pin + `requirements-paper.lock` | **Done** |
| Cite JACC Asia 2026 multiview VLM + EchoJEPA preprint | **Done** |

## Tests / gates

```text
python -m unittest discover -s tests -v   # 100 tests OK
python scripts/validate.py --skip-eval   # OK
python scripts/eval_official_r0.py --dry-run
```

## Remaining blockers (external)

1. EchoNet-Dynamic AIMI data on disk  
2. Official EchoCLIP hub weights for bit-exact R0  
3. GPU + 5-seed paper tables  
4. CardiacCLIP external weights for comparison numbers  

Clinical EF MAE/AUC remain **待补充**. Demo metrics are never clinical.
