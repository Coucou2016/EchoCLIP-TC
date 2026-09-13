# Review response — P1 complete (2026-09-14)

Peer-review **remaining** items beyond P0, mapped to code/doc changes in
https://github.com/Coucou2016/EchoCLIP-TC. **No clinical MAE numbers invented.**

## Push status

| Field | Value |
|-------|-------|
| **Push** | Succeeded (`git push origin main`) |

| **Remote** | `https://github.com/Coucou2016/EchoCLIP-TC.git` |

| **Branch** | `main` |

| **Remote SHA** | `80694c6ce6a3fd94bc59091c153c93134387f53f` |

| **Range** | `7241ece..80694c6` |

| **When** | 2026-09-14 |

## Major Concern → Status

| Concern | Status | Where |
|---------|--------|-------|
| **Annotation-assisted sampling on VAL/TEST** | Done (P0) | `protocol.py` + eval/protocol hard-fail |
| **Oracle / annotation-assisted upper bound** | Done (P0) | `ORACLE_EDES` |
| **B0 / official reproduction path** | Done (P0+P1) | `--paper` hard-fail in **train** + eval + protocol + seeds |
| **M2 misframed as zero-shot temporal** | Done (P0+P1) | R5 = EF-supervised PEFT; not “first temporal EchoCLIP” |
| **Primary captions EF-only** | Done (P0+P1) | Dataset default filters dilation; `--use-edv-captions` opt-in |
| **EF soft contrastive as R5 default train** | Done | `train.py` default ON for temporal; `run_protocol` wires R5; `--no-ef-soft-contrastive` opt-out |
| **Supervised baselines S0–S2** | Done (P0) | R2–R4 |
| **R0–R6 matrix** | Done (P0) | `protocol.py` / PAPER.md |
| **Temporal mask / subsample** | Done (P0) | `temporal.py` |
| **Dataset set_epoch** | Done (P0) | `data.py` |
| **Calibration @40/@30 reporting** | Done | `summarize_clinical`: ECE/Brier/affine or T for 50/40/30 |
| **Pseudo-logit / affine logistic** | Done (P0+P1) | documented; affine preferred |
| **Adaptive conformal (normalized)** | Done | `calibrate.py` + `--adaptive-conformal`; fixed-width note retained |
| **Risk–coverage / AURC** | Done | `risk_coverage_curve` / `area_under_risk_coverage` |
| **Paired bootstrap ΔMAE + RMSE/R²/AUC CIs** | Done | `paired_bootstrap_delta_mae` + bootstrap keys in clinical summary |
| **Multi-seed helper** | Done | `scripts/run_seeds.py` (default 0,1,2; doc 5 for paper) |
| **Paper vs demo CLI hard-fail** | Done | train/eval/protocol/seeds; PAPER.md |
| **requirements-lock + strip E:\\ paths** | Done | `requirements-lock.txt`; configs/docs/scripts use env placeholders |
| **LICENSE / ATTRIBUTION provenance** | Done | File-level upstream vs clean-room; ASL risk; no expanded MIT claim |
| **Manuscript + research_report R0–R6** | Done | `papers/echoclip_tc_manuscript.md`, `reports/research_report.md` |
| **CardiacCLIP comparison** | Partial | `echoclip/cardiacclip_stub.py` — requires external weights; **待补充** numbers |
| **Attention / ED–ES analysis** | Partial | `scripts/analyze_attention_edes.py` toy skeleton |
| **Golden bit-exact EchoCLIP parity** | Blocked | Remaining tokenizer/crop/dtype gaps documented |
| **EchoNet clinical MAE table** | Blocked | EchoNet absent → **待补充** |

## Commands verified

```text
python -m unittest discover -s tests -v
python scripts/validate.py --skip-eval
```

## Remaining blockers

1. **EchoNet-Dynamic + official hub weights** — required for any clinical MAE/AUC (诚实: 待补充).
2. **CardiacCLIP** — external weights not bundled; stub only.
3. **Bit-exact EchoCLIP parity** — not claimed.
4. **Paper multi-seed (5)** — helper ready; full paper runs need GPU + data.
