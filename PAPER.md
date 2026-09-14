# EchoCLIP-TC / EchoCLIP-TA paper protocol

This document locks the experiment IDs, commands, and honesty rules for the
**EchoCLIP-TA** (EF-aware parameter-efficient temporal adaptation) paper path
in this repository (repo / package name remains EchoCLIP-TC / `echoclip`).

**Demo ≠ clinical.** Numbers from `data/demo/` or `--demo` must never be
reported as EchoNet or Nature Medicine EF MAE. Do **not** invent clinical MAE.

## Cite

- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for echocardiogram interpretation.* Nature Medicine (2024).
- Official inference / prompts: [echonet/echo_CLIP](https://github.com/echonet/echo_CLIP)
- EchoNet-Dynamic (and related AIMI sets): Stanford AIMI **non-commercial** terms — obtain separately.
- EchoJEPA preprint: arXiv:2602.02603; multiview video-CLIP: arXiv:2504.18800
- This repo: https://github.com/Coucou2016/EchoCLIP-TC

## Experiment matrix (R0 / R0U16 / R1–R6 + Oracle-EDES)

Primary VAL/TEST sampling is **uniform-16** (or `val_sample_strategy` from config),
except **R0 under `--paper`** which uses `official_stride` (`0:min(40,T):2`) **without pad-to-16**.
`ed_es` / `mixed` on VAL/TEST hard-fail unless the experiment is **Oracle-EDES**.



| ID | Alias | What | Train | Pool | Eval sample | Calibrate |

|----|-------|------|-------|------|-------------|-----------|

| **R0** | B0 | EchoCLIP-based zero-shot (frames → top-20% median EF). Under `--paper`: `official_stride` **without pad-to-16**; prefer `scripts/eval_official_r0.py`. `official_reproduction_verified` only after parity | No | `frames` | uniform / **official_stride (`--paper`)** | No | zeroshot |

| **R0U16** | R0-U16 | Uniform-16 zero-shot ablation vs official R0 | No | `frames` | **uniform** | No | zeroshot |

| **R1** | M1 | Uniform-16 mean pool (no extra params) | No | `mean` | **uniform** | No | zeroshot |

| **R2** | S0 | Frozen mean + linear/ridge EF head | Yes | supervised | **uniform** | No | **direct_regression** |

| **R3** | S1 | Frozen mean + MLP EF head | Yes | supervised | **uniform** | No | **direct_regression** |

| **R4** | S2 | Temporal aggregator + direct L1/Huber EF | Yes | supervised | **uniform** | No | **direct_regression** |

| **R5** | M2 | EF-label-supervised temporal adaptation (contrastive) of **frozen** EchoCLIP — **not** a zero-shot temporal extension | Yes | `temporal` | **uniform** (mixed = EDES ablation) | No | zeroshot |

| **R6** | M4 | R5 + val-fit affine_logistic / conformal (paper default) | Yes (reuse R5) | `temporal` | **uniform** | Yes (VAL only) | zeroshot |

| **ORACLE_EDES** | Oracle-EDES | Annotation-assisted upper bound (ED/ES indices) — **label clearly; not primary** | No | `mean` | **ed_es** | No | zeroshot |



Definitions live in `echoclip/protocol.py`. Runner: `scripts/run_protocol.py`.  

Table aggregate: `scripts/write_protocol_table.py` → `checkpoints/protocol/comparison.{json,md}`.



### R0 / B0 naming and `--paper`



- **Default (non-paper):** report as **"EchoCLIP-based zero-shot baseline"** until golden parity with official echo_CLIP is proven.

- **`--paper` / `--official-reproduction`:** hard-fails **everywhere** (train / eval_clinical / run_protocol / run_seeds) unless official weights load successfully:

  - EF prompt grid `0..100` step 1

  - Prefer `official_stride` frame selection (`0:min(40,T):2`)

  - Prefer open_clip preprocess when hub load succeeds

  - `allow_scratch_fallback=False` → **RuntimeError** / non-zero exit if official weights are missing

  - Incompatible with `--demo`, `--no-official`, `simple_cnn`, and `ECHOCLIP_SKIP_HUB=1`

  - Prefer `scripts/eval_official_r0.py` for the dedicated official path
  - `official_reproduction_verified=true` only when formula parity + hub load OK
    (set `ECHOCLIP_OFFICIAL_PARITY_OK=1`); otherwise `paper_mode=true`, verified=false
  - Still document remaining gaps (tokenizer quirks, crop zoom / 640→224 vs direct 224, BGR vs RGB, dtype) — do not claim bit-exact parity
  - Golden aggregation tests: `tests/test_official_b0_parity.py` (fixed tensors vs official utils)
  - Optional AVI+hub compare: `scripts/compare_official_b0.py`
  - **R0U16**: uniform-16 control ablation (never conflate with official stride R0)



### M2 / R5 framing and default train path



R5 is **parameter-efficient / EF-aware temporal adaptation** of frozen EchoCLIP towers —

**not** “the first temporal EchoCLIP.” Novelty is protocol fairness + PEFT-style temporal

module + calibration, not foundation-scale priority claims.



**Default R5 train path:**

- `EFSoftContrastiveLoss` **on** (disable with `--no-ef-soft-contrastive`)

- **EF-only** captions (primary). EDV dilation captions are opt-in via `--use-edv-captions`

  (builder: `--include-dilation`).



### Commands



```powershell

# Set roots via env (do not hard-code machine paths)

$env:ECHONET_ROOT = "<AIMI_EchoNet-Dynamic>"

$env:ECHOCLIP_ROOT = (Get-Location).Path



python scripts\build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000



python scripts\run_protocol.py --list



# Demo wiring only (NOT clinical)

python scripts\run_protocol.py --demo --experiments R0,R1 --vision-backbone simple_cnn



# Paper path (hard-fails without official weights)

python scripts\run_protocol.py --paper --experiments R0



# Primary + supervised + temporal matrix

python scripts\run_protocol.py --experiments R0,R1,R2,R3,R4,R5,R6



# Multi-seed helper (default seeds 0,1,2; use 0–4 for paper)

python scripts\run_seeds.py --demo --experiments R0,R1 --seeds 0,1,2 --vision-backbone simple_cnn

# Paper: python scripts\run_seeds.py --paper --experiments R0 --seeds 0,1,2,3,4



# Legacy aliases still work

python scripts\run_protocol.py --experiments B0,M1,M2,M4



# Oracle (annotation-assisted) — opt-in

python scripts\run_protocol.py --experiments ORACLE_EDES

```



### Calibration honesty



Threshold scores use a **pseudo-logit** `(threshold - pred_EF)`, not true classifier

logits. Temperature scaling is therefore limited. Prefer

`--calibration-method affine_logistic` for P(EF&lt;50/40/30).



**Basic split conformal** uses a fixed absolute residual quantile → **constant interval

width** (width-based abstention is vacuous; probability abstention is used instead).

Optional **adaptive conformal** (`--adaptive-conformal`): normalized score `|y-ŷ|/s(x)`

with heuristic or learned positive scale head → variable width + risk–coverage / AURC.

Paired bootstrap CIs for ΔMAE and bootstrap CIs for RMSE/R²/AUCs are emitted in

`summarize_clinical` / `metrics.json`.



## Honesty rules



1. Never report demo MAE/AUC as clinical results; never invent MAE numbers.

2. Fit temperature / conformal / affine logistic **only** on VAL; never retune on TEST.

3. State `load_source` (hub vs local vs scratch) next to any table number.

4. State which split (TEST vs subset_5000) and seed (multi-seed: mean±SD; paper prefers 5 seeds).

5. `simple_cnn` / missing hub = plumbing only.

6. Oracle-EDES must be labeled annotation-assisted in every table.

7. `--pool temporal` hard-fails when the model has no temporal aggregator.

8. `--paper` must hard-fail in train and eval if weights are not official.



## Remaining gaps (need real assets)



| Gap | Needed |

|-----|--------|

| EchoNet-Dynamic videos + FileList | AIMI download |

| Official EchoCLIP weights | open_clip hub or local `.pt` |

| CardiacCLIP comparison | external weights (`echoclip/cardiacclip_stub.py`; no invented numbers) |

| GPU + `convnext_base` | paper-scale R0/R5 |

| Environment lock | `requirements-lock.txt` + `requirements.txt` |



## Primary metric script



`scripts/eval_clinical.py` (EF MAE/RMSE/R², AUC@50/40/30, ECE/Brier@50/40/30,

conformal, optional adaptive conformal / AURC, bootstrap CIs, abstention).  

`scripts/eval.py` retrieval R@k is diagnostic only — not Table 1.
