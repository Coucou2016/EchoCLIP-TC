# Attribution



## EchoCLIP (Nature Medicine 2024)



Architecture and clinical prompt templates are derived from the published EchoCLIP work and the official inference repository:



- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for echocardiogram interpretation.* Nature Medicine, 2024. https://doi.org/10.1038/s41591-024-02959-y

- Code: https://github.com/echonet/echo_CLIP



EchoNet-Dynamic, if used, is licensed separately by Stanford AIMI (non-commercial research) and is not bundled in this repository.



## File-level provenance (upstream-derived vs clean-room)



### Upstream-derived / pattern-following (not claimed as original clinical IP)



| Path | Relationship |

|------|----------------|

| `echoclip/prompts.py` | Prompt strings follow echonet/echo_CLIP `prompts_used.json` patterns |

| `echoclip/text.py` (report-cleaning regexes) | Cleaning patterns inspired by echonet/echo_CLIP `utils.py` |

| `echoclip/preprocess.py` | Echo-specific crop / frame IO aligned with published EchoCLIP preprocessing notes |

| `echoclip/structured_text.py` | Fills **official** EF/dilation templates from EchoNet measurements; no new clinical wording |

| Official hub weights `hf-hub:mkaichristensen/echo-clip` | **Never redistributed** here; loaded at runtime when permitted |



### Clean-room / original scaffold (this repository)



| Path | Notes |

|------|--------|

| `echoclip/temporal.py` | Attention pool + Temporal Transformer on frozen embeddings |

| `echoclip/calibrate.py` | Temperature, affine logistic, split / adaptive conformal, AURC |

| `echoclip/clinical.py` | EF regression metrics, bootstrap CIs, paired ΔMAE |

| `echoclip/protocol.py` | Locked R0–R6 + Oracle-EDES experiment matrix |

| `echoclip/supervised.py` | S0–S2 supervised baseline heads |

| `echoclip/loss.py` (`EFSoftContrastiveLoss`) | EF-aware soft multi-positive contrastive |

| `echoclip/cycle_sample.py` | Cycle-aware frame sampling |

| `echoclip/cardiacclip_stub.py` | Comparison interface only (external weights required) |

| `scripts/train.py`, `eval_clinical.py`, `run_protocol.py`, `run_seeds.py` | Training / eval / multi-seed protocol runners |

| `scripts/analyze_attention_edes.py` | Toy attention / ED–ES analysis skeleton |

| `tests/` | Unit tests for protocol, calibration, fairness guards |



## Provenance risk / license boundary (important)



**Academic Software License risk:** Official EchoCLIP (Cedars-Sinai / echonet) materials and Stanford AIMI datasets may be under **non-MIT / Academic Software / non-commercial** terms. The MIT `LICENSE` in this repository covers **original scaffold code written here** only. It does **not** automatically re-license:



- Upstream-derived prompt templates and cleaning regexes

- Official EchoCLIP weights (never redistributed here)

- EchoNet / CAMUS / related clinical datasets

- Any verbatim upstream source files (none should be vendored without audit)



**Do not falsely expand the MIT claim** to cover upstream prompts, weights, or AIMI data. Prefer linking to upstream licenses rather than copying restricted text. A formal Academic Software License notice may be required before redistribution of any upstream-derived artifacts beyond fair-use documentation.



## CLIP / OpenCLIP



- Radford et al., CLIP (ICML 2021)

- Text tokenizer: OpenAI CLIP via Hugging Face `transformers`

- Optional weight init: OpenCLIP LAION checkpoints (`open-clip-torch`)



## CardiacCLIP (external comparator)



- Du, Guo & Li. CardiacCLIP. MICCAI 2025. arXiv:2509.17065

- https://github.com/xmed-lab/CardiacCLIP

- Not bundled; see `echoclip/cardiacclip_stub.py`



## This repository



Implementation code listed under “clean-room / original scaffold” is MIT-licensed (see `LICENSE`), subject to the provenance caveat above. It is an independent training/inference scaffold—not the Cedars-Sinai production release and not a redistribution of official EchoCLIP weights.



Public code: https://github.com/Coucou2016/EchoCLIP-TC
