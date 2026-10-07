# Attribution

## EchoCLIP (Nature Medicine 2024)

Architecture and clinical prompt templates are derived from the published EchoCLIP work
and the official inference repository:

- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for
  echocardiogram interpretation.* Nature Medicine, 2024.
  https://doi.org/10.1038/s41591-024-02959-y
- Code: https://github.com/echonet/echo_CLIP

EchoNet-Dynamic, if used, is licensed separately by Stanford AIMI (non-commercial
research) and is not bundled in this repository.

## File-level provenance (upstream-derived vs original)

Per-file origin, whether upstream source was read, line-level similarity, and license
status are inventoried in `PROVENANCE.md`.

### Upstream-derived / pattern-following (not claimed as original clinical IP)

| Path | Relationship |
|------|--------------|
| `echoclip/prompts.py` | Prompt strings follow the published echonet/echo_CLIP prompt patterns (parity path); verbatim upstream clinical sentences, **not MIT** |
| `echoclip/official_parity.py` | Self-described "bit-aligned" reference logic for the upstream echonet/echo_CLIP utils.py (`compute_regression_metric`, `crop_and_scale`); upstream-aligned — **needs author decision** |
| `echoclip/preprocess.py` | Echo-specific crop / frame IO aligned with published EchoCLIP preprocessing notes |
| `echoclip/structured_text.py` | Fills official or clean-room TA captions from measurements; uses upstream prompt strings by import |
| Official hub weights `hf-hub:mkaichristensen/echo-clip` | **Never redistributed** here; loaded at runtime when permitted |

### Original scaffold (this repository)

| Path | Notes |
|------|-------|
| `echoclip/text.py` | `echoclip/text.py` is an independently authored report normalizer whose behavior intentionally overlaps upstream EchoCLIP report cleaning; no upstream code is vendored or line-copied in this repository, and whether the upstream echonet/echo_CLIP utils.py source was consulted is **unclear — needs author confirmation** (canonical wording; see `PROVENANCE.md`) |
| `echoclip/temporal.py` | Attention pool + Temporal Transformer on frozen embeddings |
| `echoclip/calibrate.py` | Temperature, affine logistic, split / adaptive conformal, AURC |
| `echoclip/clinical.py` | EF regression metrics, bootstrap CIs, paired ΔMAE |
| `echoclip/protocol.py` | Locked R0–R6 + Oracle-EDES experiment matrix |
| `echoclip/supervised.py` | S0–S2 supervised baseline heads |
| `echoclip/loss.py` (`EFSoftContrastiveLoss`) | EF-aware soft multi-positive contrastive |
| `echoclip/cycle_sample.py` | Cycle-aware frame sampling (`official_stride` mirrors the upstream stride rule) |
| `echoclip/cardiacclip.py` (+ stub alias) | Comparison adapter + download instructions (external weights) |
| `echoclip/prompts_ta.py` | Clean-room EF/dilation captions for optional TA training |
| `echoclip/efficiency.py` | Trainable param / timing helpers for metrics.json |
| `scripts/train.py`, `scripts/eval_clinical.py`, `scripts/run_protocol.py`, `scripts/run_seeds.py`, `scripts/run_paper_matrix.py`, `scripts/run_label_efficiency.py` | Training / eval / matrix / label-efficiency runners |
| `scripts/analyze_attention_edes.py` | Attention / ED–ES figure + CSV (demo complete; EchoNet path blocked on assets) |
| `tests/` | Unit tests for protocol, calibration, fairness guards |

## Provenance risk / license boundary (important)

**Academic Software License boundary:** Upstream EchoCLIP (Cedars-Sinai / echonet)
materials and Stanford AIMI datasets are under **non-MIT / Academic Software /
non-commercial** terms. The MIT `LICENSE` in this repository covers **original scaffold
code written here** only. It does **not** re-license:

- Upstream-derived prompt strings and any upstream-aligned logic
- Official EchoCLIP weights (never redistributed here)
- EchoNet / CAMUS / related clinical datasets
- Any verbatim upstream source files (none are vendored in this repository)

**Do not falsely expand the MIT claim** to cover upstream prompts, weights, or AIMI data.
Prefer linking to upstream licenses rather than copying restricted text.

Neither the official EchoCLIP weights nor EchoNet-Dynamic data are redistributed in this
repository; users must obtain them under their own upstream terms
(`docs/OBTAIN_DATA_AND_WEIGHTS.md`).

If you believe material in this repository is upstream-derived and should be handled
differently, please contact the maintainers — we will clean-room rewrite or remove it,
or re-license/label it correctly.

This section is an honest, conservative statement and **not** a legal conclusion; no
legal review has occurred. Recommended pre-submission action: a per-file provenance
audit with the author's institution (see `PROVENANCE.md`).

## CLIP / OpenCLIP

- Radford et al., CLIP (ICML 2021)
- Text tokenizer: OpenAI CLIP via Hugging Face `transformers`
- Optional weight init: OpenCLIP LAION checkpoints (`open-clip-torch`)

## CardiacCLIP (external comparator)

- Du, Guo & Li. CardiacCLIP. MICCAI 2025. arXiv:2509.17065
- https://github.com/xmed-lab/CardiacCLIP
- Not bundled; see `echoclip/cardiacclip_stub.py`

## This repository

Original code listed under "original scaffold (this repository)" is MIT-licensed (see
`LICENSE`), subject to the provenance caveat above. It is an independent
training/inference scaffold — not the Cedars-Sinai production release and not a
redistribution of official EchoCLIP weights.

Public code: https://github.com/Coucou2016/EchoCLIP-TC
