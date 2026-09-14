# EchoCLIP-TA: EF-aware parameter-efficient temporal adaptation and calibrated evaluation on frozen EchoCLIP

**Status:** Methods manuscript draft.  
**Honesty:** Clinical EchoNet-Dynamic EF MAE / AUC numbers are **待补充** until official weights and AIMI data are available locally. Demo / synthetic figures are labeled **DEMO** and are not clinical results.  
**Code:** https://github.com/Coucou2016/EchoCLIP-TC  

**Axes:** `task=manuscript` · `paper_type=methods` · `language=en`.

**One-sentence argument.** We present EchoCLIP-TA (parameter-efficient, EF-aware temporal adaptation on frozen EchoCLIP) with validation-only calibration (temperature / affine logistic, split and optional adaptive conformal), together with a locked R0–R6 (+ R0U16, Oracle-EDES) public-data protocol for video-level ejection-fraction (EF) evaluation; clinical superiority claims remain contingent on EchoNet-Dynamic + official EchoCLIP weights (**待补充**). We do **not** claim to be “the first temporal EchoCLIP” or a cycle-aware zero-shot extension.

---

## Terminology ledger (canonical forms)

| Term | Canonical form | Notes |
|------|----------------|-------|
| Base model | EchoCLIP | Christensen et al., *Nat Med* 2024 |
| This work | EchoCLIP-TA (repo: EchoCLIP-TC) | EF-aware PEFT temporal adaptation + calibrated eval |
| Primary metric | EF MAE | Mean absolute error in EF percentage points |
| Video embedding | \(z_v\) | Video-level representation after pooling |
| Protocol IDs | R0 / R0U16 / R1–R6 + Oracle-EDES (aliases B0/M1/M2/M4, S0–S2) | Locked in `PAPER.md` / `echoclip/protocol.py` |
| Calibration | Temperature / affine logistic; ECE; Brier; split + optional adaptive conformal; bootstrap CIs | Fit on VAL only |
| This work framing | Parameter-efficient / EF-aware temporal adaptation (EchoCLIP-TA) | **Not** “first temporal EchoCLIP”; **not** cycle-aware zero-shot |
| Public data | EchoNet-Dynamic | Stanford AIMI; non-commercial |
| Related VLMs | EchoPrime; CardiacCLIP; EchoJEPA; multiview video-CLIP | Baselines for positioning, not reimplemented here |

---

## Title options

1. **EchoCLIP-TA: EF-aware parameter-efficient temporal adaptation and calibrated evaluation on frozen EchoCLIP** (preferred)
2. A reproducible temporal–calibration protocol for frozen EchoCLIP on public echocardiography videos
3. From frames to calibrated video vectors: EchoCLIP-TA for EF estimation under a locked public protocol

---

## Abstract

**Background.** EchoCLIP aligns echocardiogram frames with clinical text and supports zero-shot estimation of left ventricular ejection fraction (EF), but the published pipeline is primarily frame-centric and reports uncalibrated cosine similarities.

**Methods.** We introduce EchoCLIP-TA: a lightweight temporal aggregator (attention pooling or Temporal Transformer) on frozen EchoCLIP towers, **EF-only** structured captions by default (EDV dilation opt-in), EF soft multi-positive contrastive training for R5, supervised EF heads (R2–R4) evaluated with direct regression (not zero-shot prompts), and validation-only temperature / affine-logistic calibration with split-conformal (and optional normalized adaptive conformal + AURC) EF intervals and abstention. We lock the R0 / R0U16 / R1–R6 matrix (legacy aliases B0/M1/M2/M4; supervised S0–S2 as R2–R4) plus Oracle-EDES for EchoNet-Dynamic. Protocol defaults: \(T=16\) for uniform paths; R0 under `--paper` uses official stride `0:min(40,T):2` without pad-to-16; seed 42; **primary VAL/TEST sampling is uniform** (ed_es/mixed hard-fail except Oracle). R5 is EF-label-supervised PEFT adaptation of frozen towers—not a zero-shot temporal extension.

**Results.** **待补充 (EchoNet-Dynamic + official hub weights).** Locally we only demonstrate end-to-end pipeline smoke tests on synthetic demo pairs (`load_source=scratch_fallback` / `scratch`; demo uses \(T=4\)). Demo MAE/ECE/conformal coverage must not be read as clinical performance.

**Conclusions.** EchoCLIP-TA reframes honest innovation as *parameter-efficient EF-aware temporal adaptation + trustworthy uncertainty + fair public protocol*, without claiming private million-scale pretraining or temporal-first priority. Clinical claims require completing the gated evaluation path (`--paper` hard-fails without official weights; `official_reproduction_verified` only after parity).

**Keywords:** echocardiography; vision–language model; temporal adaptation; PEFT; calibration; conformal prediction; EchoCLIP

---

## 1. Introduction

Echocardiography remains the frontline modality for cardiac structure and function. Vision–language models (VLMs) reduce dependence on task-specific labels by aligning images or videos with report text. EchoCLIP demonstrated that contrastive pretraining on >1M clinical video–text pairs enables zero-shot EF estimation (reported external EF MAE ≈ 7.1% in Christensen et al., *Nature Medicine* 2024; internal MAE ≈ 8.4%) and device recognition (AUCs of 0.84 / 0.92 / 0.97 for pacemaker / mitral repair / aortic valve in the same paper). At clinical EF thresholds on their evaluation, EchoCLIP reported AUCs ≈ 0.89–0.90 (EF < 50%), 0.93–0.94 (EF < 40%), and 0.95–0.97 (EF < 30%)—**literature values only**, not reproduced here.

Two gaps matter for clinical reuse. First, **temporal aggregation**: EchoCLIP’s public inference path emphasizes per-frame encoding with post-hoc aggregation of ranked EF prompts, whereas cardiac function is inherently dynamic; subsequent models (EchoPrime; CardiacCLIP) emphasize multi-view or multi-frame video modeling, but often with different training budgets and evaluation contracts. Second, **calibration**: cosine similarities and prompt-rank EF estimates are rarely reported with expected calibration error (ECE), Brier score, or finite-sample prediction intervals.

EchoCLIP-TA addresses both gaps *without rewriting the dual encoder*: we freeze pretrained towers when available, train only a temporal module for video-level \(z_v\), and fit temperature / conformal procedures exclusively on a validation split. Innovation claims are therefore **protocol- and reliability-centric**, not “another foundation model trained on private 1M reports.”

---

## 2. Related work

**EchoCLIP (Nature Medicine 2024).** Christensen et al. pretrained a frame–text CLIP-style foundation model on >1M echocardiogram–report pairs; zero-shot LVEF via prompt ranking (external EchoNet-Dynamic EF MAE ≈ 7.1%; internal MAE ≈ 8.4%); EchoCLIP-R for long-context retrieval. We treat official weights + prompts as the B0 baseline and do **not** claim private-scale re-pretraining. Code/weights: echonet/echo_CLIP; hub `mkaichristensen/echo-clip`.

**EchoPrime (Nature 2026 / arXiv:2410.09704).** Vukadinovic et al. introduced a multi-video, view-primed VLM trained on >12M video–report pairs with view-informed anatomical attention and retrieval-augmented study-level interpretation (doi:10.1038/s41586-025-09850-x; *Nature* 2026;650:970–977). We cite it as the multi-view / multi-exam upper bound; EchoCLIP-TC stays single-clip video-vector aggregation on frozen EchoCLIP and does not attempt multi-exam fusion.

**CardiacCLIP (MICCAI 2025).** Du, Guo & Li adapt CLIP for few-shot LVEF with Multi-Frame Learning (attention frame fusion) and EchoZoom multi-resolution inputs (arXiv:2509.17065; papers.miccai.org; github.com/xmed-lab/CardiacCLIP). Closest methodological neighbor for temporal fusion; they report a 1-shot EchoNet-Dynamic MAE reduction of 2.07 under their few-shot protocol—**their claim, not ours**. Our contribution emphasizes *frozen EchoCLIP compatibility*, a locked ablation ladder (**R0–R6**), and explicit calibration / conformal reporting rather than few-shot SOTA claims without EchoNet runs. Comparison interface: `echoclip/cardiacclip_stub.py` (requires external weights; no invented numbers).

**Multiview video-CLIP / JACC: Asia line (2025–2026).** Tohyama and colleagues train video–language models that ingest multi-view echocardiogram sequences for comprehensive interpretation (preprint arXiv:2504.18800; related JACC: Asia clinical AI reporting, e.g. doi:10.1016/j.jacasi.2024.10.012 for ECG→echo abnormality screening). We cite this line as a multi-view contrast: EchoCLIP-TA stays single-clip PEFT on frozen EchoCLIP and does not claim multi-view fusion.

**EchoJEPA (preprint 2026).** Munim et al. propose a latent predictive (JEPA) foundation model pretrained on ~18M echocardiograms (arXiv:2602.02603). We position EchoCLIP-TA as a *label-efficient adapter on a published VLM*, not a competing foundation-scale pretraining effort.

**Public video / segmentation benchmarks.** EchoNet-Dynamic (Ouyang et al., *Nature* 2020; doi:10.1038/s41586-020-2145-8) provides the primary public EF video benchmark (~10k A4C clips) used for EchoCLIP external validation and for our locked protocol. CAMUS (Leclerc et al., *IEEE TMI* 2019; doi:10.1109/TMI.2019.2900516) remains the canonical open multi-structure 2D echo segmentation / EF resource (500 patients, A2C/A4C); we keep CAMUS as an optional external generalization stub (**待补充** on disk), not a substitute for EchoNet protocol numbers.

**Calibration and uncertainty.** Temperature scaling (Guo et al., ICML 2017) and split conformal prediction (Angelopoulos & Bates) provide lightweight, model-agnostic uncertainty tools suitable for frozen VLMs. Optional `--adaptive-conformal` uses normalized residuals with a heuristic or learned positive scale; when scale and conformal quantiles are both fit on the same VAL split, metrics are labeled as an **empirical heuristic** (prefer a held-out VAL-cal vs VAL-scale split when data allow). EchoCLIP-TA calibrates *prompt-rank / similarity-derived or direct-regression* EF scores under a VAL-only contract.

### 2.1 Honest innovation surface (non-claims explicit)

1. Frozen-encoder **video-level** \(z_v\) temporal module compatible with official EchoCLIP towers (**parameter-efficient adaptation**, not a new foundation model).  
2. Explicit **R0 ≠ R0U16 ≠ R1** semantics (official stride vs uniform-16 vs mean-pool); primary eval uses **uniform** except paper R0 (`official_stride`, no pad-to-16).  
3. **VAL-only** temperature / affine logistic @50/40/30 / ECE / Brier / split-conformal (+ optional adaptive conformal / AURC) / abstention as first-class metrics (R6).  
4. Locked **public-data** reproducibility contract (EchoNet-Dynamic + PAPER.md)—no invented clinical MAE; multi-seed mean±SD (prefer 5 seeds for paper; R0/R1 are deterministic once weights+sampling are fixed).  
5. **Non-claim:** we do not assert priority as “the first temporal EchoCLIP” or a cycle-aware zero-shot method.

---

## 3. Methods

### 3.1 Problem setup

Given a video \(V=\{x_t\}_{t=1}^{T}\) and optional structured text from EF/EDV fields, produce (i) a point EF estimate \(\hat{y}\) via zero-shot prompt ranking on a video embedding \(z_v\), and (ii) calibrated probabilities / conformal intervals for clinical thresholds (e.g., EF < 50%).

### 3.2 Frozen dual encoder

Image tower: ConvNeXt-Base (paper) or documented fallbacks (`resnet18` / `simple_cnn` for plumbing only). Text tower: CLIP-style transformer with official tokenizer / prompt templates. Contrastive InfoNCE is unchanged; TC does not replace pretraining. Successful clinical runs require `load_source` recording the hub id `hf-hub:mkaichristensen/echo-clip` (not `scratch_fallback`).

### 3.3 Frame sampling

Strategies: `random`, `uniform`, `ed_es`, `mixed`, `official_stride` (`echoclip/cycle_sample.py`). Locked protocol defaults (`echoclip/protocol.py`):

| ID | Alias | \(T\) (paper) | Train sample | Eval sample (VAL/TEST) | Notes |
|----|-------|---------------|--------------|----------------------|-------|
| R0 | B0 | variable | — | `official_stride` under `--paper`; else `uniform` | Zero-shot frames; **no pad-to-16** on official path |
| R0U16 | — | 16 | — | **`uniform`** | Uniform-16 ablation vs official R0 |
| R1 | M1 | 16 | — | **`uniform`** | Mean-pool \(z_v\) |
| R2–R4 | S0–S2 | 16 | `uniform` | **`uniform`** | Direct EF regression heads |
| R5 / R6 | M2 / M4 | 16 | `uniform` (mixed = EDES ablation) | **`uniform`** | Temporal \(z_v\); R6 + cal |
| Oracle-EDES | — | 16 | — | `ed_es` | Annotation-assisted upper bound |

Primary VAL/TEST **hard-fails** on `ed_es`/`mixed` unless Oracle-EDES. Seed 42 throughout (multi-seed helper: `scripts/run_seeds.py`; paper prefers 5 seeds; R0/R1 are deterministic once weights and sampling are fixed). Local **DEMO** smoke runs may use \(T=4\) and must be labeled DEMO.

### 3.4 Temporal aggregation

Frame embeddings \(Z\in\mathbb{R}^{T\times D}\) → \(z_v\in\mathbb{R}^{D}\) via mean pool (R1) or Temporal Transformer / attention pool (R5/R6; `echoclip/temporal.py`: `AttentionPool`, `TemporalTransformer` with CLS token). Only the temporal module (and optionally `logit_scale`) is trained; towers remain frozen when official weights load. Default R5 loss: `EFSoftContrastiveLoss` with **EF-only** captions (`--use-edv-captions` opt-in).

### 3.5 Zero-shot EF

Implementation: `echoclip/zeroshot.compute_regression_score` — cosine similarities → argsort prompts per frame → take top 20% of ranked EF values → median. Shape convention: 2D tensors are interpreted as `(T, D)` (one video), **not** `(B, D)`; batched video vectors must be reshaped to `(B, 1, D)` by the caller (`EchoCLIPInference.zero_shot_ef_batch`).

**R0/B0** keeps frame embeddings and applies the aggregator per frame then aggregates EF scalars across frames. **R1/R5/R6** build one \(z_v\) first, then apply the aggregator once. Because ranking is nonlinear, R0 ≠ R1 in general (documented in `PAPER.md` and unit tests). Optional scalar-mean “M1b” is **not** the locked R1.

### 3.6 Calibration (R6)

On VAL only (`echoclip/calibrate.py`): temperature or **affine logistic** for binary EF-threshold scores at **50/40/30**; report ECE and Brier at each threshold; fit split-conformal absolute residuals for target coverage \(1-\alpha\) (default 90%). Basic split conformal has **fixed width** (limitation documented); optional `--adaptive-conformal` uses normalized residuals \(|y-\hat{y}|/s(x)\) with heuristic or learned positive scale + risk–coverage / AURC. Bootstrap CIs for MAE/RMSE/\(R^2\)/AUCs and paired ΔMAE between methods are emitted in `echoclip/clinical.py`. Protocol runner hard-fails if `cal_manifest` equals the test manifest outside demo mode. Demo mode may use overlapping toy splits—**not** clinically interpretable.

### 3.7 Experiment matrix

| ID | Alias | Train | Pool | Calibrate | Role |
|----|-------|-------|------|-----------|------|
| R0 | B0 | No | frames | No | EchoCLIP-based zero-shot (`--paper` for official path) |
| R1 | M1 | No | mean | No | No-parameter video-vector ablation |
| R2–R4 | S0–S2 | Yes | supervised | No | Linear / MLP / temporal L1 heads |
| R5 | M2 | Yes (temporal + EF soft contrastive) | temporal | No | EF-aware PEFT temporal adaptation |
| R6 | M4 | Reuse R5 | temporal | Yes (VAL) | Calibrated TC |
| Oracle-EDES | — | No | mean | No | Annotation-assisted upper bound |

### 3.8 Metrics

Primary: EF MAE, RMSE, \(R^2\) (with bootstrap CIs); AUC at EF < 50/40/30 (with bootstrap CIs); ECE/Brier @50/40/30; conformal coverage / width; optional adaptive conformal + AURC; abstention MAE; paired bootstrap ΔMAE. Retrieval R@k is diagnostic only. Comparison table: `scripts/write_protocol_table.py` → `checkpoints/protocol/comparison.{json,md}`.
---

## 4. Experiments

### 4.1 Datasets

- **EchoNet-Dynamic (primary):** **待补充** — not present in this workspace after disk search of common paths (`data/`, env `ECHONET_ROOT`). Builder exits with AIMI download instructions (`DATA.md`). Legal access: Stanford AIMI non-commercial request at https://echonet.github.io/dynamic/ — not redistributed here.
- **CAMUS / EchoNet-Pediatric / EchoNet-LVH:** config stubs + builders; **待补充** on-disk data.
- **Demo synthetic pairs:** pipeline wiring only (`demo_is_not_clinical=true`; `ef_source=text_parse_demo_only`).

### 4.2 Implementation

PyTorch scaffold in this repository; protocol runner `scripts/run_protocol.py`; clinical eval `scripts/eval_clinical.py`. Gates: `python -m unittest discover -s tests -v` and `python scripts/validate.py --skip-eval`.

### 4.3 Results (clinical)

**待补充.** Do not substitute demo MAE for EchoNet or for the published EchoCLIP external 7.1% figure. The 7.1% value is attributed to Christensen et al. and is a *reproduction target* for B0 with hub weights + seed-42 5000-subset and/or full TEST, not a result of this draft.

### 4.4 Results (DEMO pipeline only — not clinical)

Local `checkpoints/protocol/*/metrics.json` (synthetic demo, \(T=4\), scratch weights):

| ID | DEMO MAE | DEMO ECE@50 | DEMO conformal coverage | load_source | n |
|----|----------|-------------|-------------------------|-------------|---|
| B0 | 11.25 | ≈0.65 | — | scratch_fallback | 32 |
| M1 | 11.25 | ≈0.65 | — | scratch_fallback | 32 |
| M2 | 8.125 | ≈0.34 | — | scratch | 32 |
| M4 | 8.125 | ≈0.00 | 1.0 (width 30; toy) | scratch | 32 |

**Interpretation bound:** These numbers prove metrics I/O and calibration code paths execute. They do **not** support any clinical ranking of R0–R6 (legacy B0/M1/M2/M4). R6’s near-zero DEMO ECE and perfect coverage reflect toy overlap / scratch embeddings, not calibrated clinical reliability.

### 4.5 Figures (this draft)

- Fig. 1 — Protocol architecture schematic  
- Fig. 2 — B0/M1/M2 ablation schematic  
- Fig. 3 — Calibration reliability **DEMO** cartoons  
- Fig. 4 — Protocol smoke metrics **DEMO ONLY**  
- Fig. 5 — Bilingual research roadmap (marks 待补充)  
- Fig. 6 — Split-conformal interval **DEMO** cartoon  

---

## 5. Discussion

**Honest innovation surface.** (i) Parameter-efficient / EF-aware temporal module on frozen EchoCLIP; (ii) explicit R0 vs R1 semantics + uniform primary eval; (iii) VAL-only calibration + conformal (+ optional adaptive) + abstention as first-class paper metrics; (iv) public-data reproducibility contract. **Non-claim:** not “first temporal EchoCLIP.”

**Positioning vs peers (claims we do *not* make).**

| Peer | Their scale / claim (literature) | Our stance |
|------|----------------------------------|------------|
| EchoCLIP | External EF MAE ≈7.1%; foundation pretraining | B0 reproduction *target*; TC adds temporal + calibration layer |
| EchoPrime | >12M pairs; multi-view study-level interpretation | Cite as upper-bound contrast; we stay single-clip frozen EchoCLIP |
| CardiacCLIP | Few-shot MFL + EchoZoom; reported 1-shot MAE Δ−2.07 on EchoNet | Closest temporal neighbor; we do not claim few-shot SOTA without EchoNet runs |

**What we do not claim.** Private-scale pretraining; multi-view exam fusion; any demo number as clinical EF MAE; equivalence of B0 and mean-pool without the documented nonlinear caveat; “cycle-aware zero-shot” as the primary framing.

---

## 6. Limitations

1. EchoNet-Dynamic videos and official hub weights are unavailable in the current environment → clinical tables **待补充**.  
2. `simple_cnn` / `scratch_fallback` paths are plumbing, not paper models.  
3. Demo calibration uses overlapping toy splits unsuitable for clinical conformal claims.  
4. Cross-dataset generalization untested without CAMUS/Pediatric/LVH on disk.  
5. No prospective clinical reader study.  
6. `official_reproduction_verified` remains false until open_clip hub load + aggregation parity (and preferably `compare_official_b0.py` on real AVI) succeed.

---

## 7. Conclusions

EchoCLIP-TA provides a parameter-efficient temporal and calibration layer—and a locked evaluation protocol—on top of EchoCLIP for public echocardiography videos. Completing EchoNet-Dynamic evaluation with official weights is required before any clinical performance claim. Until then, this manuscript documents methods, ablations, and honesty boundaries for a methods-focused submission.

---

## Data and code availability

- Code: https://github.com/Coucou2016/EchoCLIP-TC (MIT).  
- EchoNet-Dynamic: Stanford AIMI (separate non-commercial terms) — **待补充** local copy.  
- Official EchoCLIP weights: Hugging Face `mkaichristensen/echo-clip` / echonet/echo_CLIP — **待补充** successful hub load (`load_source` must record hub id).

## References (seed list; verified 2026-09)

1. Christensen M, Vukadinovic M, Yuan N, Ouyang D. Vision–language foundation model for echocardiogram interpretation. *Nat Med*. 2024;30:1481–1488. doi:10.1038/s41591-024-02959-y  
2. Vukadinovic M, Chiu IM, Tang X, et al. Comprehensive echocardiogram evaluation with view primed vision language AI (EchoPrime). *Nature*. 2026;650:970–977. doi:10.1038/s41586-025-09850-x ; preprint arXiv:2410.09704  
3. Du Y, Guo J, Li X. CardiacCLIP: Video-based CLIP adaptation for LVEF prediction in a few-shot manner. MICCAI 2025. arXiv:2509.17065 ; https://papers.miccai.org/miccai-2025/0127-Paper0034.html  
4. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function (EchoNet-Dynamic). *Nature*. 2020;580:252–256. doi:10.1038/s41586-020-2145-8  
5. Leclerc S, Smistad E, Pedrosa J, et al. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography (CAMUS). *IEEE Trans Med Imaging*. 2019;38(9):2198–2210. doi:10.1109/TMI.2019.2900516  
6. Radford A, et al. Learning transferable visual models from natural language supervision (CLIP). ICML 2021.  
7. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.  
8. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511 / *Found Trends Mach Learn*.  
9. Munim A, Fallahpour A, Szasz T, et al. EchoJEPA: A latent predictive foundation model for echocardiography. arXiv:2602.02603 (2026).  
10. Tohyama T, et al. Video CLIP model for multi-view echocardiography interpretation. arXiv:2504.18800 (2025).  
11. Related JACC: Asia clinical AI context: deep learning identification of echocardiographic abnormalities from ECGs. *JACC: Asia*. doi:10.1016/j.jacasi.2024.10.012.

---

## Assumptions or missing inputs

- Clinical metrics tables: **missing** → marked 待补充.  
- Author list, affiliations, ethics, funding: **missing** → 待补充.  
- Exact EchoPrime / CardiacCLIP numerical comparisons on the same split: **not run here**.  
- Drafting notes and agent handoff logs live under `notes/` / `reports/` (not manuscript body).
