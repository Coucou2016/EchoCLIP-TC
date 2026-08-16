# EchoCLIP-TC: Temporal aggregation and calibrated zero-shot evaluation for echocardiogram vision–language models

**Status:** Methods manuscript draft (Nature-skills / methods paper type).  
**Honesty:** Clinical EchoNet-Dynamic EF MAE / AUC numbers are **待补充** until official weights and AIMI data are available locally. Demo / synthetic figures are labeled **DEMO** and are not clinical results.

**Axes (nature-writing):** `task=manuscript` · `paper_type=methods` · `language=en` · `journal=nature-family` (methods / Nat Commun–style framing).

**One-sentence argument.** In echocardiogram vision–language interpretation, we present EchoCLIP-TC—a frozen-encoder temporal aggregator with validation-only temperature scaling and split-conformal intervals—together with a locked B0/M1/M2/M4 public-data protocol that makes video-level ejection-fraction (EF; left-ventricular ejection fraction) evaluation and uncertainty reporting reproducible; clinical superiority claims remain contingent on EchoNet-Dynamic + official EchoCLIP weights (**待补充**).

---

## Terminology ledger (canonical forms)

| Term | Canonical form | Notes |
|------|----------------|-------|
| Base model | EchoCLIP | Christensen et al., *Nat Med* 2024 |
| This work | EchoCLIP-TC | Temporal, Calibrated adaptation layer |
| Primary metric | EF MAE | Mean absolute error in EF percentage points |
| Video embedding | \(z_v\) | Video-level representation after pooling |
| Protocol IDs | B0, M1, M2, M4 | Locked in `PAPER.md` |
| Calibration | Temperature scaling; ECE; Brier; split-conformal | Fit on VAL only |
| Public data | EchoNet-Dynamic | Stanford AIMI; non-commercial |
| Related VLMs | EchoPrime; CardiacCLIP | Baselines for positioning, not reimplemented here |

---

## Title options

1. **EchoCLIP-TC: Cycle-aware temporal aggregation and calibrated evaluation for echocardiogram vision–language models** (preferred)
2. A reproducible temporal–calibration protocol for frozen EchoCLIP on public echocardiography videos
3. From frames to calibrated video vectors: EchoCLIP-TC for zero-shot EF estimation

---

## Abstract

**Background.** EchoCLIP aligns echocardiogram frames with clinical text and supports zero-shot estimation of left ventricular ejection fraction (EF), but the published pipeline is primarily frame-centric and reports uncalibrated cosine similarities.

**Methods.** We introduce EchoCLIP-TC (Temporal, Calibrated): a lightweight temporal aggregator (attention pooling or Temporal Transformer) on frozen EchoCLIP towers, cycle-aware frame sampling, structured EchoNet caption templates, and validation-only temperature scaling with split-conformal EF intervals and abstention. We lock a four-arm protocol—B0 (official per-frame EF aggregation), M1 (mean-pooled video vector), M2 (learned temporal \(z_v\)), M4 (M2 + calibration)—for EchoNet-Dynamic.

**Results.** **待补充 (EchoNet-Dynamic + official hub weights).** Locally we only demonstrate end-to-end pipeline smoke tests on synthetic demo pairs (`load_source=scratch_fallback`); demo MAE/ECE must not be read as clinical performance.

**Conclusions.** EchoCLIP-TC reframes honest innovation as *reproducible temporal aggregation + trustworthy uncertainty* on public data, without claiming private million-scale pretraining. Clinical claims require completing the gated evaluation path.

**Keywords:** echocardiography; vision–language model; temporal aggregation; calibration; conformal prediction; EchoCLIP

---

## 1. Introduction

Echocardiography remains the frontline modality for cardiac structure and function. Vision–language models (VLMs) reduce dependence on task-specific labels by aligning images or videos with report text. EchoCLIP demonstrated that contrastive pretraining on >1M clinical video–text pairs enables zero-shot EF estimation (reported external EF MAE ≈ 7.1% in Christensen et al., *Nature Medicine* 2024) and device recognition.

Two gaps matter for clinical reuse. First, **temporal aggregation**: EchoCLIP’s public inference path emphasizes per-frame encoding with post-hoc aggregation of ranked EF prompts, whereas cardiac function is inherently dynamic; subsequent models (EchoPrime; CardiacCLIP) emphasize multi-view or multi-frame video modeling, but often with different training budgets and evaluation contracts. Second, **calibration**: cosine similarities and prompt-rank EF estimates are rarely reported with expected calibration error (ECE), Brier score, or finite-sample prediction intervals.

EchoCLIP-TC addresses both gaps *without rewriting the dual encoder*: we freeze pretrained towers when available, train only a temporal module for video-level \(z_v\), and fit temperature / conformal procedures exclusively on a validation split. Innovation claims are therefore **protocol- and reliability-centric**, not “another foundation model trained on private 1M reports.”

---

## 2. Related work

**EchoCLIP (Nature Medicine 2024).** Christensen et al. pretrained a frame–text CLIP-style foundation model on >1M echocardiogram–report pairs; zero-shot LVEF via prompt ranking (external EchoNet-Dynamic EF MAE ≈ 7.1%; internal MAE ≈ 8.4%); EchoCLIP-R for long-context retrieval. We treat official weights + prompts as the B0 baseline and do **not** claim private-scale re-pretraining.

**EchoPrime (Nature 2026 / arXiv:2410.09704).** Vukadinovic et al. introduced a multi-video, view-primed VLM trained on >12M video–report pairs with view-informed anatomical attention and retrieval-augmented study-level interpretation (doi:10.1038/s41586-025-09850-x). We cite it as the multi-view / multi-exam upper bound; EchoCLIP-TC stays single-clip video-vector aggregation on frozen EchoCLIP and does not attempt multi-exam fusion.

**CardiacCLIP (MICCAI 2025).** Du, Guo & Li adapt CLIP for few-shot LVEF with Multi-Frame Learning (attention frame fusion) and EchoZoom multi-resolution inputs (arXiv:2509.17065; papers.miccai.org). Closest methodological neighbor for temporal fusion; our contribution emphasizes *frozen EchoCLIP compatibility*, a locked ablation ladder (B0/M1/M2/M4), and explicit calibration / conformal reporting rather than few-shot SOTA claims without EchoNet runs.

**Public video / segmentation benchmarks.** EchoNet-Dynamic (Ouyang et al., *Nature* 2020; doi:10.1038/s41586-020-2145-8) provides the primary public EF video benchmark (~10k A4C clips) used for EchoCLIP external validation and for our locked protocol. CAMUS (Leclerc et al., *IEEE TMI* 2019; doi:10.1109/TMI.2019.2900516) remains the canonical open multi-structure 2D echo segmentation / EF resource (500 patients, A2C/A4C); we keep CAMUS as an optional external generalization stub (**待补充** on disk), not a substitute for EchoNet protocol numbers.

**Calibration and uncertainty.** Temperature scaling (Guo et al., ICML 2017) and split conformal prediction (Angelopoulos & Bates) provide lightweight, model-agnostic uncertainty tools suitable for frozen VLMs. Echo cardiac uncertainty work more often targets segmentation→EF pipelines; EchoCLIP-TC instead calibrates *prompt-rank / similarity-derived* EF scores under a VAL-only contract.

### 2.1 Writing architectures to imitate (venue-level)

| Venue / exemplar | What to imitate | What not to imitate for EchoCLIP-TC |
|------------------|-----------------|-------------------------------------|
| *Nat Med* EchoCLIP | Clinical motivation → foundation capability → zero-shot tasks → external validation; clear honesty on training scale | Claiming a new private 1M+ foundation model |
| *Nature* EchoPrime | Multi-view clinical breadth framing; study-level integration as *contrast* | Matching 12M-scale training narrative |
| *MICCAI* CardiacCLIP | Compact related-work contrast; ablation / few-shot tables; method figures for MFL-style fusion | Few-shot SOTA numbers without reproducing their protocol |
| *IEEE TMI* CAMUS-style methods | Dataset contract, metrics definitions, segmentation/EF reporting discipline | Segmentation-first story as the main claim |

**Recommended hybrid outline for this paper:** Nat Med–style Intro + honest positioning; MICCAI-style Methods/Experiments with B0/M1/M2/M4 tables; reliability diagrams + conformal coverage/width (calibration-first Results); Discussion that separates *protocol innovation* from *foundation-scale* claims.

### 2.2 Honest innovation surface (non-claims explicit)

1. Frozen-encoder **video-level** \(z_v\) temporal module compatible with official EchoCLIP towers.  
2. Explicit **B0 ≠ M1** semantics (nonlinear prompt ranking ⇒ frame-then-aggregate ≠ aggregate-then-rank).  
3. **VAL-only** temperature / ECE / Brier / split-conformal / abstention as first-class metrics (M4).  
4. Locked **public-data** reproducibility contract (EchoNet-Dynamic + PAPER.md protocol IDs)—no invented clinical MAE.

---

## 3. Methods

### 3.1 Problem setup

Given a video \(V=\{x_t\}_{t=1}^{T}\) and optional structured text from EF/EDV fields, produce (i) a point EF estimate \(\hat{y}\) via zero-shot prompt ranking on a video embedding \(z_v\), and (ii) calibrated probabilities / conformal intervals for clinical thresholds (e.g., EF < 50%).

### 3.2 Frozen dual encoder

Image tower: ConvNeXt-Base (paper) or documented fallbacks (`resnet18` / `simple_cnn` for plumbing only). Text tower: CLIP-style transformer with official tokenizer / prompt templates. Contrastive InfoNCE is unchanged; TC does not replace pretraining.

### 3.3 Cycle-aware sampling

Strategies: `random`, `uniform`, `ed_es`, `mixed`. Protocol default for EchoNet-Dynamic: \(T=16\), seed 42 (see `configs/echonet_dynamic.yaml`).

### 3.4 Temporal aggregation

Frame embeddings \(Z\in\mathbb{R}^{T\times D}\) → \(z_v\in\mathbb{R}^{D}\) via mean pool (M1) or Temporal Transformer / attention pool (M2/M4). Only the temporal module (and optionally `logit_scale`) is trained; towers remain frozen when official weights load.

### 3.5 Zero-shot EF

Official EchoCLIP path: rank EF prompt templates; take the median of the top 20% ranked EF values. **B0** applies this per frame then aggregates scalars across frames. **M1/M2/M4** build one \(z_v\) first, then apply the aggregator once. Because ranking is nonlinear, B0 ≠ M1 in general (documented in `PAPER.md` and unit tests).

### 3.6 Calibration (M4)

On VAL only: temperature scaling for binary EF-threshold scores; report ECE and Brier; fit split-conformal absolute residuals for target coverage \(1-\alpha\) (default 90%); optional width-based abstention. Hard-fail if `cal_manifest` equals the test manifest outside demo mode.

### 3.7 Experiment matrix

| ID | Train | Pool | Calibrate | Role |
|----|-------|------|-----------|------|
| B0 | No | frames | No | Official-style baseline |
| M1 | No | mean | No | No-parameter video-vector ablation |
| M2 | Yes (temporal) | temporal | No | Primary TC model |
| M4 | Reuse M2 | temporal | Yes (VAL) | Calibrated TC |

### 3.8 Metrics

Primary: EF MAE, RMSE, \(R^2\); AUC at EF < 50/40/30; ECE; Brier; conformal coverage / width; abstention MAE. Retrieval R@k is diagnostic only.

---

## 4. Experiments

### 4.1 Datasets

- **EchoNet-Dynamic (primary):** **待补充** — not present in this workspace; builder exits with download instructions.
- **CAMUS / EchoNet-Pediatric / EchoNet-LVH:** config stubs + builders; **待补充** on-disk data.
- **Demo synthetic pairs:** pipeline wiring only (`demo_is_not_clinical=true`).

### 4.2 Implementation

PyTorch scaffold in this repository; protocol runner `scripts/run_protocol.py`; clinical eval `scripts/eval_clinical.py`. Gates: `python -m unittest discover -s tests -v` and `python scripts/validate.py --skip-eval`.

### 4.3 Results (clinical)

**待补充.** Do not substitute demo MAE (e.g., B0/M1 demo MAE 11.25, M2/M4 demo MAE 8.125 under `scratch_fallback`) for EchoNet or for the published EchoCLIP external 7.1% figure. The 7.1% value is attributed to Christensen et al. and is a *reproduction target* for B0 with hub weights, not a result of this draft.

### 4.4 Figures (this draft)

- Fig. 1 — Protocol architecture schematic  
- Fig. 2 — B0/M1/M2 ablation schematic  
- Fig. 3 — Calibration reliability **DEMO** cartoons  
- Fig. 4 — Protocol smoke metrics **DEMO ONLY**  
- Fig. 5 — Bilingual research roadmap (marks 待补充)  
- Fig. 6 — Split-conformal interval **DEMO** cartoon  

---

## 5. Discussion

**Honest innovation surface.** (i) Video-vector temporal module compatible with frozen EchoCLIP; (ii) explicit B0 vs M1 semantics to prevent unfair ablations; (iii) VAL-only calibration + conformal + abstention as first-class paper metrics; (iv) public-data reproducibility contract.

**What we do not claim.** Private-scale pretraining; multi-view exam fusion (EchoPrime); few-shot SOTA without EchoNet runs (CardiacCLIP-style); any demo number as clinical EF MAE.

**Imitation architecture.** Prefer Nat Med EchoCLIP’s clinical→zero-shot→external-validation arc, MICCAI CardiacCLIP’s ablation-table density, and TMI-style metric/dataset contracts—not EchoPrime’s multi-exam foundation narrative. Separate *model*, *protocol*, and *reliability* sections; keep innovation claims protocol- and calibration-centric.

---

## 6. Limitations

1. EchoNet-Dynamic videos and official hub weights are unavailable in the current environment → clinical tables **待补充**.  
2. `simple_cnn` / `scratch_fallback` paths are plumbing, not paper models.  
3. Demo calibration uses overlapping toy splits unsuitable for clinical conformal claims.  
4. Cross-dataset generalization untested without CAMUS/Pediatric/LVH on disk.  
5. No prospective clinical reader study.

---

## 7. Conclusions

EchoCLIP-TC provides a temporal and calibration layer—and a locked evaluation protocol—on top of EchoCLIP for public echocardiography videos. Completing EchoNet-Dynamic evaluation with official weights is required before any clinical performance claim. Until then, this manuscript documents methods, ablations, and honesty boundaries suitable for a methods / Nature-family follow-on paper.

---

## Data and code availability

- Code: this repository (MIT).  
- EchoNet-Dynamic: Stanford AIMI (separate non-commercial terms) — **待补充** local copy.  
- Official EchoCLIP weights: Hugging Face `mkaichristensen/echo-clip` / echonet/echo_CLIP — **待补充** successful hub load (`load_source` must record hub id).

## References (seed list; WebSearch-verified 2026-08-16)

1. Christensen M, Vukadinovic M, Yuan N, Ouyang D. Vision–language foundation model for echocardiogram interpretation. *Nat Med*. 2024;30:1481–1488. doi:10.1038/s41591-024-02959-y  
2. Vukadinovic M, Chiu IM, Tang X, et al. Comprehensive echocardiogram evaluation with view primed vision language AI (EchoPrime). *Nature*. 2026;650:970–977. doi:10.1038/s41586-025-09850-x ; preprint arXiv:2410.09704  
3. Du Y, Guo J, Li X. CardiacCLIP: Video-based CLIP adaptation for LVEF prediction in a few-shot manner. MICCAI 2025. arXiv:2509.17065 ; https://papers.miccai.org/miccai-2025/paper/0034_paper.pdf  
4. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function (EchoNet-Dynamic). *Nature*. 2020;580:252–256. doi:10.1038/s41586-020-2145-8  
5. Leclerc S, Smistad E, Pedrosa J, et al. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography (CAMUS). *IEEE Trans Med Imaging*. 2019;38(9):2198–2210. doi:10.1109/TMI.2019.2900516  
6. Radford A, et al. Learning transferable visual models from natural language supervision (CLIP). ICML 2021.  
7. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.  
8. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511 / *Found Trends Mach Learn*.

---

## Assumptions or missing inputs (nature-writing notes)

- Clinical metrics tables: **missing** → marked 待补充.  
- Author list, affiliations, ethics, funding: **missing** → 待补充.  
- Exact EchoPrime / CardiacCLIP numerical comparisons on the same split: **not run here**.  
- ChatGPT live literature chat on 2026-08-16 (retry): **browser MCP still blocked** (`cursor-ide-browser` absent; `open_resource` → unknown agent); framing advanced via independent WebSearch; prior dual-agent chat https://chatgpt.com/c/6a80922d-d1d0-83ea-970c-67b829457cd6 used for B0/M1 semantics only. See `reports/echoclip_tc_literature_chatgpt_20260816.md`.
