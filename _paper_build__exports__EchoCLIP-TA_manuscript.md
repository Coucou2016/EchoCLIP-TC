# EchoCLIP-TA: Parameter-efficient temporal adaptation and calibrated evaluation on frozen EchoCLIP for echocardiographic ejection fraction

**Authors.** EchoCLIP-TA project team
**Correspondence.** Repository: https://github.com/Coucou2016/EchoCLIP-TC
**Keywords.** echocardiography; vision–language model; temporal adaptation; parameter-efficient fine-tuning; calibration; EchoCLIP; EchoNet-Dynamic

## Abstract

Left ventricular ejection fraction (LVEF) estimation from echocardiography remains central to cardiac care. EchoCLIP showed that a vision–language foundation model can recover LVEF by ranking clinical prompts over video frames, yet the public inference path is mainly frame-centric and reports uncalibrated similarities. In this study we present EchoCLIP-TA, a parameter-efficient temporal adaptation stack that freezes official EchoCLIP towers and adds a video-level temporal module, supervised LVEF heads, and validation-only calibration under a locked public R0–R6 protocol on EchoNet-Dynamic. We separate zero-shot prompt ranking from direct LVEF regression, and we keep mixed-pool historical anchors out of adapted test reporting. On a TEST-only subset of 50 EchoNet-Dynamic videos with local EchoCLIP weights, official-stride zero-shot R0 reached mean absolute error (MAE) of 11.98 percentage points, with AUC for EF < 50% of 0.768. Dataset characterization of the full 10,030-video corpus confirms split balance and threshold prevalence used by the protocol. EchoCLIP-TA reframes contribution as fair temporal PEFT and trustworthy evaluation on a frozen foundation model rather than private-scale re-pretraining.

## 1. Introduction

Echocardiography is the frontline modality for assessing cardiac structure and function. Automated LVEF estimation reduces inter-observer variability and supports large-scale screening workflows [1]. Deep video models such as EchoNet-Dynamic established strong supervised baselines on publicly released apical four-chamber clips [1]. Vision–language models then opened a complementary route that aligns ultrasound frames with free-text reports. EchoCLIP pretrained a ConvNeXt–transformer dual encoder on more than one million echocardiogram–report pairs and reported external LVEF MAE near 7.1% on EchoNet-Dynamic together with strong device recognition AUCs [2].

Two practical gaps remain for clinical reuse of frozen EchoCLIP. First, temporal aggregation is underspecified. Public zero-shot LVEF ranking often encodes frames independently and aggregates ranked prompt scores, while cardiac function is dynamic across the cycle. Later systems such as EchoPrime and CardiacCLIP emphasize multi-view or multi-frame modeling, usually with different training budgets and evaluation contracts [3,4]. Second, calibration is rarely first-class. Cosine similarities and prompt-rank LVEF estimates are seldom accompanied by expected calibration error, Brier scores, or finite-sample prediction intervals [5,6].

EchoCLIP-TA addresses both gaps without rewriting the dual encoder. We freeze pretrained EchoCLIP towers when local official weights are present, train only lightweight temporal or supervised heads, and fit temperature or affine-logistic calibration exclusively on a validation split. Innovation claims are therefore protocol- and reliability-centric. We do not claim priority as the first temporal EchoCLIP, and we do not treat demo wiring metrics as clinical evidence.

## 2. Related work

**EchoCLIP.** Christensen et al. introduced a frame–text foundation model for echocardiography with zero-shot LVEF via prompt ranking and a long-context retrieval variant EchoCLIP-R [2]. We treat official weights and prompts as the R0 baseline.

**Supervised video LVEF.** Ouyang et al. released EchoNet-Dynamic with 10,030 labeled A4C videos and a spatiotemporal CNN that achieved internal LVEF MAE of 4.1% and external MAE of 6.0% [1]. That supervised video path remains the reference for fully labeled training.

**Video and multi-view VLMs.** EchoPrime scales to multi-video study-level interpretation [3]. CardiacCLIP adapts CLIP with multi-frame attention and multi-resolution inputs for few-shot LVEF and reported a 1-shot EchoNet MAE reduction of 2.07 under its own protocol [4]. Multiview video–language models further expand coverage across standard views [7]. EchoCLIP-TA stays on single-clip PEFT atop frozen EchoCLIP and uses these works for positioning rather than reimplementation.

**Open segmentation resources.** CAMUS provides fully open multi-structure 2D echo annotations for 500 patients [8]. We cite it as an optional generalization resource. It is not a substitute for the EchoNet protocol numbers in this paper.

**Calibration and conformal prediction.** Temperature scaling and related post-hoc methods recalibrate classification scores with a held-out validation set [5]. Split conformal prediction supplies finite-sample intervals under exchangeability [6]. EchoCLIP-TA places these tools in experiment R6 with a validation-only fitting contract.

## 3. Methods

### 3.1 Problem setup

Given a video V = {x_t} from t = 1 to T and structured EF text when available, we produce a point estimate ŷ of LVEF and, for calibrated runs, probabilities or intervals at clinical thresholds EF < 50%, 40%, and 30%. Prediction operates in one of two modes. Zero-shot mode ranks EF prompts on a video embedding. Direct regression mode maps pooled visual features to a scalar EF with a supervised head.

### 3.2 Frozen dual encoder

We use the official EchoCLIP open_clip configuration with a ConvNeXt-Base image tower and a 12-layer text transformer (embedding dimension 512, context length 77) [2]. Local weights were loaded from models/echo-clip/open_clip_pytorch_model.bin with zero missing keys relative to the published configuration. Image preprocessing uses the published CLIP mean and standard deviation at resolution 224.

### 3.3 Frame sampling

Sampling strategies live in echoclip/cycle_sample.py. Under the paper path, R0 uses official stride indices 0:min(40,T):2 without pad-to-16. Primary VAL/TEST sampling for other experiments is uniform with T = 16. Annotation-assisted ed_es sampling is reserved for Oracle-EDES. Train sampling is uniform unless the R5_EDESTRAIN ablation is selected.

### 3.4 Temporal aggregation and supervised heads

Frame embeddings Z in R^(T×D) map to a video vector z_v in R^D by mean pooling (R1) or by a Temporal Transformer / attention pool (R5/R6). Supervised baselines R2–R4 attach linear, MLP, or temporal regression heads and use direct EF regression. Default R5 training uses EF soft multi-positive contrastive loss with EF-only captions. Towers remain frozen when official weights load.

### 3.5 Zero-shot LVEF

For each video we encode selected frames, compute cosine similarities against the official EF prompt grid spanning 0–100, take the top 20% ranked EF values, and report their median [2]. R0 applies this ranking per frame and aggregates EF scalars. R1 and adapted temporal runs build one z_v first. Because ranking is nonlinear, R0 and R1 are not interchangeable.

### 3.6 Calibration (R6)

On validation only we fit temperature or affine logistic maps for threshold scores, then report ECE and Brier scores. Split conformal absolute residuals target coverage 1−α (default 90%). Optional adaptive conformal uses normalized residuals. Calibration manifests must not equal the test manifest outside demo mode.

### 3.7 Locked experiment matrix

| ID | Role | Train | Pool | Mode | Calibrate |
|----|------|-------|------|------|-----------|
| R0 | Official-stride zero-shot | No | frames | zeroshot | No |
| R0U16 | Uniform-16 zero-shot control | No | frames | zeroshot | No |
| R1 | Mean-pool video vector | No | mean | zeroshot | No |
| R2–R4 | Linear / MLP / temporal heads | Yes | supervised | direct_regression | No |
| R5 | EF-aware temporal PEFT | Yes | temporal | zeroshot | No |
| R6 | Calibrated temporal PEFT | Reuse R5 | temporal | zeroshot | Yes (VAL) |
| Oracle-EDES | Annotation-assisted upper bound | No | mean | zeroshot | No |

Supervised checkpoints use SCHEMA_VERSION = 2. The mixed-pool subset r0_external_anchor_5000 is a historical external anchor and is not used for adapted R2–R6 reporting. Adapted evaluation uses TEST-only subsets such as test_subset_500 (seed 42).

### 3.8 Metrics

Primary metrics are EF MAE, RMSE, and R². Threshold discrimination uses AUC at EF < 50/40/30. Calibration metrics include ECE, Brier score, conformal coverage, and interval width.

## 4. Experiments

### 4.1 Dataset and local corpus

We evaluated on EchoNet-Dynamic videos present at E:\Datasets\EchoNet-Dynamic with FileList.csv, Videos/, and VolumeTracings.csv. The local mirror contains 10,030 AVI clips matching the published corpus size [1]. Official splits contain 7,465 TRAIN, 1,288 VAL, and 1,277 TEST videos. Across all videos, EF mean was 55.75 with standard deviation 12.37. Fractions of videos with EF < 50%, 40%, and 30% were 0.224, 0.126, and 0.062. Figure 1 shows the EF histogram. Figure 2 shows split sizes. Figure 3 shows threshold prevalence.

Project manifests under data/echonet_dynamic/ encode EF, EDV, ESV, and ED/ES frame indices. The TEST-only subset test_subset_500 holds 500 videos (seed 42) with EF mean 54.82 and standard deviation 12.84. Newly computed zero-shot tables use the first 50 videos of that seeded TEST-only subset under CPU evaluation with local official weights.

Licensing note. The local videos follow a Kaggle-mirror layout used for research computation in this workspace. We do not claim Stanford AIMI Research Use Agreement authorization for this copy. Redistribution of patient videos is not permitted by this paper or repository.

### 4.2 Implementation and hardware

Code lives in the EchoCLIP-TC repository (package echoclip). Protocol definitions are in echoclip/protocol.py and PAPER.md. Official EchoCLIP weights load from models/echo-clip/. The machine GPU is a GeForce GTX 950M with 4 GB memory. Full ConvNeXt-Base GPU batching was unstable, so reported zero-shot runs used CPU with frame batch size 1. Wall-clock cost was about 12–15 seconds per video after model load.

Exact R0 command:

python F:\EchoCLIP_paper_build\scripts\eval_r0_resume.py --limit 50 --output F:\EchoCLIP_paper_build\results\R0_test_subset_50.json --experiment-id R0 --device cpu

Seed 42. Weights models/echo-clip/open_clip_pytorch_model.bin. Manifest data/echonet_dynamic/test_subset_500.json truncated to the first 50 pairs. Echonet root E:\Datasets\EchoNet-Dynamic.

### 4.3 Zero-shot results on TEST-only subset

Table 1 reports completed local zero-shot runs. Figure 4 shows the R0 scatter of predicted versus reference EF. Figure 5 shows R0 residuals.

**Table 1. Zero-shot LVEF on TEST-only subset prefix (local official EchoCLIP weights).**

| Experiment | n | Sampling | Pool | MAE | RMSE | R² | AUC (EF<50) | AUC (EF<40) | AUC (EF<30) |
|------------|---|----------|------|-----|------|----|-------------|-------------|-------------|
| R0 | 50 | official_stride | frames | 11.98 | 15.35 | -0.21 | 0.768 | 0.793 | 0.853 |
| R1 | 50 | uniform | mean | 12.02 | 15.51 | -0.23 | 0.777 | 0.768 | 0.853 |

R0 predictions showed a positive bias of 10.93 EF percentage points (predicted mean 62.81, reference mean 51.88). The published EchoCLIP external MAE near 7.1% remains a literature reference on broader external testing [2]. It is not a substitute for Table 1.

### 4.4 Protocol completeness versus executed training

R0U16, supervised R2–R4, and adapted R5–R6 are defined in code and form the intended ablation ladder. In this hardware session we prioritized reproducible zero-shot evaluation with local official weights. Adapted temporal training and multi-seed matrices remain executable through scripts/run_protocol.py and scripts/run_paper_matrix.py when a GPU with sufficient memory is available. We therefore report no invented adapted MAE numbers. Demo protocol smoke metrics in older reports are excluded from Table 1 because they use synthetic pairs and scratch towers.

### 4.5 Attention to ED/ES (diagnostic)

An attention diagnostic utility exists in scripts/analyze_attention_edes.py. Prior demo attention summaries are labeled non-clinical and are not used as EchoNet evidence.

## 5. Discussion

The central methodological claim of EchoCLIP-TA is not that a frozen encoder with a small temporal head will always beat foundation-scale video VLMs. The claim is that public evaluation of EchoCLIP-style LVEF needs an explicit ladder that separates sampling choices, pooling choices, prediction modes, and calibration. R0 versus R1 is the simplest illustration. Frame-wise prompt ranking and mean-pooled video ranking answer different mathematical questions because the top-20% median operator is nonlinear.

Relative to CardiacCLIP and EchoPrime, our scope is narrower and more protocol-bound [3,4]. Those systems explore few-shot adaptation or multi-exam fusion. EchoCLIP-TA keeps single-clip PEFT on frozen EchoCLIP and spends capacity on calibration contracts and honest split discipline. The mixed-pool r0_external_anchor_5000 subset is retained only as a historical external anchor. Adapted claims use TEST-only subsets.

Local zero-shot MAE on the completed subset should be read with clear boundaries. The evaluated prefix is smaller than full TEST and has a lower reference EF mean than the full corpus. CPU official-stride decoding may differ in low-level details from upstream example scripts even when aggregation formula checks pass. The 4 GB GPU forced CPU inference, so runtime comparisons with hub GPUs are not meaningful. The observed positive bias suggests that prompt-rank aggregation on this subset overestimates EF on average, which motivates the supervised and calibrated PEFT stages in R2–R6 once adequate compute is available.

## 6. Limitations

1. Adapted R5–R6 training and supervised R2–R4 full matrices were not completed on the 4 GB GPU in this session.
2. Newly computed zero-shot results use a TEST-only prefix subset rather than all 1,277 TEST videos.
3. No prospective reader study was performed.
4. Cross-dataset generalization to CAMUS or pediatric/LVH cohorts was not run.
5. Data licensing for the local Kaggle-mirror copy is stated without AIMI RUA authorization claims.
6. Repository flag official_reproduction_verified still depends on upstream parity assets beyond formula checks.

## 7. Conclusions

EchoCLIP-TA packages frozen EchoCLIP with a locked R0–R6 public protocol for temporal PEFT, supervised heads, and validation-only calibration. On local EchoNet-Dynamic assets and official EchoCLIP weights, official-stride zero-shot R0 produced MAE 11.98 on n = 50 TEST-only videos, with threshold AUCs 0.768 / 0.793 / 0.853 at EF < 50/40/30. The reusable contribution is the evaluation contract and the PEFT interface. Completing adapted training on larger GPUs is the direct next measurement, not a change in the scientific question.

## Data and code availability

Code: https://github.com/Coucou2016/EchoCLIP-TC
EchoNet-Dynamic project page: https://echonet.github.io/dynamic/
EchoCLIP weights hub id: mkaichristensen/echo-clip
Local analysis scripts and numerical JSON: F:\EchoCLIP_paper_build\results\
OA PDFs archived under F:\EchoCLIP_paper_build\researchwrite\echoclip-ta\sources\

## References

1. Ouyang D, He B, Ghorbani A, Yuan N, Ebinger J, Langlotz CP, Heidenreich PA, Harrington RA, Liang DH, Ashley EA, Zou JY. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020;580:252–256. doi:10.1038/s41586-020-2145-8
2. Christensen M, Vukadinovic M, Yuan N, Ouyang D. Vision–language foundation model for echocardiogram interpretation. Nat Med. 2024. doi:10.1038/s41591-024-02959-y
3. Vukadinovic M, et al. EchoPrime: A multi-video view-informed vision–language model for comprehensive echocardiography interpretation. arXiv:2410.09704. Nature. 2026;650:970–977. doi:10.1038/s41586-025-09850-x
4. Du Y, Guo J, Li X. CardiacCLIP: Video-based CLIP adaptation for LVEF prediction in a few-shot manner. MICCAI 2025. arXiv:2509.17065. doi:10.1007/978-3-032-04971-1_5
5. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017. arXiv:1706.04599
6. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511
7. Tohyama T, et al. Multiview echocardiogram video–language modeling. arXiv:2504.18800
8. Leclerc S, Smistad E, Pedrosa J, Østvik A, et al. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography. IEEE Trans Med Imaging. 2019;38(9):2198–2210. doi:10.1109/TMI.2019.2900516
9. Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision. ICML 2021. arXiv:2103.00020
