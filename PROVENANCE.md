# PROVENANCE — file-level origin, upstream similarity, and license status

**Project:** EchoCLIP-TA (repository retained as `EchoCLIP-TC` for URL continuity;
Python package `echoclip`). This is the working tree-layout copy.

**Audit date:** 2026-10-07
**Auditor:** automated provenance/licensing closure pass (human author confirmation required — see below).

**Docstring alignment follow-up (2026-10-08):** the module docstrings of
`echoclip/text.py` and `echoclip/official_r0.py` no longer self-describe as
"clean-room"; they now carry the conservative canonical wording (independently
authored, behavior intentionally overlaps upstream, no upstream code vendored or
line-copied, whether upstream source was consulted **unclear — needs author
confirmation**). This is a documentation-only change: the per-file provenance
classifications below and the open author/legal decisions are **unchanged**.

> **Scope and honesty limits of this audit.**
> 1. The upstream repository `https://github.com/echonet/echo_CLIP` is **not checked out
>    in this workspace**, so *line-level similarity is not diff-verified here*. The
>    "line-level similarity" column records what is evidenced **in-repo** (source
>    comments, docstrings, verbatim strings) and marks unverifiable items explicitly.
> 2. Where origin cannot be established from the working copy, the entry says
>    **"unclear — needs author confirmation"** instead of asserting clean-room or derived.
> 3. This document does **not** assert a legal conclusion and does not claim that legal
>    review has occurred. It is an engineering inventory to support a pre-submission,
>    per-file audit by the author(s) with their institution.
> 4. Upstream EchoCLIP is described (per its `LICENSE` / Academic Software License) as
>    academic/non-profit use, derivatives allowed, **further transfer of the software
>    including derivative works restricted**, commercial use requiring separate
>    permission. That characterization is taken as given by the project brief and was
>    not re-verified against the upstream file in this workspace.

## Origin vocabulary

| Term | Meaning |
|------|---------|
| **original** | Written for this repository; no upstream EchoCLIP code or behavior copied. |
| **clean-room reimplementation** | Independently authored code, written to a functional spec, without copying upstream source. |
| **behaviorally-inspired** | Independently authored but deliberately reproduces upstream behavior/algorithm; may have been written after reading upstream. |
| **upstream-derived** | Content copied or near-copied from upstream (verbatim strings, line-level logic). |
| **unclear — needs author confirmation** | Cannot be determined from the working copy. |

## Per-file provenance table

| file | origin | did we read upstream source? | line-level similarity to upstream? | license | action needed |
|------|--------|------------------------------|-------------------------------------|---------|---------------|
| `echoclip/text.py` | Independently authored report normalizer; behavior intentionally overlaps upstream EchoCLIP report cleaning. | **Unclear — needs author confirmation.** No upstream checkout or vendored copy is present in-repo; the module docstring states it is independently authored (aligned 2026-10-08 to the conservative canonical wording — see the follow-up note above). | Not diff-verified (upstream absent). No upstream code body is vendored; overlap is textual/behavioral (uppercase, noise/tense/severity normalization, whitespace collapse, period insertion). | MIT (this repo) for the code as authored; **not** covered by upstream Academic Software License on the basis of this audit. | Author to confirm whether upstream echonet/echo_CLIP utils.py was read during authoring. If yes, decide whether to keep MIT on the independent expression or relabel as behaviorally-inspired/upstream-derived. |
| `echoclip/prompts.py` | **Upstream-derived** — verbatim published EchoCLIP zero-shot prompt sentence templates. | Effectively yes: the module docstring says the strings are "from EchoCLIP (echonet/echo_CLIP)" and they match the published prompt list. | Verbatim clinical prompt sentences (not independently authored text). | Upstream EchoCLIP Academic Software License / published-prompt terms. **Not MIT.** | Keep explicit upstream attribution and a non-MIT license note; do not relicense. Do not claim as original clinical IP. |
| `echoclip/official_parity.py` | **Upstream-derived / behaviorally-aligned reference logic.** Docstring: "aligned with echonet/echo_CLIP utils.py"; functions self-described as "Bit-aligned copy of … compute_regression_metric" and "Match … utils.crop_and_scale". | **Yes — indicated by the in-code description of alignment with upstream.** | High for `compute_regression_metric_official` and `crop_and_scale_official` (self-described bit-aligned copy / match). Not diff-verified here. | Upstream Academic Software License applies to the aligned logic. **Not MIT** as a verbatim/near-verbatim copy. | Before submission: either (a) keep and label clearly as upstream-derived under upstream terms, or (b) replace with an independently authored version plus citations. Requires author decision. |
| `echoclip/official_r0.py` | **Behaviorally-inspired** standalone official-style R0 evaluation that imports `crop_and_scale_official` / `compute_regression_metric_official` from `echoclip/official_parity.py`, which is itself self-described as a "bit-aligned copy". (Its docstring previously self-described as "clean-room"; it was aligned on 2026-10-08 to the conservative canonical wording — see the follow-up note above.) | Yes — via the imported `official_parity` reference logic; the module targets a documented upstream behavior (`UPSTREAM_COMMIT = echonet/echo_CLIP@main`). | Low in its own file body (new code), but the aligned math it calls inherits the `echoclip/official_parity.py` similarity. Not diff-verified. | MIT (this repo) for the authored body; the aligned logic it imports carries upstream terms. | **Newly added since the 2026-10-07 audit** (by the concurrent Python owner). Re-audit together with `echoclip/official_parity.py`; the "clean-room" docstring was reconciled on 2026-10-08, and the keep-and-label-vs-rewrite decision for this module remains open (see recommendation 1). |
| `echoclip/temporal.py` | **Original** (attention pool + Temporal Transformer on frozen embeddings). | No upstream echo_CLIP equivalent. | None identified. Uses standard `torch.nn` primitives. | MIT (this repo). | None. |
| `echoclip/loss.py` (`EFSoftContrastiveLoss`, `ClipLoss`, `TemporalClipLoss`) | **Original.** `ClipLoss` is the standard CLIP symmetric InfoNCE (Radford et al. common architecture), not echo_CLIP-specific; EF-soft multi-positive loss is authored here. | No upstream echo_CLIP loss source consulted for the EF-soft variant. | None identified for `EFSoftContrastiveLoss`. `ClipLoss` reproduces the standard CLIP loss formula. | MIT (this repo). | None. |
| `echoclip/cardiacclip.py` and `echoclip/cardiacclip_stub.py` | **Original adapter** (interface/template only; references upstream CardiacCLIP by URL). No CardiacCLIP code copied. | No CardiacCLIP source vendored. | None identified (download instructions + dataclasses authored here). | MIT (this repo); external CardiacCLIP weights under their own upstream Academic terms and **not bundled**. | None beyond keeping the "weights not redistributed" statement. |
| `echoclip/preprocess.py` | **Behaviorally-inspired** echo preprocessing. Docstring: "matches echonet/echo_CLIP utils"; `crop_and_scale` reproduces upstream letterbox/zoom behavior with added empty-slice guards. | Likely yes (docstring states alignment); **confirm.** | Moderate: same algorithm (aspect-ratio padding, `zoom=0.1`, cubic resize) with guard clauses added. Not diff-verified. | MIT (this repo) for the authored expression; algorithm overlaps upstream. | Author to confirm and decide whether to label behaviorally-inspired (recommended) vs upstream-derived. |
| `echoclip/zeroshot.py` (`compute_regression_score`) | **Behaviorally-inspired** official-style EF aggregation (argsort prompts per frame → mean over frames → median of top 20%). | Unclear — docstring cites upstream `compute_regression_metric` behavior; source not present for diff. | Moderate: same aggregation semantics; adds a `top_k < 1 → 1` guard absent upstream. | MIT (this repo). | Author to confirm whether upstream source was read; keep the documented behavioral-parity note. |
| `echoclip/structured_text.py` | **Original** mapping logic; **uses upstream-derived prompt strings** from `echoclip/prompts.py`. | Uses upstream strings by import (not a copy of upstream code). | None in code; upstream string content via `ZERO_SHOT_PROMPTS`. | MIT (this repo) for logic; embedded prompt strings remain upstream terms. | None; keep the "no invented clinical language" note. |
| `echoclip/prompts_ta.py` | **Original / clean-room** (TA EF/dilation captions authored here). | No. | None identified; docstring states these are not copies of the upstream echonet/echo_CLIP prompts_used.json. | MIT (this repo). | None. |
| `echoclip/model.py` | **Original** CLIP-style dual encoder reimplementation using standard CLIP architecture concepts. | No echo_CLIP source; based on public CLIP architecture. | None identified (standard CLIP components; no echo_CLIP-specific code). | MIT (this repo). | None. |
| `echoclip/cycle_sample.py` | **Original** frame sampling (includes `official_stride` semantics derived from upstream `0:min(40,T):2`). | Unclear; stride behavior cited from upstream example. | The `official_stride` index rule intentionally mirrors upstream frame selection. | MIT (this repo); stride rule is a factual protocol parameter. | None; keep the citation. |
| `echoclip/protocol.py` | **Original** experiment matrix (R0–R6 + Oracle-EDES). | No. | None identified. | MIT (this repo). | None. |
| `echoclip/calibrate.py` | **Original** calibration/conformal utilities. | No. | None identified (standard temperature scaling / split conformal). | MIT (this repo). | None. |
| `echoclip/clinical.py` | **Original** clinical metrics. | No. | None identified. | MIT (this repo). | None. |
| `echoclip/supervised.py` | **Original** EF regression heads. | No. | None identified. | MIT (this repo). | None. |
| `echoclip/efficiency.py` | **Original** parameter/timing helpers. | No. | None identified. | MIT (this repo). | None. |
| `echoclip/checkpoint.py`, `echoclip/config.py`, `echoclip/data.py`, `echoclip/eval.py`, `echoclip/utils.py`, `echoclip/__init__.py` | **Original** scaffold. | No. | None identified. | MIT (this repo). | None. |
| `scripts/eval_official_r0.py` | **Original script** that re-implements the upstream official R0 *procedure* (open_clip load, `official_stride`, EF 0–100). It calls `echoclip/official_parity.py` for aligned logic. | Uses upstream procedure description; no upstream script copied verbatim. | Low–moderate (procedure mirrored; code authored here). | MIT (this repo). | None; the aligned math it imports is covered by the `echoclip/official_parity.py` action above. |
| `scripts/compare_official_b0.py` | **Original** comparison harness. | No. | None identified. | MIT (this repo). | None. |
| `tests/test_official_b0_parity.py` | **Original** golden tests against fixed tensors (do not embed upstream code). | No. | None identified. | MIT (this repo). | None. |

## Datasets, weights, and third-party assets (not files in this repo)

| asset | status | license / handling |
|-------|--------|--------------------|
| Official EchoCLIP weights (`hf-hub:mkaichristensen/echo-clip`) | **Not redistributed** (loaded at runtime only). | Upstream EchoCLIP Academic Software License; obtain under upstream terms. |
| EchoNet-Dynamic videos / FileList.csv / VolumeTracings.csv | **Not redistributed** (user-obtained). | Stanford AIMI non-commercial Research Use Agreement; see `docs/OBTAIN_DATA_AND_WEIGHTS.md`. |
| CardiacCLIP weights | **Not redistributed**. | Upstream CardiacCLIP Academic terms; external comparator only. |
| Python dependencies (`torch`, `open_clip`, `transformers`, `timm`, …) | Installed from PyPI/HF. | Their own licenses; see requirements.txt and `requirements-paper.lock`. |

## Canonical `echoclip/text.py` characterization (single source of truth)

The identical sentence below is used in `NOTICE`, `ATTRIBUTION.md`, and `DATA.md`:

> `echoclip/text.py` is an independently authored report normalizer whose behavior intentionally overlaps upstream EchoCLIP report cleaning; no upstream code is vendored or line-copied in this repository, and whether the upstream echonet/echo_CLIP utils.py source was consulted is **unclear — needs author confirmation**.

**Reasoning.** The module docstring (aligned on 2026-10-08 to the canonical
wording) declares the normalizer independently authored with intentional behavioral
overlap and states that no upstream code is vendored or line-copied here. The
working copy contains no upstream source to diff against. The functional goals
(uppercase, noise stripping,
tense/severity normalization, whitespace/period normalization) match public echo report
preparation, which is what earlier docs described as "inspired by" before. Asserting
pure clean-room would overclaim what this audit can verify; asserting upstream-derived
would contradict the in-code declaration and the absence of any vendored copy. Hence the
conservative "independently authored with behavioral overlap; upstream consultation
unclear" framing, with author confirmation requested.

## Recommendations requiring a human decision

1. **`echoclip/official_parity.py`**, **`echoclip/preprocess.py`**, and the newly added
   **`echoclip/official_r0.py`**: decide between (a) keeping upstream-aligned logic with
   explicit upstream license labeling, or (b) an independently authored rewrite. This is
   the highest-risk set because `echoclip/official_parity.py` self-describes "bit-aligned
   copy" behavior and `echoclip/official_r0.py` imports that aligned logic (its "clean-room"
   docstring was reconciled on 2026-10-08, but the import is unchanged).
2. **`echoclip/text.py`**: author to confirm whether upstream echonet/echo_CLIP utils.py was read.
3. **`LICENSE` and `LICENSES/README.md`**: **reconciled on 2026-10-08** — both now carry
   the canonical echoclip/text.py sentence and an explicit MIT-scope boundary; the MIT license
   text itself is unchanged. See `reports/manuscript_hygiene_20261007.md` (addendum).
4. **Institution review:** run a per-file provenance audit with the author's institution
   before submission; this document is an engineering input, not a legal opinion.
