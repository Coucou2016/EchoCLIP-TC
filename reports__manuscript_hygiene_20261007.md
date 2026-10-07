# Manuscript hygiene check — 2026-10-07

**Scope:** `PAPER.md` and manuscript files under `papers/` (primary:
`papers/echoclip_tc_manuscript.md`). Scanned for internal development notes,
AI-collaboration traces, local host paths, and draft-only wording.

**Boundary respected:** no scientific claims, numbers, or experimental definitions were
changed. Clear-cut, non-scientific items were fixed; anything touching scientific
content or framing is listed below for the human author.

**Not in scope but flagged:** `reports/research_report.md` and the `reports/echoclip_tc_*`
handoff logs contain explicit `E:\Projects\20260522-EchoCLIP` host paths and dual-agent /
ChatGPT development narratives. They are internal working documents, not manuscript
files, so they were not rewritten; do not include them in a submission bundle.

---

## A. Fixes applied

| File | Location | Finding | Action |
|------|----------|---------|--------|
| `PAPER.md` | line 1 (title) | Title was `# EchoCLIP-TC / EchoCLIP-TA paper protocol` — stale project name. | Renamed to `# EchoCLIP-TA paper protocol`; repository URL continuity noted in the intro paragraph. |
| `PAPER.md` | "Table aggregate" line | Backticked `checkpoints/protocol/comparison.{json,md}` (brace-glob, generated at run time — not a literal file). | Reworded to plain text "writes checkpoints/protocol/comparison.json and comparison.md (generated on run)". |
| `PAPER.md` | Calibration honesty section | Backticked metrics.json (generated output, not a tracked file). | Reworded to "the `summarize_clinical` helper in generated metrics.json". |

`PAPER.md` contained **no** AI-collaboration traces, local host paths, or TODO/FIXME
markers.

## B. Items requiring author judgement (not changed)

These are in the manuscript `papers/echoclip_tc_manuscript.md`. Each is a development/
framing artifact rather than a scientific claim, but fixing them may affect how the draft
reads, so they are left to the author.

| Location | Exact text (abridged) | Problem | Suggested replacement |
|----------|------------------------|---------|------------------------|
| `papers/echoclip_tc_manuscript.md` line 3 | `**Status:** Methods manuscript draft.` | Draft-status marker that should not survive into a submission. | Remove before submission, or move to a private cover note. |
| `papers/echoclip_tc_manuscript.md` line 232 | "Drafting notes and agent handoff logs live under `notes/` / `reports/` (not manuscript body)." | Explicit **AI/agent-collaboration trace**; also references internal folders. | Delete this bullet from the manuscript; keep it only in an internal tracker. |
| `papers/echoclip_tc_manuscript.md` line 150 | "Local `checkpoints/protocol/*` metrics.json (synthetic demo, \(T=4\), scratch weights):" | References a developer output path; "scratch weights" is plumbing language. | Rephrase as "Synthetic demo smoke run (not clinical):" and drop the path. |
| `papers/echoclip_tc_manuscript.md` line 159 | "These numbers prove metrics I/O and calibration code paths execute." | Software-testing rationale inside a Results subsection. | Move to a footnote or the Methods/Implementation paragraph; keep the "not clinical" caveat. |
| `papers/echoclip_tc_manuscript.md` line 161 | `### 4.5 Figures (this draft)` | "(this draft)" is draft-only wording. | `### 4.5 Figures`. |
| `papers/echoclip_tc_manuscript.md` line 5 & 207 | `**Code:** https://github.com/Coucou2016/EchoCLIP-TC` | Legacy repo name (intentional URL continuity). | Keep, but if a new repo name is adopted, update once here and in `CITATION.cff`. |
| `papers/echoclip_tc_manuscript.md` line 16 | `| This work | EchoCLIP-TA (repo: EchoCLIP-TC) |` | Legacy name in the terminology ledger (continuity). | Keep as the single explicit continuity note, or change to `EchoCLIP-TA` if the ledger should be name-pure. |

## C. Verified clean

- No local Windows/host paths (`E:\...`, `C:\...`) in `PAPER.md` or
  `papers/echoclip_tc_manuscript.md`.
- No TODO / FIXME / "XXX" markers in `PAPER.md`.
- Manuscript placeholders for missing clinical numbers use the honest `待补充` marker and
  are explicitly labelled as not-clinical in the surrounding text — retained.
- `papers/echoclip_tc_manuscript.html` is a generated rendering of the `.md`; it will
  become stale if the `.md` is edited. Regenerate it with the repo's HTML build step
  after any manuscript change (build script not in this task's scope).

## D. Note for the author

`PAPER.md` uses a few "software-complete / software ready" framings (e.g. section
headings "One-shot runners (software complete)" and "Remaining gaps (external assets
only — software ready)"). These are defensible for an internal protocol doc; if
`PAPER.md` is ever shipped alongside the manuscript, consider softening them to avoid
reading as internal status reporting.

---

## Addendum — license-file reconciliation pass (2026-10-08)

Follow-up to the provenance work closing the outstanding item from that report
("`LICENSE` / `LICENSES/README.md` still carry the older echoclip/text.py clean-room SPDX
line"). Files changed in this pass: `LICENSE`, `LICENSES/README.md` (this report only
records the pass).

### What changed

- `LICENSES/README.md`: the `echoclip/text.py` row now carries the **same canonical
  sentence** used in `NOTICE`, `ATTRIBUTION.md`, and `DATA.md` (independently authored
  normalizer with behavioral overlap; upstream consultation **unclear — needs author
  confirmation**). Added an explicit MIT-scope boundary section (MIT covers original
  scaffold only; does not relicense upstream prompts / upstream-aligned logic / official
  weights / AIMI data). Added `echoclip/official_parity.py` and `echoclip/preprocess.py`
  rows flagged as upstream-aligned / needs author decision.
- `LICENSE`: replaced the scope footnote that pointed at a stale section title and said
  echoclip/text.py was clean-room. The MIT license text itself is **unchanged byte-for-byte**
  (verified against `git show HEAD:LICENSE`); only the post-`---` scope note was
  rewritten to state the MIT boundary, the no-redistribution statement, the canonical
  echoclip/text.py sentence, and the "not legal advice / institution audit" recommendation.

### Canonical sentence (now in `NOTICE`, `ATTRIBUTION.md`, `DATA.md`, `LICENSES/README.md`, `LICENSE`)

> `echoclip/text.py` is an independently authored report normalizer whose behavior
> intentionally overlaps upstream EchoCLIP report cleaning; no upstream code is
> vendored or line-copied in this repository, and whether the upstream
> echonet/echo_CLIP utils.py source was consulted is **unclear — needs author
> confirmation**.

### Repo-wide sweep (excluding `.git`)

Searched all files for the combination "echoclip/text.py + clean-room". Remaining hits and
their status:

| File:line | Content | Status |
|-----------|---------|--------|
| `reports/ZERO_TAIL_COMPLETE_20260915.md:33` | "…LICENSE, NOTICE, LICENSES/README.md; clean-room text.py + prompts_ta.py" | **Historical dated report** — describes the state as of 2026-09-15. Superseded by this pass. Not edited (records history); flagged here. |
| `reports/review_response_p1_complete_20260914.md:44` | "File-level upstream vs clean-room; ASL risk; no expanded MIT claim" | **Historical dated report** — superseded. Not edited; flagged here. |
| `echoclip/text.py` lines 3, 19, 72 | Module docstring/comment call the normalizer "clean-room" | **`.py` file — not edited** per task constraints. The docstring is now stricter than the canonical wording; a code change (align docstring to "independently authored … upstream consultation unclear") is **recommended for the Python owner** (see `PROVENANCE.md` recommendation 2). |
| `echoclip/official_r0.py` lines 1, 16, 60 | Module docstring/comments call the module "clean-room" | **`.py` file — not edited** per task constraints. **Newly added since the 2026-10-07 audit** by the concurrent Python owner. The module imports the self-described "bit-aligned" logic from echoclip/official_parity.py, so "clean-room" overstates it; `PROVENANCE.md` now records this as **behaviorally-inspired**. Recommended for the Python owner: align the docstring and re-audit with the parity module. |

Remaining `clean-room` mentions that are **not** about echoclip/text.py and are intentional
(describe the genuinely original prompts_ta.py captions, the "clean-room rewrite or
remove" offer, or the origin-vocabulary glossary): `PROVENANCE.md` (glossary lines 15,
30, 49, 82), `README.md:66`, `ATTRIBUTION.md:28,44,70`, `NOTICE:45,64`,
echoclip/prompts_ta.py:1,16, echoclip/structured_text.py:4,28,
tests/test_zero_tail.py:55. These are consistent with the canonical story and were
left unchanged.

### Remaining inconsistency I could not fix

`echoclip/text.py` (lines 3, 19, 72) and `echoclip/official_r0.py` (lines 1, 16, 60)
still self-describe as "clean-room", which is stricter than the canonical wording (and,
for the official R0 module, inconsistent with the echoclip/official_parity.py logic it imports).
Editing them would require touching `.py` files, which is outside my allowed change set
for this follow-up. Recommended resolution: the Python owner updates those
docstrings/comments, or the wording is accepted as-is with the caveat now recorded in
`PROVENANCE.md`.

