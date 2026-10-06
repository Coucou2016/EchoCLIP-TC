# EchoCLIP-TC — Flat Repository (no subfolders, by design)

**This repository intentionally has no directory hierarchy.**
Every file — source code, documentation, manuscript, figures, results data, reports — sits in the
**repository root**, flat.

**This file (`README.md`) is the only entry point you need**, plus the machine-readable index
`FLAT_LAYOUT_MANIFEST.json` and the human-readable rationale `FLAT_LAYOUT.md`.

---

## Why is everything flat?

The goal of this repository is **complete, unobstructed, cross-agent reading**:

1. **Full-context ingestion by LLM agents.** ChatGPT / Claude / Cursor / any other agent that
   lists a repository, crawls a sitemap, or batch-downloads a tarball gets **100% of the
   project content in one flat namespace** — no directory recursion, no missed subtrees, no
   "I only read `src/`" partial views.
2. **Cross-review between independent agents.** A second agent can audit the manuscript against
   the actual code, the reported numbers against the raw JSON, and the review-response letters
   against the tests, without needing any external context about where things live.
3. **No hidden content.** With a flat layout, a reviewer can enumerate the entire repository in
   one request and verify that the count of files matches the manifest. Nothing can hide in a
   deep folder.
4. **Durable, link-stable paths.** Every file has a single, unique, top-level name, so any
   external note, issue, or prompt can reference it unambiguously.

---

## How the flat names are formed

Original path separators become a **double underscore** `__`:

```
papers/echoclip_tc_manuscript.md        ->  papers__echoclip_tc_manuscript.md
echoclip/model.py                       ->  echoclip__model.py
checkpoints/protocol/M2/metrics.json    ->  checkpoints__protocol__M2__metrics.json
reports/figures/fig1_....png            ->  reports__figures__fig1_....png
```

Files that were already at the repository root keep their original name
(the only exception is the pre-existing root `README.md`, renamed
`README_project_original.md`; **this** file takes the `README.md` slot).

The original project `README.md` is still available as `README_project_original.md`; the
`.gitignore` in this repository is a regenerated flat-repo version, not the original one.

**Prefix legend**

| Prefix | Original location | Content |
|---|---|---|
| *(none)* | repo root | license, citation, top-level docs, report bundle |
| `PAPER.md`, `DATA.md` | repo root | manuscript pointer / dataset protocol |
| `papers__` | `papers/` | **full manuscript** (MD + HTML) |
| `reports__` | `reports/` | **research report, review responses, collaboration logs, figures** |
| `figures__` | `figures/` | publication figures (PNG + PDF) |
| `echoclip__` | `echoclip/` | library source |
| `scripts__` | `scripts/` | experiment / evaluation CLI entry points |
| `tests__` | `tests/` | unit + end-to-end tests |
| `configs__` | `configs/` | YAML configs |
| `checkpoints__` | `checkpoints/` | **core result data** (metrics JSON, splits) |
| `docs__` | `docs/` | data/weights acquisition HOWTO |
| `data__` | `data/` | demo manifests/templates (no patient videos) |
| `notes__` | `notes/` | process notes |
| `LICENSES__` | `LICENSES/` | third-party license notes |
| `_dual_agent_staging__` | `_dual_agent_staging/` | Cursor↔ChatGPT collaboration records |
| `.github__workflows__ci.yml` | `.github/workflows/ci.yml` | CI workflow |

---

## Start here (recommended reading order)

### 1. Paper & reports
| File | What it is |
|---|---|
| `PAPER.md` | Manuscript entry point / claims index |
| `papers__echoclip_tc_manuscript.md` | **Full manuscript (source of truth, editable)** |
| `papers__echoclip_tc_manuscript.html` | Manuscript, rendered |
| `reports__research_report.md` | **Self-contained research report** |
| `reports__research_report.html` | Research report, rendered |
| `reports__research_report.pdf` | Research report, PDF |
| `report.html` | Root-level report bundle (rendered snapshot) |
| `README_project_original.md` | The original (pre-flattening) project README |

> The numeric filenames below are **stable**; if a regenerated report has no date yet, use the
> newest file whose name starts with the given prefix.

### 2. Core result data
| File | What it is |
|---|---|
| `checkpoints__paper_matrix__paper_matrix_summary.json` | Paper matrix aggregate summary |
| `checkpoints__paper_matrix_demo__paper_matrix_summary.json` | Demo matrix per-seed detail |
| `checkpoints__protocol__summary.json` | Protocol summary (M1/M2/M4/B0) |
| `checkpoints__protocol__cardiacclip_comparison.json` | CardiacCLIP baseline comparison |
| `checkpoints__label_efficiency__label_efficiency_table.json` | Label-efficiency table |
| `checkpoints__tc_smoke__clinical_metrics.json` | Clinical smoke metrics |
| `reports__attention_edes__attention_edes_summary.json` | Attention/ED-ES analysis |
| `reports__attention_edes__attention_edes.csv` | Attention analysis table |

### 3. Audit trail
| File | What it is |
|---|---|
| `reports__ZERO_TAIL_COMPLETE_20260915.md` | Final "zero-tail" closure record |
| `reports__review_response_p0_20260913.md` | P0 review response |
| `reports__review_response_p1_complete_20260914.md` | P1 review response |
| `reports__review_response_round2_20260914.md` | Round-2 response |
| `reports__review_response_round3_20260915.md` | Round-3 response |
| `reports__review_response_complete_20260914.md` | Aggregate response |
| `reports__echoclip_tc_five_round_collab_20260816.md` | Five-round collaboration log |
| `_dual_agent_staging__chatgpt_task_brief.md` | Task brief handed to the reviewer agent |

### 4. Data & weights
| File | What it is |
|---|---|
| `docs__OBTAIN_DATA_AND_WEIGHTS.md` | How to obtain EchoNet-Dynamic **and** the official EchoCLIP weights |
| `docs__OBTAIN_STATUS.md` | Acquisition status template |
| `DATA.md` | Dataset protocol / licensing |
| `DATA.md`, `data__examples__manifest_template.csv` | Manifest formats |

### 5. Code
* Library: `echoclip__*.py` — model, loss, data, zeroshot, temporal, cycle sampling,
  calibration, clinical metrics, protocol, checkpointing, structured text, prompts.
* Entry points: `scripts__*.py` — `scripts__train.py`, `scripts__train_supervised.py`,
  `scripts__eval_clinical.py`, `scripts__eval_official_r0.py`, `scripts__run_protocol.py`,
  `scripts__run_paper_matrix.py`, `scripts__run_seeds.py`, `scripts__run_label_efficiency.py`,
  `scripts__build_echonet_manifest.py`, `scripts__build_research_report_bundle.py`, …
* Tests: `tests__test_*.py` (protocol, calibration, zero-tail, official parity, e2e, …).

---

## Running the code from this flat repository

The flat layout is optimized for **reading and review**, not for `import echoclip` directly.
Python resolves `import echoclip` from the *directory* `echoclip/`, which does not exist here.

Two supported options:

1. **Rehydrate the original tree** (one command):

   ```powershell
   pwsh -File restore_flat_to_tree.ps1 -Destination ..\EchoCLIP-TC-tree
   ```

   This reconstructs `echoclip/`, `scripts/`, `tests/`, `papers/`, … from the `__`-flattened
   names, so `python -m pytest tests -q` and the CLI scripts work as documented.

2. **Read-only use**: treat `X__Y__Z.py` as `X/Y/Z.py` and read/normalize the import paths
   accordingly.

> The canonical, directly runnable repository layout lives in the main working copy; this flat
> repository is the **audit surface** for cross-agent review.

---

## Excluded from this repository (by design)

Large binaries are **not** stored here, because GitHub rejects files > 100 MB:

* model checkpoints `*.pt` (≈254–280 MB each)
* official EchoCLIP weights `open_clip_pytorch_model.bin` (≈606 MB) — obtain via
  `docs__OBTAIN_DATA_AND_WEIGHTS.md`
* EchoNet-Dynamic videos `*.avi` (license-gated clinical data)
* `*.zip` staging bundles, Python bytecode, caches

Regenerable demo images (`data/demo/images/*.png`) are also omitted; recreate with
`scripts__make_demo_data.py`.

See `FLAT_LAYOUT.md` for the full inclusion/exclusion policy.

---

## Machine-readable index

`FLAT_LAYOUT_MANIFEST.json` lists **every file** in this repository with:

* `flat_name` — the name in this flat repository
* `original_path` — its path in the original tree
* `category` — paper / report / code / test / result / doc / …
* `size_bytes`, `sha256`

Use it to enumerate the repository deterministically or to verify that an agent has read
everything.

---

## Status of the scientific claims

This repository contains **protocol-level and demo-scale results**. Clinical-grade numbers
require the license-gated EchoNet-Dynamic dataset and GPU runs; see
`docs__OBTAIN_DATA_AND_WEIGHTS.md` and `reports__ZERO_TAIL_COMPLETE_20260915.md` for the exact
state of each claim.
