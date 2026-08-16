# EchoCLIP-TC dual-agent acceptance report

**Date:** 2026-08-15  
**Lead:** Cursor (this agent)  
**External engineer:** ChatGPT Pro/Plus via cursor-ide-browser — **blocked** (see below; **retry also blocked** — `reports/echoclip_tc_dual_agent_chatgpt_retry.md`)  
**Workspace:** `E:\Projects\20260522-EchoCLIP`  
**Git:** no `.git` repository present — baseline recorded as **no git / dirty workspace**  
**Commit/push/PR:** **not performed** (not authorized) — all changes are **local-only**

> **Retry note (same day):** ChatGPT handoff was retried; IDE browser tabs still vanish before navigate/lock. Lead applied two further local patches (non-vacuous abstention fallback; hard-fail missing `external_clip` rematerialization). See `reports/echoclip_tc_dual_agent_chatgpt_retry.md`.

---

## 1. ChatGPT chat link(s)

| Item | Status |
|------|--------|
| ChatGPT conversation URL | **Unavailable** |
| Reason | `cursor-ide-browser` can create tabs (`browser_tabs` action `new`) but tabs vanish before `browser_navigate` / lock can attach (`Browser view not found` / `No browser tab available`). `open_resource` opened an external/system opener path that does not appear in the MCP tab list. |
| User action needed | Open ChatGPT in the **Cursor built-in Simple Browser / IDE browser** (logged-in session), then re-run dual-agent handoff; or paste `_dual_agent_staging/chatgpt_task_brief.md` + ZIP manually. |
| Auth blockers | No login/captcha/2FA prompt was reached — infrastructure failure before ChatGPT UI. |

Task brief prepared locally: `_dual_agent_staging/chatgpt_task_brief.md`

---

## 2. ZIP baseline

| Field | Value |
|-------|-------|
| Path | `E:\Projects\20260522-EchoCLIP\_dual_agent_staging\echoclip_tc_source_20260815.zip` |
| Size | 3,683,329 bytes (~3.51 MiB) |
| SHA-256 | `862a2f123a3b4ee5da9ee1ea7df20b02a213b244107624cd222227b448e49dc1` |
| Files | 124 |
| Secret scan | **0 hits** (`api_key`, `sk-…`, private keys, `AWS_SECRET`, `password=`) |
| Includes | `echoclip/`, `scripts/`, `configs/`, `tests/`, docs, requirements, small `data/demo` + `data/examples` |
| Excludes | `.git`, `.venv`, `checkpoints/`, `*.pt` weights, caches, `.env*` |

---

## 3. Audit findings and actual changes (Cursor lead)

Because ChatGPT could not be tasked, the lead audited and patched independently.

### Fixed

1. **`echoclip/protocol.py` — `write_subset_ids`**  
   Previously recorded seed/`n` but did **not** sample when given oversized pair lists. Now draws a deterministic `Random(seed)` subset of size `n`, with `already_sampled=True` for the EchoNet builder path that already called `subset_n`.

2. **`scripts/build_echonet_manifest.py`**  
   Passes `already_sampled=True` into `write_subset_ids` to avoid double-sampling.

3. **`scripts/run_protocol.py`**  
   - Docstring typo fixed.  
   - **M4 without VAL `cal_manifest` in non-demo mode now hard-fails** (was warning-only, which could produce uncalibrated metrics labeled M4).  
   - Demo / `--no-official` / `simple_cnn` sets `ECHOCLIP_SKIP_HUB=1` during eval to avoid useless hub download attempts.

4. **`scripts/eval_clinical.py` — `_collect_ef_labels`**  
   EF source labeling is now `manifest` / `text_parse_demo_only` / `mixed_manifest_and_text_parse` / `missing` instead of marking the whole run as text-parse if any single row was parsed.

5. **`echoclip/model.py` — `from_official_echo_clip`**  
   Syncs `config.embed_dim` to hub tower output dim before constructing / attaching the temporal aggregator; re-attaches temporal if dimensions diverge.

6. **`tests/test_protocol.py`**  
   Added `test_subset_ids_samples_when_oversized`.

### Correct / already solid (no change)

- B0/M1/M2/M4 specs in `echoclip/protocol.py` match `PAPER.md`.  
- Demo honesty flags on clinical metrics.  
- Calibration helpers, temporal shapes, public-manifest missing-root errors, builder mock layouts.  
- Prior demo protocol metrics under `checkpoints/protocol/*` are explicitly `demo_is_not_clinical`.

### External blockers (not “fixed” in code)

- EchoNet-Dynamic on disk  
- Official `hf-hub:mkaichristensen/echo-clip` weights  
- GPU / `convnext_base` for paper-scale B0/M2  
- Clinical EF MAE still **unmeasured** (requires real TEST + official weights)

---

## 4. Independent test results

| Gate | Result |
|------|--------|
| `python -m unittest discover -s tests -v` | **52 tests OK** (~4.2s) — was 51; +1 subset sampling test |
| `python scripts/validate.py --skip-eval` | **All validation steps passed** (smoke forward/backward, temporal, video stack, calibration helpers, manifest 64 pairs, unit tests) |
| `run_protocol.py --list` | OK |
| `run_protocol.py --demo --experiments B0 --dry-run` | OK |
| Clinical / EchoNet MAE | **Not run** — data/weights absent; demo numbers must not be claimed as clinical |

---

## 5. Unverified risks

- Hub `embed_dim` probing relies on `visual.output_dim` / `embed_dim` / `proj` heuristics; if a future open_clip build exposes none of these, sync may no-op (temporal still uses config dim).  
- Saving full `external_clip` into `.pt` checkpoints can be very large when hub weights load successfully — not addressed this pass.  
- TemporalTransformer still ignores padding masks (documented; pad_or_trim repeats last frame rather than true pad).  
- C: drive free space was ~8 GB during this session (Cursor SQLITE / browser instability risk).  
- ChatGPT did **not** review patches — only Cursor lead acceptance.

---

## 6. Code locality

- Changes live only in the local workspace.  
- **No git commit, push, or PR.**  
- Staging artifacts: `_dual_agent_staging/` (ZIP, pack script, brief, test logs).  
- This report: `reports/echoclip_tc_dual_agent_20260815.md`
