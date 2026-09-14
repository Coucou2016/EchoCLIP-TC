# Zero-tail closure checklist — 2026-09-15

**Product name:** EchoCLIP-TA (repo / remote: Coucou2016/EchoCLIP-TC; package: `echoclip`)  
**No clinical MAE/AUC invented.**

## Push status

| Field | Value |
|-------|-------|
| **Push** | Succeeded via GitHub git Data API (`git push` HTTPS :443 unreachable) |
| **Remote** | https://github.com/Coucou2016/EchoCLIP-TC |
| **Branch** | `main` |
| **Local content SHA** | `92fa2697b722653e663546804f021ffae3fa72f4` |
| **Remote tip SHA** | `6f26b841a76249a6bdebe506c6abc496375d601f` |
| **When** | 2026-09-15 |

Live tip: `gh api repos/Coucou2016/EchoCLIP-TC/commits/main --jq .sha`

## A. Engineering closure

| Item | Status | Evidence |
|------|--------|----------|
| Drop `stash@{0}` review_response | **Done** | Useful bits already on `main`; stash dropped |
| Audit TODO/FIXME/待补充 unfinished work | **Done** | Remaining 待补充 = clinical numbers / external assets only; Blocked entries have exact commands |
| R2–R4 `direct_regression` default in `run_protocol` | **Done** | `prediction_mode=direct_regression`; E2E tests |
| Seed propagation E2E (`train_seed` differs) | **Done** | `tests/test_supervised_e2e.py`, `tests/test_zero_tail.py` |
| Official R0 in `--paper --experiments R0`; R0U16 separate; parity honest | **Done** | `eval_official_r0.py` route; `official_reproduction_verified` gated |
| Primary train = uniform; `R5_EDESTRAIN` ablation only | **Done** | `echonet_dynamic.yaml` + protocol specs |
| EFSoft finite mask; `view_weight=0` | **Done** | `loss.py`; config `view_weight: 0.0` |
| Affine logistic paper default; adaptive conformal split or heuristic | **Done** | Paper default affine; adaptive off by default; VAL-scale/VAL-cal split when `n_cal≥8` |
| Stratified/filtered AUC bootstrap | **Done** | `clinical.bootstrap_metric_ci(stratified=True)` |
| `requirements-paper.lock` complete | **Done** | torch / open_clip / transformers pinned |
| LICENSE / NOTICE / SPDX / LICENSES | **Done** | `LICENSE`, `NOTICE`, `LICENSES/README.md`; clean-room `text.py` + `prompts_ta.py` |
| Branding EchoCLIP-TA in README/PAPER/manuscript/CITATION | **Done** | User-facing name unified; import path `echoclip` kept |

## B. Next-stage pipelines (implemented)

| Item | Status | Command |
|------|--------|---------|
| Label-efficiency runner | **Done** | `python scripts/run_label_efficiency.py --demo --epochs 1 --vision-backbone simple_cnn` |
| Efficiency metrics in metrics.json | **Done** | `echoclip/efficiency.py` + train/eval hooks |
| Attention/ED–ES analysis figure+CSV | **Done** | `python scripts/analyze_attention_edes.py` |
| CardiacCLIP adapter + template | **Done** | `echoclip/cardiacclip.py`; table auto-fills 待补充 |
| One-shot paper matrix | **Done** | `python scripts/run_paper_matrix.py --demo ...` / `--paper` hard-fails without assets |
| Manuscript submission structure | **Done** | Abstract…Conclusion; Related Work includes CardiacCLIP, EchoPrime, JACC Asia / multiview VLM, EchoJEPA |
| Research report regenerated | **Done** | `python scripts/build_research_report_bundle.py` |

## C. Data hunt

| Asset | Found? | Blocked command when missing |
|-------|--------|------------------------------|
| EchoNet-Dynamic videos / FileList | **No** (searched common roots; `ECHONET_ROOT` unset) | `set ECHONET_ROOT=<AIMI_root> && python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000 && python scripts/run_paper_matrix.py --paper` |
| Official EchoCLIP hub weights | **No** local successful load assumed | Hub `hf-hub:mkaichristensen/echo-clip` or `--official-checkpoint` |
| CardiacCLIP weights | **No** | `set CARDIACCLIP_WEIGHTS=<upstream.pt>` |
| GPU | Environment-dependent | Same paper matrix on CUDA host |

## D. Verification

| Gate | Status |
|------|--------|
| `python -m unittest discover -s tests -v` | **OK — 107 tests** |
| `python scripts/validate.py --skip-eval` | **OK** |
| `python scripts/run_paper_matrix.py --demo` (short R0,R1) | **OK** |
| `python scripts/analyze_attention_edes.py` | **OK** → `reports/attention_edes/` |
| `python scripts/run_label_efficiency.py --demo --dry-run` | **OK** |
| Push + sync to Coucou2016/EchoCLIP-TC | After this commit |

## External-only blockers (short)

1. EchoNet-Dynamic AIMI videos + FileList  
2. Official EchoCLIP hub / local weights  
3. CardiacCLIP upstream weights (comparator numbers)  
4. GPU for full 5-seed `convnext_base` paper tables  

Clinical EF MAE/AUC remain **待补充**. Demo metrics are never clinical.
