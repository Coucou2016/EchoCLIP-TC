# Metrics provenance for EchoCLIP-TA manuscript

## Newly computed (this build)

| Metric set | Path | Command summary | Seed | n | Key numbers |
|------------|------|-----------------|------|---|-------------|
| Dataset stats | computed from `E:\Datasets\EchoNet-Dynamic\FileList.csv` | Python csv pass 2026-11-07 | — | 10030 | EF mean 55.748, std 12.371; splits 7465/1288/1277; EF&lt;50/40/30 = 0.2239/0.1260/0.0622 |
| R0 zeroshot | `results/R0_test_subset_50.json` | `eval_r0_resume.py --limit 50 --experiment-id R0` official_stride/frames/CPU | 42 | 50 | MAE 11.981, RMSE 15.353, R² −0.205, bias +10.931, AUC50/40/30 = 0.768/0.793/0.853 |
| R1 zeroshot | `results/R1_test_subset_50.json` | `eval_r0_resume.py --limit 50 --pool mean --sample uniform --experiment-id R1` | 42 | 50 | MAE 12.023, RMSE 15.506, R² −0.229, bias +10.918, AUC50/40/30 = 0.777/0.768/0.853 |

Weights: `E:\Projects\20260522-EchoCLIP\models\echo-clip\open_clip_pytorch_model.bin` (0 missing keys).  
Manifest: `data/echonet_dynamic/test_subset_500.json` first 50 pairs.  
Echonet root: `E:\Datasets\EchoNet-Dynamic`.

## Existing logs (demo / non-clinical; not used in Table 1)

| Source | Notes |
|--------|-------|
| `reports/research_report.md` DEMO table | B0/M1 MAE 11.25, M2/M4 8.125 on synthetic n=32 scratch |
| `_dual_agent_staging/paper_matrix_demo.txt` | demo MAE 18.75 etc. |
| `reports/attention_edes/*` | demo_toy attention |

## Literature only (cited, not local reproduction)

| Claim | Source |
|-------|--------|
| EchoCLIP external EF MAE ≈ 7.1% | Christensen et al., Nat Med 2024 |
| EchoNet-Dynamic model MAE 4.1% / 6.0% | Ouyang et al., Nature 2020 |
| CardiacCLIP 1-shot ΔMAE −2.07 | Du et al., MICCAI 2025 / arXiv:2509.17065 |

## Not completed in this session (no invented numbers)

- Full `test_subset_500` (500) and full TEST (1277) zero-shot
- R0U16, Oracle-EDES
- Supervised R2–R4 training (`SCHEMA_VERSION=2` path)
- Adapted R5–R6 training + VAL-only calibration metrics
- Multi-seed paper matrix
