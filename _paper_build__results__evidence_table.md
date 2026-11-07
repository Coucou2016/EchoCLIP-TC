# 02_evidence_table (updated after local runs)

| claim_id | claim | evidence/source | strength | usable section | risk | status |
|----------|-------|-----------------|----------|----------------|------|--------|
| C1 | EchoNet-Dynamic local corpus has 10,030 A4C videos with TRAIN/VAL/TEST = 7465/1288/1277 | FileList.csv; Videos/ count | evidence-backed | Methods/Data | low | evidence-backed |
| C2 | Population EF mean 55.748, std 12.371; EF&lt;50/40/30 fractions 0.2239/0.1260/0.0622 | Computed 2026-11-07 from FileList.csv | evidence-backed | Methods/Results | low | evidence-backed |
| C3 | test_subset_500 is TEST-only, seed 42, n=500 | manifest meta | evidence-backed | Experiments | low | evidence-backed |
| C4 | EchoCLIP-TA freezes EchoCLIP towers and locks R0–R6 | PAPER.md; protocol.py | evidence-backed | Methods | low | evidence-backed |
| C5 | Local EchoCLIP weights load with 0 missing keys | eval load_meta | evidence-backed | Methods | low | evidence-backed |
| C6 | R0 official-stride zeroshot on first 50 of test_subset_500: MAE 11.981, RMSE 15.353, R² −0.205, bias +10.931, AUC50/40/30 = 0.768/0.793/0.853 | F:/EchoCLIP_paper_build/results/R0_test_subset_50.json | evidence-backed | Results | med (subset prefix) | evidence-backed |
| C7 | Christensen et al. external EF MAE ≈ 7.1% | Nat Med 2024 OA PDF | literature | Intro/Related | low | evidence-backed |
| C8 | R1 mean-pool uniform zeroshot on same 50: MAE 12.023, RMSE 15.506, R² −0.229, bias +10.918, AUC50/40/30 = 0.777/0.768/0.853 | F:/EchoCLIP_paper_build/results/R1_test_subset_50.json | evidence-backed | Results | med (subset prefix) | evidence-backed |
| C9 | Adapted R2–R6 clinical metrics | not executed this session (4GB GPU) | unsupported for numbers | Limitations | high | unsupported (no invented numbers) |
| C10 | Demo metrics are non-clinical | research_report.md | evidence-backed | honesty | low | evidence-backed |
| C11 | No AIMI RUA claim for Kaggle-mirror copy | user instruction | evidence-backed | Data availability | low | evidence-backed |
| C12 | SCHEMA_VERSION=2 | supervised_checkpoint.py | evidence-backed | Methods | low | evidence-backed |
