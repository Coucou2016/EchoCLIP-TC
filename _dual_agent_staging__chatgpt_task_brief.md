# EchoCLIP-TC dual-agent engineering task (for ChatGPT Pro/Plus)

## Role
You are the external senior engineer. Cursor lead will independently verify all claims against source and tests. Do not invent clinical MAE numbers. Demo ≠ clinical.

## Baseline
- Project: EchoCLIP-TC (Temporal, Calibrated) on EchoCLIP dual-encoder
- Workspace has NO git repo; treat as dirty working tree
- Source ZIP: `echoclip_tc_source_20260815.zip` (SHA-256 below when uploaded)
- Protocol: B0 / M1 / M2 / M4 locked in `echoclip/protocol.py` + `PAPER.md`

## Goals
1. Audit EchoCLIP-TC for remaining defects in protocol / clinical eval / training / checkpoint / temporal / calibration paths.
2. Propose **minimal** patches (unified diff or full file contents).
3. Provide short report + test plan. No fake clinical validation.

## Non-breakable boundaries
- Do not rewrite the dual encoder CLIP core as a new architecture.
- Never report demo/`--demo`/`simple_cnn`/`scratch_fallback` metrics as EchoNet or Nature Medicine EF MAE.
- Fit temperature / conformal **only** on VAL (`cal_manifest`); never on TEST.
- Do not require secrets, network downloads of private data, or committing.

## Known environment gaps (not code bugs to “fix” by inventing numbers)
- EchoNet-Dynamic videos often absent
- Official HF hub weights / GPU often unavailable
- Clinical MAE not measurable without real data + official weights

## Deliverables
1. Prioritized defect list (path, symptom, root cause, minimal fix)
2. Patches for real bugs only
3. Test plan: `python -m unittest discover -s tests -v` and `python scripts/validate.py --skip-eval`; optional `run_protocol.py --demo ...`
4. Explicit list of unverified risks / external blockers

## Acceptance criteria
- Unit tests pass; no regressions
- Protocol honesty rules preserved
- M4 without VAL cal-manifest must fail clearly in non-demo mode
- Subset ID locking must sample deterministically when given oversized pair lists
- Official hub embed_dim must sync with temporal aggregator when dimensions differ
