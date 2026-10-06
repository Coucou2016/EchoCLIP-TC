# Review response — P0 / high-ROI P1 fixes (2026-09-13)

Peer-review concerns mapped to code/doc changes in
https://github.com/Coucou2016/EchoCLIP-TC. **No clinical MAE numbers invented.**

## Major Concern → Change

| Concern | Status | Where |
|---------|--------|-------|
| **Annotation-assisted sampling on VAL/TEST** (ed_es/mixed contaminates primary eval) | Fixed | `echoclip/protocol.py` `assert_primary_eval_sampling` / `resolve_eval_sample_strategy`; R0–R6 force `eval_sample_strategy=uniform`; `scripts/eval_clinical.py` + `scripts/run_protocol.py` honor `val_sample_strategy` and hard-fail on val/test + ed_es/mixed |
| **Oracle / annotation-assisted upper bound** | Added | Experiment `ORACLE_EDES` (aliases Oracle-EDES), `annotation_assisted=True`, clearly labeled in PAPER.md / comparison table |
| **B0 not proven official parity** | Fixed | Default title **"EchoCLIP-based zero-shot baseline"**; `--paper` / `--official-reproduction` path with EF `0..100`, `official_stride` (`0:min(40,T):2`), open_clip preprocess when available, `allow_scratch_fallback=False` → `RuntimeError` |
| **M2 misframed as zero-shot temporal** | Fixed | R5/M2 description: EF-label-supervised temporal adaptation of frozen EchoCLIP (EchoCLIP-TA); PAPER.md + protocol notes |
| **Primary captions should be EF-only** | Fixed | `structured_text` default `include_dilation=False`; builder `--include-dilation` opt-in |
| **Missing supervised baselines** | Added | R2/S0 linear-ridge, R3/S1 MLP, R4/S2 temporal L1/Huber in `echoclip/supervised.py` + `scripts/train_supervised.py` (toy-tensor tests) |
| **Controlled ablation matrix** | Rebuilt | R0–R6 + Oracle-EDES; legacy B0/M1/M2/M4/S0–S2 aliases in `LEGACY_ALIASES` |
| **TemporalTransformer ignores mask / silent truncate** | Fixed | Padding `key_padding_mask` used; over-length → uniform subsample (not silent prefix) |
| **Dataset epoch seeding** | Fixed | `set_epoch` + epoch in `_item_seed`; VAL/TEST set epoch 0 |
| **EF soft multi-positive contrastive** | Added | `EFSoftContrastiveLoss` + `--ef-soft-contrastive` train flag + tests |
| **Calibration pseudo-logit limitation** | Documented + option | PAPER.md note; `fit_affine_logistic` / `--calibration-method affine_logistic` (EF&lt;50 + hooks 40/30) |
| **CITATION.cff wrong URL/title** | Fixed | Points to Coucou2016/EchoCLIP-TC; title EchoCLIP-TC / EchoCLIP-TA |
| **README hard-coded `E:\...`** | Fixed | Env placeholders (`ECHONET_ROOT`, `ECHOCLIP_ROOT`); `--demo` vs `--paper` |
| **LICENSE / provenance risk** | Noted | ATTRIBUTION.md + LICENSE footnote: MIT ≠ upstream Academic Software / EchoNet terms; TODO audit |

## Commands verified (this change)

```text
python -m unittest discover -s tests -v
python scripts/validate.py --skip-eval
```

## Remaining gaps (explicit)

- **CardiacCLIP comparison:** needs external weights/data (not bundled).
- **Adaptive conformal:** optional; not implemented.
- **Golden bit-exact EchoCLIP parity:** remaining gaps (tokenizer quirks, crop zoom, dtype) documented; `--paper` aligns grid + stride + no-scratch but does not invent MAE.
- **EchoNet / hub weights:** absent on many CI hosts → paper path correctly hard-fails; use `--demo` for wiring.
