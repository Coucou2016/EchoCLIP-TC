# EchoCLIP / EchoCLIP-TC (EchoCLIP-TA)

Vision-language foundation model for **echocardiogram interpretation**, implementing the contrastive CLIP-style architecture described in:

> Christensen et al., *Vision–language foundation model for echocardiogram interpretation*, Nature Medicine (2024).  
> [Paper](https://www.nature.com/articles/s41591-024-02959-y) · [Official inference repo](https://github.com/echonet/echo_CLIP)

This repository (**https://github.com/Coucou2016/EchoCLIP-TC**) provides a **trainable PyTorch implementation** of the dual-encoder CLIP (echo frames + report text) plus **EchoCLIP-TC** (Temporal, Calibrated) / soft rename **EchoCLIP-TA** (parameter-efficient temporal adaptation of frozen EchoCLIP). The Python package name remains `echoclip`.

## Architecture

| Component | Paper (EchoCLIP) | This implementation |
|-----------|------------------|---------------------|
| Image encoder | ConvNeXt-Base | `timm` backbone (`convnext_base` or `resnet18` for demos); `simple_cnn` if timm fails |
| Text encoder | CLIP decoder-only transformer (77 tokens) | 12-layer transformer + `CLIPTokenizer` (GPT-2 BPE) |
| Objective | Symmetric InfoNCE / CLIP loss | `ClipLoss` + `TemporalClipLoss` + optional EF soft multi-positive |
| Pretraining | LAION-400M CLIP → echo finetune | Optional `init_open_clip` or official `hf-hub:mkaichristensen/echo-clip` |
| Temporal (TC/TA) | None (frame encoder + mean) | `echoclip/temporal.py` — attention pool / Temporal Transformer `(B,T,D)→(B,D)` |
| Supervised baselines | — | S0/S1/S2 (`echoclip/supervised.py`): linear/ridge, MLP, temporal L1/Huber |
| Calibration (TC) | Uncalibrated cosine | Temperature **or** affine logistic, ECE, Brier, split-conformal, abstention |

## Project layout

```
echoclip/                 # Dual encoder, TC modules, zero-shot, clinical metrics
  protocol.py             # R0–R6 + Oracle-EDES matrix (aliases B0/M1/M2/M4)
  temporal.py             # Temporal Transformer / attention pooling
  supervised.py           # S0/S1/S2 EF heads
  cycle_sample.py         # random / uniform / ED-ES / mixed / official_stride
  calibrate.py            # temperature, affine logistic, ECE, conformal
  structured_text.py      # EF prompts (primary); EDV dilation optional ablation
scripts/
  run_protocol.py         # --demo vs --paper protocol runner
  eval_clinical.py        # Paper primary: EF + calibration → metrics.json
  train_supervised.py     # R2–R4 supervised baselines
  train.py                # Contrastive / temporal adaptation (R5)
PAPER.md                  # Experiment IDs, honesty rules
DATA.md                   # Manifest format + EchoNet-Dynamic
```

## Install

```powershell
cd $env:ECHOCLIP_ROOT   # or: cd <path-to-EchoCLIP-TC>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Windows CPU without timm/torchvision:
pip install -r requirements-minimal.txt
```

GPU recommended for `convnext_base` training; CPU works for demo/`resnet18`/`simple_cnn`.

**Windows note:** If `import timm` or `torchvision` fails with `_lzma` DLL errors, use `vision_backbone: simple_cnn`. Official EchoCLIP hub weights may fail to download; loaders fall back to `simple_cnn` **unless** `--paper` / `--official-reproduction` is set (then they **RuntimeError**).

## Quick start (demo — not clinical)

```powershell
python scripts\make_demo_data.py
python scripts\validate.py --skip-eval
python scripts\run_protocol.py --demo --experiments R0,R1 --vision-backbone simple_cnn
```

## EchoNet + paper path

EchoNet-Dynamic is **not** in this repo (Stanford AIMI non-commercial):

- https://echonet.github.io/dynamic/
- https://stanfordaimi.azurewebsites.net/

```powershell
$env:ECHONET_ROOT = "<AIMI_EchoNet-Dynamic>"
python scripts\build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000
# Optional ablation: --include-dilation  (primary captions are EF-only)

# Strict official reproduction (hard-fails without real EchoCLIP weights)
python scripts\run_protocol.py --paper --experiments R0

# Full primary matrix (aliases B0,M1,M2,M4 still work)
python scripts\run_protocol.py --experiments R0,R1,R5,R6
```

Edit `configs/echonet_dynamic.yaml` paths using env-style placeholders (`${ECHONET_ROOT}` documented in comments).

See [PAPER.md](PAPER.md) for the locked R0–R6 + Oracle-EDES matrix. **Do not invent clinical MAE.** Published EchoCLIP external ~7.1% EF MAE is from Christensen et al.; reproduce only with official weights + seed-42 subset / full TEST under `--paper`.

## Protocol notes

- **VAL/TEST primary:** `sample_strategy=uniform` (honors `val_sample_strategy`). `ed_es`/`mixed` → `ValueError` unless **Oracle-EDES**.
- **R5/M2:** EF-label-supervised temporal adaptation of frozen EchoCLIP — **not** a zero-shot temporal extension.
- **R0/B0 default name:** "EchoCLIP-based zero-shot baseline" until golden parity is proven.
- **Calibration:** pseudo-logit limitation documented; prefer `--calibration-method affine_logistic`.

## Attribution / license

See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE). MIT covers **this scaffold**; upstream EchoCLIP prompts/code and EchoNet terms may differ — do not assume MIT covers all derived materials without audit.
