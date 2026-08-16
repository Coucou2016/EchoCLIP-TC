# EchoCLIP / EchoCLIP-TC

Vision-language foundation model for **echocardiogram interpretation**, implementing the contrastive CLIP-style architecture described in:

> Christensen et al., *Vision–language foundation model for echocardiogram interpretation*, Nature Medicine (2024).  
> [Paper](https://www.nature.com/articles/s41591-024-02959-y) · [Official inference repo](https://github.com/echonet/echo_CLIP)

This repository provides a **trainable PyTorch implementation** of the dual-encoder CLIP (echo frames + report text) plus **EchoCLIP-TC** (Temporal, Calibrated): cycle-aware temporal aggregation, structured EchoNet captions, and calibration/conformal evaluation. The dual encoder is not rewritten — TC sits on top of it.

## Architecture

| Component | Paper (EchoCLIP) | This implementation |
|-----------|------------------|---------------------|
| Image encoder | ConvNeXt-Base | `timm` backbone (`convnext_base` or `resnet18` for demos); `simple_cnn` if timm fails |
| Text encoder | CLIP decoder-only transformer (77 tokens) | 12-layer transformer + `CLIPTokenizer` (GPT-2 BPE) |
| Objective | Symmetric InfoNCE / CLIP loss | `ClipLoss` + optional `TemporalClipLoss` (video-level InfoNCE) |
| Pretraining | LAION-400M CLIP → echo finetune | Optional `init_open_clip` or official `hf-hub:mkaichristensen/echo-clip` |
| Temporal (TC) | None (frame encoder + mean) | `echoclip/temporal.py` — attention pool / Temporal Transformer `(B,T,D)→(B,D)` |
| Calibration (TC) | Uncalibrated cosine | Temperature scaling, ECE, Brier, split-conformal EF intervals, abstention |

## Project layout

```
echoclip/                 # Dual encoder, TC modules, zero-shot, clinical metrics
  temporal.py             # Temporal Transformer / attention pooling
  cycle_sample.py         # random / uniform / ED-ES / mixed frame sampling
  calibrate.py            # temperature, ECE, conformal, abstention
  structured_text.py      # EF/EDV → official prompt captions
  clinical.py             # EF MAE/RMSE/R², AUC@50/40/30
scripts/
  make_demo_data.py       # Synthetic echo images + reports (pipeline only)
  train.py                # Training (1-frame demo or T-frame TC)
  infer.py                # Similarity, pacemaker, EF, retrieval
  eval.py                 # Retrieval R@k — diagnostic, not the paper primary
  eval_clinical.py        # Paper primary: EF + calibration → metrics.json
  run_protocol.py         # Table-1 modes B0/M1/M2/M4 → checkpoints/protocol/
  build_echonet_manifest.py
  build_public_echo_manifest.py  # CAMUS / Pediatric / LVH adapters
  validate.py             # Smoke + manifest + unit tests (+ optional eval)
  smoke_test.py
tests/
configs/default.yaml      # 1-frame demo training
configs/echonet_dynamic.yaml  # Frozen EchoCLIP-TC protocol (T=16)
configs/camus.yaml        # External stub
configs/echonet_pediatric.yaml
configs/echonet_lvh.yaml
PAPER.md                  # Experiment IDs, commands, honesty rules
DATA.md                   # Manifest format + EchoNet-Dynamic
```

## Install

```powershell
cd E:\Projects\20260522-EchoCLIP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Windows CPU without timm/torchvision:
pip install -r requirements-minimal.txt
```

GPU recommended for `convnext_base` training; CPU works for demo/`resnet18`/`simple_cnn`.

**Windows note:** If `import timm` or `torchvision` fails with `_lzma` DLL errors, use `vision_backbone: simple_cnn` (training auto-falls back when timm is missing). Official EchoCLIP hub weights may also fail to download here; loaders fall back to a local checkpoint or `simple_cnn` and will not crash.

## Quick start (demo data)

Demo pairs measure **pipeline correctness only**. They are not clinical results and must not be reported as EchoNet EF MAE.

```powershell
python E:\Projects\20260522-EchoCLIP\scripts\make_demo_data.py
python E:\Projects\20260522-EchoCLIP\scripts\validate.py --skip-eval
python E:\Projects\20260522-EchoCLIP\scripts\train.py
# Optional: python scripts\train.py --vision-backbone simple_cnn --epochs 2 --batch-size 8
```

Inference (retrieval / similarity) still uses `scripts/infer.py` and `scripts/eval.py`. For paper-style numbers use `scripts/eval_clinical.py` (below).

## EchoCLIP-TC

Protocol (frozen in `configs/echonet_dynamic.yaml`): `seed=42`, `video_frames=16`, temporal Transformer (2 layers), freeze vision/text towers, train the temporal aggregator (+ optional `logit_scale`). Text for EchoNet is **official EchoCLIP prompt templates** filled from EF/EDV — not invented report language.

### 1. Public data (EchoNet-Dynamic)

EchoNet-Dynamic is **not** in this repo. It is released by Stanford AIMI under a **non-commercial** research license:

- https://echonet.github.io/dynamic/
- https://stanfordaimi.azurewebsites.net/

```powershell
python scripts\build_echonet_manifest.py --echonet-root E:\data\EchoNet-Dynamic --subset-5000
```

If `FileList.csv` / `Videos/` are missing, the script exits with download instructions. Output JSON follows [DATA.md](DATA.md) (`image` + `text`) and adds `ef`, `edv`, `esv`, `split`, and `ed_frame`/`es_frame` when `VolumeTracings.csv` is present.

Edit paths in `configs/echonet_dynamic.yaml` (`echonet_root`, `manifest`, `manifest_dir`).

### 2. Train only the temporal module

When official weights are available (open_clip hub or `official_checkpoint`):

```powershell
python scripts\train.py --config configs\echonet_dynamic.yaml
```

When they are not (this environment): `--no-official` and `simple_cnn` still run the TC path on whatever manifest you pass (including demo, for a crash test only):

```powershell
python scripts\train.py --config configs\echonet_dynamic.yaml `
  --manifest E:\Projects\20260522-EchoCLIP\data\demo\manifest.json `
  --vision-backbone simple_cnn --no-official `
  --video-frames 4 --epochs 1 --batch-size 4 `
  --output-dir E:\Projects\20260522-EchoCLIP\checkpoints\tc_smoke
```

### 3. Clinical + calibration eval (paper primary)

```powershell
python scripts\eval_clinical.py `
  --config configs\echonet_dynamic.yaml `
  --checkpoint E:\Projects\20260522-EchoCLIP\checkpoints\echoclip_tc\best.pt `
  --manifest E:\Projects\20260522-EchoCLIP\data\echonet_dynamic\test.json `
  --cal-manifest E:\Projects\20260522-EchoCLIP\data\echonet_dynamic\val.json `
  --output E:\Projects\20260522-EchoCLIP\checkpoints\echoclip_tc\metrics.json
```

Writes MAE / RMSE / R², AUC at EF&lt;50/40/30, ECE, Brier, 90% conformal coverage, and abstention MAE. Temperature and conformal quantiles are **fit on the calibration manifest only**.

`scripts/eval.py` retrieval R@k is kept for sanity checks. **Do not put demo R@1 in a paper table.**

Zero-shot EF aggregation is unchanged from official echo_CLIP (rank prompts, median of top 20% values). TC only changes how the video vector `z_v` is obtained.

## Paper protocol (B0 → M4)

See [PAPER.md](PAPER.md) for the locked Table-1 experiment IDs, honesty rules, and the exact B0 path toward EchoCLIP’s published external EF MAE (~7.1% with official weights + seed-42 subset / full TEST).

```powershell
python scripts\run_protocol.py --list
python scripts\run_protocol.py --experiments B0,M1,M2,M4
# Windows CPU wiring only (NOT clinical):
python scripts\run_protocol.py --demo --experiments B0,M1 --vision-backbone simple_cnn
```

Metrics land in `checkpoints/protocol/<ID>/metrics.json`.

Cross-experiment table (after any protocol run, or standalone):

```powershell
python scripts\write_protocol_table.py --print
```

### Cross-dataset manifests

```powershell
python scripts\build_public_echo_manifest.py --dataset camus --root E:\data\CAMUS
python scripts\build_public_echo_manifest.py --dataset echonet_pediatric --root E:\data\EchoNet-Pediatric
python scripts\build_public_echo_manifest.py --dataset echonet_lvh --root E:\data\EchoNet-LVH
```

Missing data exits with download instructions. Config stubs: `configs/camus.yaml`, `configs/echonet_pediatric.yaml`, `configs/echonet_lvh.yaml`.

## Real clinical data (generic manifest)

Prepare JSON (paths relative to `manifest_dir`):

```json
{
  "pairs": [
    {"image": "videos/study001.avi", "text": "LV EJECTION FRACTION IS 60%. ..."},
    {"image": "frames/study002_plax.png", "text": "NORMAL LEFT VENTRICULAR SIZE. ..."}
  ]
}
```

Or CSV with columns `image_path,text`. See [DATA.md](DATA.md) for optional `ef` / `ed_frame` fields.

## Zero-shot API (Python)

```python
from echoclip import load_checkpoint
from echoclip.zeroshot import EchoCLIPInference

model, _ = load_checkpoint("checkpoints/best.pt", device="cpu")
engine = EchoCLIPInference(model)

# frame_emb: (n_frames, embed_dim) from preprocess.frames_to_tensor + encode
# or video-level z_v from model.encode_video(frames)  # (B, T, C, H, W)
```

Prompt templates: `echoclip.prompts` (official repo). Structured EchoNet captions: `echoclip.structured_text`.

## Pretrained EchoCLIP weights

Published weights live in [echonet/echo_CLIP](https://github.com/echonet/echo_CLIP) / Hugging Face `mkaichristensen/echo-clip` (not redistributed here). `EchoCLIP.from_official_echo_clip()` tries the hub, then a local path, then `simple_cnn`. Set `ECHOCLIP_SKIP_HUB=1` to skip the network.

## Trustworthiness and limitations

| Claim | Status in this repo |
|-------|---------------------|
| CLIP-style dual encoder + contrastive loss | Implemented and unit-tested |
| Cycle-aware temporal aggregation (EchoCLIP-TC) | Implemented; needs EchoNet + official weights for paper numbers |
| Calibration / conformal / abstention | Implemented; fit on val, report on test |
| Echo report text cleaning / prompts | Ported from echonet/echo_CLIP patterns |
| Paper-scale accuracy (EF MAE 7.1%, device AUC) | **Not reproducible here** without AIMI data + official weights |
| Demo metrics (`scripts/eval.py`) | In-batch retrieval R@k; demo pacemaker uses **keyword pseudo-labels** only |
| Clinical use | Research scaffold only — not a medical device |

Run `python scripts/validate.py --skip-eval` before trusting a local install. Metrics on `data/demo/` measure pipeline correctness, not clinical validity.

## Citation and attribution

- Cite Christensen et al., Nature Medicine 2024 (see `CITATION.cff`)
- Prompts/text rules: [ATTRIBUTION.md](ATTRIBUTION.md)
- EchoNet-Dynamic: Stanford AIMI non-commercial license (you must obtain the data yourself)
- License: [LICENSE](MIT)

## References

- Christensen, Vukadinovic, Yuan, Ouyang. Nature Medicine, 2024.
- Radford et al. CLIP. ICML 2021.
- Official code: https://github.com/echonet/echo_CLIP
- EchoNet-Dynamic: https://echonet.github.io/dynamic/

## License

MIT — see [LICENSE](LICENSE). Clinical deployment requires validation on your patient population and compliance with applicable regulations. EchoNet-Dynamic terms are separate and non-commercial.
