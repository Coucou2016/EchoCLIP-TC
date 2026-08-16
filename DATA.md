# EchoCLIP data format

## Overview

EchoCLIP trains on **paired** echocardiogram media (still frame or video) and clinical report text. Paths in manifests are relative to `manifest_dir` unless absolute.

This repository does **not** distribute real patient data. Use `scripts/make_demo_data.py` for synthetic pairs (pipeline smoke tests only), or prepare your own manifest from institutional data under appropriate IRB/consent and de-identification policies.

**Demo ≠ clinical.** Numbers computed on `data/demo/` must not be reported as EchoNet or Nature Medicine EF MAE.

## JSON manifest (recommended)

```json
{
  "pairs": [
    {"image": "videos/study001.avi", "text": "LV EJECTION FRACTION IS 55%. ..."},
    {"image": "frames/study002.png", "text": "NORMAL LEFT VENTRICULAR SIZE. ..."}
  ]
}
```

Templates: `data/examples/manifest_template.json`

### Optional clinical fields (EchoCLIP-TC / EchoNet)

`scripts/build_echonet_manifest.py` writes these extra keys while remaining DATA.md-compatible (`image` + `text` required):

| Key | Meaning |
|-----|---------|
| `ef` | Ejection fraction (%) — paper primary label |
| `edv` / `esv` | End-diastolic / end-systolic volume (mL) |
| `ed_frame` / `es_frame` | Frame indices from VolumeTracings (ED = larger tracing, ES = smaller) |
| `split` | TRAIN / VAL / TEST from FileList.csv |
| `captions` | List of official prompt sentences (EF ± dilation templates) |
| `file_name` | Original EchoNet file name |

`text` is always filled from **official** `echoclip.prompts` templates via `echoclip.structured_text` (no invented clinical language). Dilation sentences are included only when EDV maps onto the official mild/moderate/severe LV dilation prompts (absolute mL heuristic documented in `structured_text.py`).

## CSV manifest

Columns (header required):

| Column | Aliases | Description |
|--------|---------|-------------|
| `image_path` | `image`, `path` | Relative or absolute path to `.png`, `.jpg`, `.avi`, `.mp4`, etc. |
| `text` | `report`, `caption` | Full or excerpted echocardiography report |

Template: `data/examples/manifest_template.csv`

## Media types

| Type | Extensions | Training behavior |
|------|------------|-------------------|
| Still frame | `.png`, `.jpg`, `.jpeg`, `.bmp` | Single image per step; if `video_frames>1`, the still is repeated along T |
| Video | `.avi`, `.mp4`, `.mov`, `.mkv` | Cycle sampling (`random` / `uniform` / `ed_es` / `mixed`); default T=1 is one random frame |

Videos are letterbox-cropped and resized via `echoclip.preprocess.crop_and_scale` (640×480) before CLIP normalization (224×224 default). Sampling: `echoclip.cycle_sample`.

`configs/default.yaml` keeps `video_frames: 1` (demo). `configs/echonet_dynamic.yaml` uses `video_frames: 16`.

## Text preprocessing

Reports are uppercased and normalized with `echoclip.text.clean_report_text` (regex rules from [echonet/echo_CLIP](https://github.com/echonet/echo_CLIP)), then tokenized with `CLIPTokenizer` (77 tokens default).

## EchoNet-Dynamic (public clinical eval)

**License:** Stanford AIMI **non-commercial** research use. You must request the data; it is not redistributed here.

- Project page: https://echonet.github.io/dynamic/
- AIMI download: https://stanfordaimi.azurewebsites.net/

Expected layout:

```
<echonet_root>/FileList.csv
<echonet_root>/Videos/*.avi
<echonet_root>/VolumeTracings.csv    # optional, ED/ES indices
```

Build manifests:

```powershell
python E:\Projects\20260522-EchoCLIP\scripts\build_echonet_manifest.py `
  --echonet-root E:\data\EchoNet-Dynamic `
  --output-dir E:\Projects\20260522-EchoCLIP\data\echonet_dynamic `
  --subset-5000
```

`--subset-5000` writes `subset_5000.json` (seed=42) and locks IDs in
`subset_5000_ids.json` / `.txt`, approximating the EchoCLIP paper’s random
5000-study external protocol. Always also report the official **TEST** split
when you have it. If files are missing, the script errors with the download
instructions above.

### Other public sets

```powershell
python scripts\build_public_echo_manifest.py --dataset camus --root E:\data\CAMUS
python scripts\build_public_echo_manifest.py --dataset echonet_pediatric --root E:\data\EchoNet-Pediatric
python scripts\build_public_echo_manifest.py --dataset echonet_lvh --root E:\data\EchoNet-LVH
```


Point `configs/echonet_dynamic.yaml` at the generated `train.json` / `val.json` / `test.json` and set `manifest_dir` to the EchoNet root (so `Videos/...` resolves).

Paper-primary metrics:

```powershell
python E:\Projects\20260522-EchoCLIP\scripts\eval_clinical.py `
  --config E:\Projects\20260522-EchoCLIP\configs\echonet_dynamic.yaml `
  --checkpoint E:\path\to\best.pt `
  --manifest E:\Projects\20260522-EchoCLIP\data\echonet_dynamic\test.json `
  --cal-manifest E:\Projects\20260522-EchoCLIP\data\echonet_dynamic\val.json
```

Fit calibration **only** on VAL; do not retune on TEST.

## Validation

```powershell
python E:\Projects\20260522-EchoCLIP\scripts\validate.py `
  --manifest E:\path\to\manifest.json `
  --manifest-dir E:\path\to\data\root `
  --skip-eval
```

Or programmatically:

```python
from echoclip.data import load_manifest, validate_manifest
errors = validate_manifest(load_manifest("manifest.json"), root="data/root")
```

## Paper-scale training (user-provided)

Christensen et al. (Nature Medicine 2024) used **>1M** image–report pairs from **~225k** studies. EchoCLIP-TC instead adapts a **frozen** official (or local) dual encoder with a small temporal module on **public EchoNet-Dynamic** structured captions. To approach paper-grade zero-shot EF you still need:

1. EchoNet-Dynamic (AIMI, non-commercial) on disk  
2. Official EchoCLIP weights (`hf-hub:mkaichristensen/echo-clip`) or a local copy  
3. `vision_backbone: convnext_base` on GPU (`simple_cnn` is a Windows/CPU smoke fallback only)  
4. External validation: CAMUS / EchoNet-Pediatric / EchoNet-LVH via
   `scripts/build_public_echo_manifest.py` (see [PAPER.md](PAPER.md)); data not bundled

Official pretrained weights: https://github.com/echonet/echo_CLIP
