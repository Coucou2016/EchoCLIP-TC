# Obtain EchoNet-Dynamic + official EchoCLIP weights  
# 获取 EchoNet-Dynamic 与官方 EchoCLIP 权重

**Research use only / 仅限研究用途.** EchoNet-Dynamic is **non-commercial, non-clinical** (Stanford AIMI Research Use Agreement). Do not use outputs for diagnosis or patient care. This repo does **not** redistribute patient videos or hub weights.

**Demo ≠ clinical.** Numbers from `data/demo/` or `--demo` must never be reported as EchoNet / Nature Medicine EF MAE.

---

## Quick links / 快速链接

| Asset | Official pages |
|-------|----------------|
| EchoNet-Dynamic project | https://echonet.github.io/dynamic/ |
| AIMI dataset card | https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound |
| AIMI Shared Datasets portal | https://stanfordaimi.azurewebsites.net/ |
| AIMI Redivis (zip ≈ 7 GB) | https://stanford.redivis.com/datasets/66s1-2hsmzj5rn |
| EchoCLIP code | https://github.com/echonet/echo_CLIP |
| EchoCLIP weights (HF) | https://huggingface.co/mkaichristensen/echo-clip |
| EchoCLIP-R (retrieval; optional) | https://huggingface.co/mkaichristensen/echo-clip-r |
| Cite dataset DOI | https://doi.org/10.71718/yqp5-y078 |

Also documented in [DATA.md](../DATA.md), [PAPER.md](../PAPER.md), [README.md](../README.md).

---

## 1. EchoNet-Dynamic (videos + labels)

### License / 许可（必读）

Stanford University School of Medicine **Research Use Agreement** (on the project page):

- Personal, **non-commercial** research only — no sale / monetization.
- **Do not redistribute** the dataset or share your download link; each user must register.
- **Non-clinical / Research Use Only** — not FDA-reviewed; must not be used for diagnosis or patient care.
- No re-identification of subjects.

中文要点：仅个人非商业研究；禁止分享下载链接与数据副本；禁止临床诊疗用途；禁止再识别。

### Registration + download steps / 注册与下载

1. Read the agreement at https://echonet.github.io/dynamic/ .
2. Request access via AIMI:
   - Portal: https://stanfordaimi.azurewebsites.net/  
   - Or Redivis: https://stanford.redivis.com/datasets/66s1-2hsmzj5rn → **Apply for access** → download `EchoNet-Dynamic.zip` (~**7 GB** compressed).
   - Dataset card “Download here”: https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound  
3. After approval, download with **your** credentials only (do not forward the link).
4. Unzip to a local folder, e.g. `E:\Datasets\EchoNet-Dynamic\` (Windows) or `/data/EchoNet-Dynamic/`.

### Expected directory layout / 本仓库期望目录

`scripts/build_echonet_manifest.py` looks for (any of these roots work):

```
<ECHONET_ROOT>/
  FileList.csv
  Videos/*.avi          # ~10,030 apical-4-chamber clips, 112×112
  VolumeTracings.csv    # optional but recommended (ED/ES frame indices)
```

Also accepted: `<ECHONET_ROOT>/EchoNet-Dynamic/FileList.csv` (+ nested `Videos/`).

Contents (from AIMI / Nature 2020): EF, EDV, ESV, TRAIN/VAL/TEST splits in `FileList.csv`; LV tracings in `VolumeTracings.csv`.

### Disk space / 磁盘

| Item | Rough size |
|------|------------|
| Zip download | ~7 GB |
| Unpacked videos + CSVs | plan **≥12–15 GB** free |
| Manifests under `data/echonet_dynamic/` | small (MBs) |
| Paper checkpoints / matrix | tens of GB depending on seeds |

---

## 2. Official EchoCLIP weights

### Source / 来源

- Code + examples: https://github.com/echonet/echo_CLIP  
- Hub id used by this repo: **`hf-hub:mkaichristensen/echo-clip`**  
  - Files: https://huggingface.co/mkaichristensen/echo-clip  
  - Main weight: `open_clip_pytorch_model.bin` (~**606 MB**)
- Optional retrieval variant: `hf-hub:mkaichristensen/echo-clip-r` (not the default paper R0 path here)

Weights are **not** bundled; loaded at runtime via [open_clip](https://github.com/mlfoundations/open_clip). See `echoclip.model.OFFICIAL_ECHOCLIP_HUB` and `EchoCLIP.from_official_echo_clip`.

### Hugging Face login (if gated / rate-limited) / 登录

Official `echo_CLIP` examples recommend:

```powershell
pip install open-clip-torch huggingface_hub
huggingface-cli login
# paste token from https://huggingface.co/settings/token
```

Then either let the paper scripts pull the hub automatically, or smoke-test:

```python
from open_clip import create_model_and_transforms

model, _, preprocess = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip",
    # optional: precision="bf16", device="cuda"
)
print("ok", type(model))
```

Successful clinical runs must record `load_source=hf-hub:mkaichristensen/echo-clip` (not `scratch_fallback`).

### Local checkpoint alternative / 本地权重

If you already cached or exported a compatible checkpoint:

```powershell
python scripts\run_protocol.py --paper --experiments R0 --official-checkpoint E:\path\to\official.pt
```

### Env vars that affect hub load / 影响权重加载的环境变量

| Variable | Effect |
|----------|--------|
| `ECHOCLIP_SKIP_HUB=1` | Skip hub; **incompatible with `--paper`** |
| `ECHOCLIP_OFFICIAL_PARITY_OK=1` | Set only after you verify parity (see PAPER.md); marks `official_reproduction_verified` |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | Optional cache location for downloaded weights |
| `HF_ENDPOINT` | Optional hub endpoint (e.g. `https://hf-mirror.com` if `huggingface.co` times out) |
| `ECHONET_ROOT` | EchoNet-Dynamic root (`FileList.csv` + `Videos/`) |
| `ECHOCLIP_ROOT` | This repository root (optional convenience) |
| `CARDIACCLIP_WEIGHTS` | Optional comparator only — not EchoCLIP |
| `AIMI_DOWNLOAD_URL` | Optional personal signed zip URL for `scripts/obtain_echonet_zip.py` |

Unset `ECHOCLIP_SKIP_HUB` for paper runs. Do not invent clinical MAE if hub load fails.

Helpers: `scripts/obtain_echo_clip_weights.py`, `scripts/obtain_echonet_zip.py`.  
Latest local machine notes: [OBTAIN_STATUS.md](OBTAIN_STATUS.md).

---

## 3. Build manifests + run paper matrix

### Windows (cmd)

```bat
cd /d E:\Projects\20260522-EchoCLIP
set ECHONET_ROOT=E:\Datasets\EchoNet-Dynamic
set ECHOCLIP_ROOT=E:\Projects\20260522-EchoCLIP

python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
REM paper-grade pins (optional): pip install -r requirements-paper.lock

python scripts\build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000

python scripts\run_paper_matrix.py --paper --seeds 0,1,2,3,4
```

### Windows (PowerShell)

```powershell
cd E:\Projects\20260522-EchoCLIP
$env:ECHONET_ROOT = "E:\Datasets\EchoNet-Dynamic"
$env:ECHOCLIP_ROOT = (Get-Location).Path

python scripts\build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000
python scripts\run_paper_matrix.py --paper --seeds 0,1,2,3,4
```

### What the builder writes / 清单输出

Default `--output-dir` = `data/echonet_dynamic/`:

| File | Role |
|------|------|
| `manifest.json` | All pairs |
| `train.json` / `val.json` / `test.json` | Official splits |
| `subset_5000.json` + `subset_5000_ids.json` | Seed-42 EchoCLIP-style 5000 subset (`--subset-5000`) |

`manifest_dir` in `configs/echonet_dynamic.yaml` should resolve video paths as `Videos/...` relative to `ECHONET_ROOT`.

Smaller entry points (same assets):

```powershell
python scripts\run_protocol.py --paper --experiments R0
python scripts\eval_official_r0.py --paper
```

Wiring-only (no clinical claims):

```powershell
python scripts\run_paper_matrix.py --demo --epochs 1 --seeds 0,1 --vision-backbone simple_cnn
```

---

## 4. Hardware notes / 硬件

| Need | Guidance |
|------|----------|
| GPU | Strongly recommended for `convnext_base` + paper matrix; CPU/`simple_cnn` is demo-only |
| VRAM | Prefer a CUDA GPU with enough memory for ConvNeXt-Base + T=16 (exact GB depends on batch size) |
| Windows | If `timm`/`torchvision` fail with `_lzma` DLL errors, demo can use `simple_cnn`; **`--paper` hard-fails** without real hub weights |
| Network | First hub download needs Hugging Face access (~606 MB + tokenizer files) |

---

## 5. Honesty checklist / 诚实性检查

- [ ] EchoNet used only under AIMI non-commercial terms; no clinical deployment.
- [ ] `load_source` in metrics is the hub id (or documented local official checkpoint), not `scratch_fallback`.
- [ ] Report TEST and/or locked `subset_5000`; never substitute demo MAE.
- [ ] Calibration / conformal fitted on **VAL only**.
- [ ] Published EchoCLIP external EF MAE ≈ 7.1% is a **literature reproduction target**, not a local result until you finish `--paper` runs.

For locked experiment IDs (R0–R6), see [PAPER.md](../PAPER.md).
