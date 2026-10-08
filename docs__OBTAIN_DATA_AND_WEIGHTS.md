# Obtain EchoNet-Dynamic + official EchoCLIP weights  
# 获取 EchoNet-Dynamic 与官方 EchoCLIP 权重

**Research use only / 仅限研究用途.** EchoNet-Dynamic is **non-commercial, non-clinical** (Stanford AIMI Research Use Agreement). Do not use outputs for diagnosis or patient care. This repo does **not** redistribute patient videos or hub weights.

**Demo ≠ clinical.** Numbers from `data/demo/` or `--demo` must never be reported as EchoNet / Nature Medicine EF MAE.

**License reality / 许可现实（2026 调研）.** No echo dataset in this space is fully open (CC-BY / CC0) for **LVEF-from-video regression**. Every usable source is **NonCommercial**, and several also require **ShareAlike**. Read §3 before promising an unencumbered open release — this directly interacts with this repo's MIT-licensed code and the open publishing plan. 本领域目前没有一个可用于"视频 → LVEF 回归"的数据集是完全开放许可（CC-BY / CC0）；所有可用来源均为**非商业**，部分还要求 **ShareAlike**。

---

## Quick links / 快速链接

| Asset | Official pages |
|-------|----------------|
| EchoNet-Dynamic project | https://echonet.github.io/dynamic/ |
| AIMI dataset card | https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound |
| AIMI Shared Datasets portal (legacy) | https://stanfordaimi.azurewebsites.net/ |
| AIMI Redivis (current; zip ≈ 7 GB) | https://stanford.redivis.com/datasets/66s1-2hsmzj5rn |
| EchoCLIP code | https://github.com/echonet/echo_CLIP |
| EchoCLIP weights (HF) | https://huggingface.co/mkaichristensen/echo-clip |
| EchoCLIP-R (retrieval; optional) | https://huggingface.co/mkaichristensen/echo-clip-r |
| CAMUS (fully open, no registration) | https://www.creatis.insa-lyon.fr/Challenge/camus/ |
| EchoNet-Pediatric | https://echonet.github.io/pediatric/ |
| EchoNet-LVH | https://echonet.github.io/lvh/ |
| MIMIC-IV-ECHO (PhysioNet) | https://physionet.org/ (search project name `MIMIC-IV-ECHO`) |
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
- The agreement also contains a **clause forbidding derivative works + redistribution**. See §3 for what this means for publishing adapted/adapter weights.

中文要点：仅个人非商业研究；禁止分享下载链接与数据副本；禁止临床诊疗用途；禁止再识别；协议含"禁止衍生作品与再分发"条款。

### Access path (2026): Stanford Redivis / 访问路径

EchoNet-Dynamic access has **moved to Stanford Redivis**. Registration is per-person and approval is gated by the RUA.

1. Create a **Redivis** account using an **institutional email**.
2. Open the dataset: https://stanford.redivis.com/datasets/66s1-2hsmzj5rn
3. Join the **Stanford AIMI** organization (prompted from the dataset page).
4. Click **"Apply for access"** and complete the **Research Use Agreement (RUA)** form.
5. After approval, download `EchoNet-Dynamic.zip` (~**7 GB** compressed) **with your own credentials**.

> **Superseded path.** The older **AIMI Azure blob** portal (https://stanfordaimi.azurewebsites.net/) is **superseded** by Redivis for EchoNet-Dynamic. Links remain for reference only — do not assume they provide a direct download.

中文要点：EchoNet-Dynamic 的获取已迁至 Stanford Redivis：用院校邮箱注册 Redivis → 打开数据集页 → 加入 Stanford AIMI 组织 → 点击 "Apply for access" 并填写 RUA 表单 → 审批后用**本人凭据**下载 `EchoNet-Dynamic.zip`（约 7 GB）。旧的 AIMI Azure blob 门户已被取代，仅为存档保留；它**不是**直接下载入口。

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

## 2. Other public echo datasets (2026 inventory) / 其他公开数据集

EchoNet-Dynamic is **not** the only current source of EF-labeled echo video. The table below records what a 2026 research pass verified, what each set can and cannot be used for, and what remains unverified. Sizes/licenses marked **UNVERIFIED** were not confirmed from a primary source in that pass and must be re-checked before relying on them.

**可以**用哪个数据集做什么 · 哪些尚未核实。

### 2.1 CAMUS — fastest fully-open option, **no registration** / 最快、完全开放、**无需注册**

- **Access:** direct download from the **CREATIS Girder** portal; **collection id `6373703d73e9f0047faa1bc8`**; download size ≈ **3.83 GB**. **No registration or gate.**
- **License:** **CC BY-NC-SA 4.0** — **not** plain CC-BY — plus a **mandatory citation**:
  Leclerc et al., *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography*, **IEEE TMI 2019**, doi:[10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516).
- **Contents:** A4C cine + EF + ED/ES indices (also A2C). Small (~**500** patients).
- **Use it for:** immediate pipeline smoke testing and a first A4C cine + EF path with **zero waiting**.
- **Do not use it as:** the primary adaptation corpus — at ~500 patients it is too small.

中文要点：CAMUS 可直接从 CREATIS Girder 下载（集合 id `6373703d73e9f0047faa1bc8`，约 3.83 GB），**无需注册**；许可为 **CC BY-NC-SA 4.0**（不是 CC-BY），须引用 Leclerc 等 IEEE TMI 2019。含 A4C 电影 + EF + ED/ES。适合立刻做流程冒烟测试，但约 500 例，太小，不适合作为主要适配语料。

### 2.2 MIMIC-IV-ECHO (PhysioNet) — new in 2026 / 2026 新增

- **Versions:** v1.0 published **2026-03-10**; **v1.0.1** on **2026-08-25**.
- **Scale:** **206,488** studies with structured measurements; **~524k DICOMs** across **~7,243** studies.
- **Labels:** `lvef`, `biplane_lvef`, `lvef_3d` in `structured_measurement.csv`.
- **Access:** requires PhysioNet **credentialing** + **CITI "Data or Specimens Only Research"** training + a **DUA**. **Delays can reach ~45 days.**
- **Use it for:** large-scale EF measurement linkage; strong candidate as a pooled external validation source once credentialed.
- **UNVERIFIED:** DICOM download size.

中文要点：MIMIC-IV-ECHO v1.0（2026-03-10）、v1.0.1（2026-08-25）；约 206,488 项结构化测量、约 52.4 万个 DICOM（约 7,243 项研究）；`structured_measurement.csv` 含 `lvef` / `biplane_lvef` / `lvef_3d`。需 PhysioNet 认证 + CITI "Data or Specimens Only Research" 培训 + DUA，审批可能长达约 45 天。下载体积**未核实**。

### 2.3 MIMIC-IV-ECHO-Ext-LVVOLUMES-A4C-ROI v1.0.0 — best external-validation option / 最佳外部验证选项

- **Released:** 2026-02-26.
- **Contents:** **1,064** A4C DICOM videos, **806** patients; `LVEF_A4C` / `LVEF_BP` plus LVEDV / LVESV; also **256×256 MP4/NPZ** and **ROI masks**.
- **Designed explicitly as an external test set for EchoNet-trained models.**
- **Access:** same PhysioNet credentialing as §2.2.
- **Recommendation:** this is the **best external-validation option** for an EchoNet-trained / adapted EF model because it is purpose-built and already preprocessed.

中文要点：2026-02-26 发布；1,064 段 A4C DICOM 视频、806 例患者；含 `LVEF_A4C` / `LVEF_BP` 及 LVEDV / LVESV，另提供 256×256 MP4/NPZ 与 ROI 掩膜。明确面向 EchoNet 训练模型的外部测试，是**最佳外部验证选项**；需与 §2.2 相同的 PhysioNet 认证。

### 2.4 EchoRisk (MICCAI 2026) — currently unavailable / 目前不可用

- Exists (2,159 EF-labeled videos incl. A4C) and is CC BY-NC-SA, **but its Synapse access is currently CLOSED with no confirmed future release**.
- Treat as **effectively unavailable**; do not plan around it.

中文要点：EchoRisk（MICCAI 2026，含 A4C 的 2,159 段带 EF 视频，CC BY-NC-SA）目前 Synapse 访问**已关闭**，且无确定的重开时间，视为**实际不可用**。

### 2.5 EchoNet-Pediatric — same AIMI gating / 同为 AIMI 门控

- **7,643** videos, A4C + PSAX; EF via **area-length** method.
- Access via Stanford AIMI (same non-commercial gating as EchoNet-Dynamic).
- **UNVERIFIED:** approval timing.

### 2.6 EchoNet-LVH — **no native EF** / 无原生 EF

- **12,000** PLAX videos; **wall thickness only — NO native EF.**
- Same AIMI gating.
- **Cannot validate EF regression.** Use only for wall-thickness / hypertrophy style auxiliary tasks.
- **UNVERIFIED:** approval timing.

中文要点：EchoNet-Pediatric（7,643 段，A4C + PSAX，EF 由 area-length 计算）与 EchoNet-LVH（12,000 段 PLAX，**仅有室壁厚度、无原生 EF**）均为 AIMI 门控。LVH **不能**用于 EF 回归验证。审批时间**未核实**。

### 2.7 Video-/still-only sets — pipeline testing only, **NOT** for EF validation / 仅用于流程测试，**不能**做 EF 验证

| Dataset | What it has | Why it cannot validate EF | License note |
|---------|-------------|---------------------------|--------------|
| TMED-2 | **stills only** | no EF | non-commercial, apply-for-access |
| HMC-QU | 162 A4C + 130 A2C | MI / RWMA labels only | Kaggle license metadata **self-contradictory** |
| CardiacUDA / cardiacUDC | A4C 364 videos (public release) | no EF labels reported | **internally inconsistent** — Apache-2.0 declared vs CC BY-NC 4.0 stated |
| RVENet | RVEF | **RVEF only**, not LVEF | — |
| Unity Imaging | incl. a multi-expert EF validation set | usable only for EF cross-checks, not adaptation | main page **CC BY-NC-ND 4.0** vs project page **CC BY-NC-SA 4.0** — **conflict** |
| TED (CREATIS) | 98 A4C full-cycle | small; no login, but NC-SA | **CC BY-NC-SA**, no login required |

**UNVERIFIED:** sizes for RVENet, TMED-2, HMC-QU, Unity Imaging; Unity/CardiacUDA/HMC-QU license conflicts; whether CardiacUDA/cardiacUDC exposes EF anywhere.

中文要点：TMED-2（仅静态图、无 EF、需申请）、HMC-QU（162 A4C + 130 A2C，仅 MI/RWMA，Kaggle 许可元数据自相矛盾）、CardiacUDA/cardiacUDC（公开版 A4C 364 段；许可**自相矛盾**：声明 Apache-2.0 vs 陈述 CC BY-NC 4.0）、RVENet（**仅 RVEF**）、Unity Imaging（含多专家 EF 验证集；主页 CC BY-NC-ND 4.0 与项目页 CC BY-NC-SA 4.0 **冲突**）、TED（CREATIS，98 段 A4C 全周期，无需登录，NC-SA）。以上均可用于流程测试，**不能**用于 EF 回归验证；相关体积与许可冲突**未核实**。

---

## 3. License reality — read before planning a release / 许可现实（发布前必读）

> **There is no fully-open (CC-BY / CC0) LVEF-from-video dataset in this space.** Every usable source is **NonCommercial**; several are also **ShareAlike**. Plan the release accordingly.

| Source | License | Practical consequence for *this* repo |
|--------|---------|---------------------------------------|
| AIMI EchoNet-Dynamic / -Pediatric / -LVH | Non-commercial RUA, with a **clause forbidding derivative works + redistribution** | Publishing **metrics** from permitted non-commercial research is normal; publishing **fine-tuned / adapter model weights is a gray zone at best**. Do **not** redistribute data or checkpoints. |
| CAMUS / TED / EchoRisk | **CC BY-NC-SA** | Derivatives — **possibly including adapted weights** — must stay **NonCommercial + ShareAlike**. You **cannot** place MIT/Apache on those weights and call them unencumbered. |
| MIMIC / PhysioNet | Credentialed DUA | **No redistribution**; commercial use restricted by the DUA. |

**Why this matters here.** This repository's **code** is MIT-licensed, and the stated plan is open publishing. Training/adapting on EchoNet-AIMI, CAMUS, TED or EchoRisk data does **not** let you slap MIT/Apache onto the resulting **weights**: the AIMI RUA restricts derivative works + redistribution, and the CC BY-NC-SA sets force NonCommercial + ShareAlike on derivatives. Publish **code, prompts, configs and evaluation protocol** openly as MIT; treat **adapted weights and any derived data artifacts as a separate, restricted (NonCommercial, and ShareAlike where applicable) release**, or do not release them at all. Metrics from permitted non-commercial research remain reportable.

中文要点：**本领域没有完全开放（CC-BY/CC0）的"视频 → LVEF"数据集**。AIMI EchoNet 系列为非商业 RUA，含**禁止衍生作品与再分发**条款：发表由许可的非商业研究得到的**指标**是正常的，但发表**微调 / 适配器权重属灰色地带**，且**禁止**再分发数据或检查点。CAMUS / TED / EchoRisk 为 **CC BY-NC-SA**，其衍生（**可能包括适配权重**）必须保持非商业 + ShareAlike，**不能**对这类权重套用 MIT/Apache 并宣称无限制。MIMIC / PhysioNet 需凭据与 DUA，**禁止再分发**、限制商业用途。因此：代码/提示词/配置/评测协议可继续以 MIT 开放；**适配权重与任何派生数据产物应按受限（非商业，必要时 ShareAlike）方式单独处理或不予发布**。许可的非商业研究所得**指标**仍可报告。

---

## 4. Mirror risk warning / 镜像风险警告

Kaggle / Hugging Face "EchoNet-Dynamic" **mirrors** (e.g. those declaring **Apache-2.0**) directly contradict the Stanford RUA and **bypass the required individual registration**. Treat them as **non-compliant** — do not download or redistribute from them. The same applies to **HF "CAMUS" conversions that relabel** the data as Apache / MIT / CC0.

Always obtain from the authoritative source:

- EchoNet-Dynamic → **Stanford Redivis** (https://stanford.redivis.com/datasets/66s1-2hsmzj5rn) / AIMI.
- CAMUS → **CREATIS** (https://www.creatis.insa-lyon.fr/Challenge/camus/).

中文要点：Kaggle / HuggingFace 上声明 **Apache-2.0** 的 "EchoNet-Dynamic" 镜像与 Stanford RUA 直接冲突，且绕过**必须的个人注册**，应视为**不合规**，不得下载或再分发。把 CAMUS 重新标注为 Apache/MIT/CC0 的 HF 转换版同理。请始终从权威来源获取：EchoNet-Dynamic 走 **Stanford Redivis**/AIMI，CAMUS 走 **CREATIS**。

---

## 5. Recommended plan / 推荐行动方案

Apply **now**, in parallel, because the gates have very different latencies:

1. **EchoNet-Dynamic (primary adaptation corpus)** — complete the Redivis RUA now (§1). This is the long gate but the main corpus.
2. **CAMUS (immediate smoke test, no gate)** — download from CREATIS immediately (§2.1) and validate the A4C cine + EF pipeline end-to-end while waiting. Accept CC BY-NC-SA + citation obligations.
3. **MIMIC-IV-ECHO-Ext-LVVOLUMES-A4C-ROI (external validation)** — start **PhysioNet credentialing + CITI training + DUA in parallel now** (§2.2 / §2.3); approval can take **~45 days**, and this is the best external-validation set.

中文要点：三条线并行推进——(1) 现在就走 Redivis 完成 EchoNet-Dynamic RUA（主适配语料，门控最慢）；(2) 立即从 CREATIS 下载 CAMUS（无需门控），先把 A4C 电影 + EF 流程跑通，同时接受 CC BY-NC-SA 与引用义务；(3) 现在就启动 PhysioNet 认证 + CITI 培训 + DUA 以获取 MIMIC-IV-ECHO-Ext-LVVOLUMES-A4C-ROI（最佳外部验证集），审批可能约 45 天。

---

## 6. Official EchoCLIP weights

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

> Reminder (§3): do **not** redistribute official hub weights or any checkpoint adapted on license-gated data.

### Env vars that affect hub load / 影响权重加载的环境变量

| Variable | Effect |
|----------|--------|
| `ECHOCLIP_SKIP_HUB=1` | Skip hub; **incompatible with `--paper`** |
| `ECHOCLIP_OFFICIAL_PARITY_OK=1` | Set only after you verify parity (see PAPER.md); marks `official_reproduction_verified` |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | Optional cache location for downloaded weights |
| `HF_ENDPOINT` | Optional hub endpoint (e.g. `https://hf-mirror.com` if `huggingface.co` times out) |
| `ECHONET_ROOT` | EchoNet-Dynamic root (`FileList.csv` + `Videos/`) |
| `CAMUS_ROOT` | CAMUS root (patient folders with `Info_*.cfg`) |
| `ECHOCLIP_ROOT` | This repository root (optional convenience) |
| `CARDIACCLIP_WEIGHTS` | Optional comparator only — not EchoCLIP |
| `AIMI_DOWNLOAD_URL` | Optional personal signed zip URL for `scripts/obtain_echonet_zip.py` |

Unset `ECHOCLIP_SKIP_HUB` for paper runs. Do not invent clinical MAE if hub load fails.

Helpers: `scripts/obtain_echo_clip_weights.py`, `scripts/obtain_echonet_zip.py`.  
Latest local machine notes: [OBTAIN_STATUS.md](OBTAIN_STATUS.md).

---

## 7. Build manifests + run paper matrix

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

### Other public sets / 其他公开数据集入口

```powershell
$env:CAMUS_ROOT = "<CAMUS_ROOT>"
python scripts\build_public_echo_manifest.py --dataset camus --root $env:CAMUS_ROOT
```

### What the builder writes / 清单输出

Default `--output-dir` = `data/echonet_dynamic/`:

| File | Role |
|------|------|
| `manifest.json` | All pairs |
| `train.json` / `val.json` / `test.json` | Official splits, emitted separately (each carries its `split` field) so TRAIN/VAL/TEST can never leak into each other |
| `r0_external_anchor_5000.json` + `r0_external_anchor_5000_ids.json` / `.txt` | **R0 historical/external anchor only** (`--subset-5000`, seed 42). Samples from the *mixed* TRAIN+VAL+TEST pool to approximate the EchoCLIP paper's random 5000-study external protocol; it embeds provenance (`pool: mixed_train_val_test`, `disjoint_from_train: false`). **Must NOT be used to evaluate adapted models (R2–R6)**, because those trained on EchoNet TRAIN and could score on a TRAIN case. |
| `test_subset_<N>.json` + `_ids.json` | **TEST-only** subset for label-efficiency / adapted-model runs (`--test-subset N`). Always also report the full official TEST split. |

> Earlier versions of this repo wrote a single ambiguous `subset_5000.json`; that name is superseded. Always report the official **TEST** split for adapted models, plus the R0 anchor for zero-shot comparability.

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

## 8. Hardware notes / 硬件

| Need | Guidance |
|------|----------|
| GPU | Strongly recommended for `convnext_base` + paper matrix; CPU/`simple_cnn` is demo-only |
| VRAM | Prefer a CUDA GPU with enough memory for ConvNeXt-Base + T=16 (exact GB depends on batch size) |
| Windows | If `timm`/`torchvision` fail with `_lzma` DLL errors, demo can use `simple_cnn`; **`--paper` hard-fails** without real hub weights |
| Network | First hub download needs Hugging Face access (~606 MB + tokenizer files) |

---

## 9. Honesty checklist / 诚实性检查

- [ ] EchoNet used only under AIMI non-commercial terms; no clinical deployment.
- [ ] No license-gated data or adapted weights redistributed (see §3 / §4).
- [ ] `load_source` in metrics is the hub id (or documented local official checkpoint), not `scratch_fallback`.
- [ ] Report the official **TEST** split for any adapted model (R2–R6); the mixed `r0_external_anchor_5000` is for the R0 zero-shot anchor only. Never substitute demo MAE.
- [ ] Calibration / conformal fitted on **VAL only**.
- [ ] Published EchoCLIP external EF MAE ≈ 7.1% is a **literature reproduction target**, not a local result until you finish `--paper` runs.
- [ ] Any fact still marked **UNVERIFIED** above is re-checked against a primary source before it appears in the manuscript.

For locked experiment IDs (R0–R6), see [PAPER.md](../PAPER.md).
