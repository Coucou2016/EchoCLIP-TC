# Obtain status (local machine) / 本机获取状态

**Research use only.** EchoNet videos are not redistributed by this repo.

Copy this template locally after you download weights or data. Do **not** commit machine-specific absolute paths.

## EchoCLIP official weights

| Item | Value |
|------|--------|
| Hub id | `hf-hub:mkaichristensen/echo-clip` |
| Local dir | `models/echo-clip/` (gitignored) |
| Main weight | `open_clip_pytorch_model.bin` (~606 MB) |
| Status | _fill in: OK / blocked / mirror used_ |
| Notes | Optional: `HF_ENDPOINT=https://hf-mirror.com` |

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # if needed
python scripts\obtain_echo_clip_weights.py
```

## EchoNet-Dynamic

| Check | Result |
|-------|--------|
| AIMI registration / signed URL | _fill in_ |
| Local zip or `ECHONET_ROOT` | _fill in_ |

```powershell
python scripts\obtain_echonet_zip.py --zip <path-to-EchoNet-Dynamic.zip>
$env:ECHONET_ROOT = "<path-to-EchoNet-Dynamic>"
python scripts\build_echonet_manifest.py --echonet-root $env:ECHONET_ROOT --subset-5000
```

Do **not** commit clinical videos, zips, or `models/` weights.
