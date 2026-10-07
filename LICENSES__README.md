# SPDX / license notes for EchoCLIP-TA

## License scope (what MIT covers, and what it does not)

**MIT covers** original scaffold code written for this repository (see the table
below and `PROVENANCE.md`). **MIT does not cover, and this repository does not
relicense:**

- Upstream EchoCLIP prompt strings (`echoclip/prompts.py`) and any
  upstream-aligned/derived logic (see `PROVENANCE.md`) — these remain under the
  upstream EchoCLIP **Academic Software License** (academic / non-profit use;
  derivatives allowed; further transfer of the software including derivative
  works is restricted; commercial use requires separate permission).
- The official EchoCLIP hub weights (`hf-hub:mkaichristensen/echo-clip`).
- EchoNet-Dynamic and other Stanford AIMI datasets.
- Third-party packages listed in the requirements files.

No verbatim upstream source files are vendored in this repository, and neither the
official EchoCLIP weights nor EchoNet-Dynamic data are redistributed here; users must
obtain them under their own upstream terms (see `docs/OBTAIN_DATA_AND_WEIGHTS.md`).

## File-level notes

| Path | License / status |
|------|------------------|
| `echoclip/temporal.py`, `echoclip/calibrate.py`, `echoclip/clinical.py`, `echoclip/protocol.py`, `echoclip/supervised.py`, `echoclip/loss.py`, `echoclip/cycle_sample.py`, `echoclip/efficiency.py`, `echoclip/cardiacclip.py`, `echoclip/prompts_ta.py` | MIT (this repo) |
| `echoclip/text.py` | `echoclip/text.py` is an independently authored report normalizer whose behavior intentionally overlaps upstream EchoCLIP report cleaning; no upstream code is vendored or line-copied in this repository, and whether the upstream echonet/echo_CLIP utils.py source was consulted is **unclear — needs author confirmation** (canonical wording; see `PROVENANCE.md`) |
| `echoclip/prompts.py` | Upstream-attributed EchoCLIP prompt templates (use under upstream EchoCLIP terms for parity; **not MIT**; not claimed as original) |
| `echoclip/official_parity.py`, `echoclip/preprocess.py` | Upstream-aligned logic — upstream terms apply; needs author decision (see `PROVENANCE.md`) |
| `scripts/*`, `tests/*`, `configs/*` | MIT (this repo) |
| Official EchoCLIP hub weights | Not redistributed; upstream / HF terms |
| EchoNet-Dynamic | Stanford AIMI non-commercial (not redistributed) |
| CardiacCLIP weights | Upstream Academic terms (not redistributed) |

See `LICENSE`, `NOTICE`, `ATTRIBUTION.md`, and `PROVENANCE.md`.
