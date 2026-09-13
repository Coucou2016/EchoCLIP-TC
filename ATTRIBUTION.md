# Attribution

## EchoCLIP (Nature Medicine 2024)

Architecture and clinical prompt templates are derived from the published EchoCLIP work and the official inference repository:

- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for echocardiogram interpretation.* Nature Medicine, 2024. https://doi.org/10.1038/s41591-024-02959-y
- Code: https://github.com/echonet/echo_CLIP

`echoclip/prompts.py` and report-cleaning regexes in `echoclip/text.py` follow patterns from echonet/echo_CLIP `utils.py` and `prompts_used.json`. EchoCLIP-TC structured captions (`echoclip/structured_text.py`) fill those same official templates from EchoNet measurements; they do not introduce new clinical wording.

EchoNet-Dynamic, if used, is licensed separately by Stanford AIMI (non-commercial research) and is not bundled in this repository.

## Provenance risk / license boundary (important)

**TODO — Academic Software License audit:** Official EchoCLIP (Cedars-Sinai / echonet) materials and Stanford AIMI datasets may be under **non-MIT / Academic Software / non-commercial** terms. The MIT `LICENSE` in this repository covers **original scaffold code written here** only. It does **not** automatically re-license:

- Upstream-derived prompt templates and cleaning regexes
- Official EchoCLIP weights (never redistributed here)
- EchoNet / CAMUS / related clinical datasets

Do not claim MIT covers those assets without a completed provenance audit and, where required, an Academic Software License notice. Prefer linking to upstream licenses rather than copying restricted text.

## CLIP / OpenCLIP

- Radford et al., CLIP (ICML 2021)
- Text tokenizer: OpenAI CLIP via Hugging Face `transformers`
- Optional weight init: OpenCLIP LAION checkpoints (`open-clip-torch`)

## This repository

Implementation code in `echoclip/` and `scripts/` authored for this project is MIT-licensed (see `LICENSE`), subject to the provenance caveat above. It is an independent training/inference scaffold—not the Cedars-Sinai production release and not a redistribution of official EchoCLIP weights.

Public code: https://github.com/Coucou2016/EchoCLIP-TC
