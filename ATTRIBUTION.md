# Attribution

## EchoCLIP (Nature Medicine 2024)

Architecture and clinical prompt templates are derived from the published EchoCLIP work and the official inference repository:

- Christensen, Vukadinovic, Yuan, Ouyang. *Vision–language foundation model for echocardiogram interpretation.* Nature Medicine, 2024. https://doi.org/10.1038/s41591-024-02959-y
- Code: https://github.com/echonet/echo_CLIP

`echoclip/prompts.py` and report-cleaning regexes in `echoclip/text.py` follow patterns from echonet/echo_CLIP `utils.py` and `prompts_used.json`. EchoCLIP-TC structured captions (`echoclip/structured_text.py`) fill those same official templates from EchoNet measurements; they do not introduce new clinical wording.

EchoNet-Dynamic, if used, is licensed separately by Stanford AIMI (non-commercial research) and is not bundled in this repository.

## CLIP / OpenCLIP

- Radford et al., CLIP (ICML 2021)
- Text tokenizer: OpenAI CLIP via Hugging Face `transformers`
- Optional weight init: OpenCLIP LAION checkpoints (`open-clip-torch`)

## This repository

Implementation code in `echoclip/` and `scripts/` is MIT-licensed (see `LICENSE`). It is an independent training/inference scaffold—not the Cedars-Sinai production release and not a redistribution of official EchoCLIP weights.
