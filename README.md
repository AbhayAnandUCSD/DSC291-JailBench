# DSC291 JailBench

Investigating how quantization affects LLM safety alignment using [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench).

## Overview

This project measures refusal rates of Llama-2-7b-chat across different quantization levels:

- **FP16** — full precision baseline
- **INT8** — GPTQ 8-bit quantization
- **INT4** — BitsAndBytes NF4 and GPTQ 4-bit quantization

## Setup

```bash
pip install jailbreakbench transformers accelerate bitsandbytes auto-gptq tqdm pandas litellm
```

Requires a HuggingFace token with access to `meta-llama/Llama-2-7b-chat-hf`.

## Running

Run `jailbreakbench.ipynb` on a GPU instance (tested on UCSD DSMLP).

Note: FP16 requires ≥14GB VRAM. INT4 fits on 11GB GPUs (e.g., GTX 1080 Ti).
