# DSC291 JailBench

Investigating how quantization affects LLM safety alignment using [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench).

## Overview

This project measures refusal rates of Llama-2-7b-chat across different quantization levels:

- **FP16** — full model weights from HuggingFace
- **INT8** — GGUF via vLLM
- **INT4** — AWQ via vLLM

A note on quantization: BitsAndBytes is a stable quantization build and can be found in select notebooks.

## Requirements

Tested on 50 GiB storage, Nvidia A6000 or Nvidia H100.

Notebooks are split due to 50 GiB storage limit, one could theoretically combine al notebooks into one with 100+ GiB.

## Setup

Requires a HuggingFace token with access to `meta-llama/Llama-2-7b-chat-hf`. [LINK](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

Run notebooks in the following order:

1. 00_quantize.ipynb
2. 01_judges.ipynb
3. 01b_guard2.ipynb
4. 02_fp16_model.ipynb # or whatever quantized model you choose to run
5. 04b_perplexity.ipynb
6. 05_attacks.ipynb
7. 06_defenses.ipynb
8. 06a_defenses_gen.ipynb
9. 06b_defenses_guard1.ipynb
10. 06c_defenses_guard3.ipynb
11. 07_benign_refusal.ipynb
12. 08_decoding.ipynb
13. 09_cross_model.ipynb
14. 11_appendix_guard3.ipynb
15. 10_summary.ipynb
