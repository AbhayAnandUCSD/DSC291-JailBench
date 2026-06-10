# INT8 variant of the JailbreakBench pipeline

This folder is a clone of `tml/` (the FP16 reference) with the Llama-2-7B-chat
target swapped for a **GPTQ INT8** checkpoint produced by
[GPTQModel](https://github.com/ModelCloud/GPTQModel) — the quantization library
recommended by [vLLM's GPTQModel docs](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/).
Everything else (Guard-1, Guard-2, Guard-3, Llama-3-8B-Instruct, the JBB
dataset, attack artifacts, defenses) is unchanged.

## Why GPTQModel (not bitsandbytes)

bitsandbytes pre-quantized checkpoints don't load reliably under vLLM —
especially at 8-bit. GPTQModel produces native GPTQ checkpoints that vLLM
auto-detects from `config.json` and runs through the Marlin / Machete kernels
on Ampere+ GPUs (A100, A10, H100). No `quantization=` kwarg, no
`load_format=` kwarg — just point vLLM at the local path.

## How the swap works

1. `00_quantize.ipynb` — runs **once** per cluster session. Calls
   `pip install -U gptqmodel --no-build-isolation`, loads
   `meta-llama/Llama-2-7b-chat-hf`, calibrates on 1024 samples from
   `allenai/c4`, and saves a `QuantizeConfig(bits=8, group_size=128)`
   checkpoint to `./quantized_int8/`.
2. `jbb_common.patch_jbb()` — extended with a redirect layer. Any
   `vllm.LLM(model="meta-llama/Llama-2-7b-chat-hf", ...)` call (including the
   ones JBB's `LLMvLLM` makes internally) is rewritten to
   `vllm.LLM(model="./quantized_int8", ...)`. vLLM auto-detects the GPTQ
   format from `config.json`. Guard models and Llama-3-8B-Instruct are not
   touched. If the checkpoint is missing the redirect raises a clear error
   pointing at `00_quantize.ipynb`.
3. `04b_perplexity.ipynb` — uses `GPTQModel.load()` (instead of
   `AutoModelForCausalLM.from_pretrained`) so the PPL sanity check exercises
   the GPTQ checkpoint with the same kernels vLLM uses.

## Run order (same as `tml/`, with `00` first)

```
00_quantize → 01 → 01b → 02 → 04b → 05 → 06 → 07 → 08 → 09 → 11 → 10
```

Restart the kernel between notebooks (same vLLM child-worker leak rule as the
FP16 reference). `00_quantize` is idempotent — re-running with the checkpoint
already on disk is a no-op.

## Cosmetic mismatch — be aware

Many cells still call the variable `llm_fp16` or label rows
`"Llama-2-7B-chat (FP16)"` in the saved `summary.csv`. The **code** loads INT8
GPTQ correctly (verified by the redirect's `print` line at runtime), but the
**labels** are inherited from the FP16 reference and weren't renamed to keep
the diff minimal. Treat any "FP16" label in this folder as "INT8 GPTQ."
