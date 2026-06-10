# INT4 variant of the JailbreakBench pipeline

This folder is a clone of `tml/` (the FP16 reference) with the Llama-2-7B-chat
target swapped for an **INT4 (NF4) bitsandbytes**-quantized checkpoint.
Everything else (Guard-1, Guard-2, Guard-3, Llama-3-8B-Instruct, the JBB
dataset, attack artifacts, defenses) is unchanged.

## How the swap works

1. `00_quantize.ipynb` — runs **once** per cluster session. Loads
   `meta-llama/Llama-2-7b-chat-hf` with
   `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.float16)`, then saves the quantized checkpoint
   to `./quantized_int4/`.
2. `jbb_common.patch_jbb()` — extended with a redirect layer. Any
   `vllm.LLM(model="meta-llama/Llama-2-7b-chat-hf", ...)` call (including the
   ones JBB's `LLMvLLM` makes internally) is rewritten to
   `vllm.LLM(model="./quantized_int4", quantization="bitsandbytes",
   load_format="bitsandbytes", ...)`. Guard models and Llama-3-8B-Instruct
   are not affected. If the checkpoint is missing the redirect raises a clear
   error pointing at `00_quantize.ipynb`.
3. `04b_perplexity.ipynb` — uses `AutoModelForCausalLM.from_pretrained` (HF
   transformers, not vLLM), so the `patch_jbb` monkey-patch doesn't catch it.
   That notebook's model path was patched directly to `./quantized_int4`.

## Run order (same as `tml/`, with `00` first)

```
00_quantize → 01 → 01b → 02 → 04b → 05 → 06 → 07 → 08 → 09 → 11 → 10
```

Restart the kernel between notebooks (same vLLM child-worker leak rule as the
FP16 reference). `00_quantize` is idempotent — re-running with the checkpoint
already on disk skips the work.

## Cosmetic mismatch — be aware

Many cells still call the variable `llm_fp16` or label rows
`"Llama-2-7B-chat (FP16)"` in the saved `summary.csv`. The **code** loads INT4
correctly (verified by the redirect's `print` line at runtime), but the
**labels** are inherited from the FP16 reference and weren't renamed to keep
the diff minimal. Treat any "FP16" label in this folder as "INT4 bnb (NF4)."
