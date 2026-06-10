## IMPORTANT
- Read/write access given to `int8/` folder only; nothing else. Do not explore outside this folder.

## TODO 2026-08-2026

**BIG ERROR MESSAGE**
```
INFO 06-08 14:51:29 [model.py:541] Resolved architecture: LlamaForCausalLM
INFO 06-08 14:51:29 [utils.py:261] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'gpu_memory_utilization': 0.85, 'disable_log_stats': True, 'quantization': 'bitsandbytes', 'enforce_eager': True, 'model': './quantized_int8'}
INFO 06-08 14:51:29 [model.py:1561] Using max model len 4096

2026-06-08 14:51:33,756	INFO util.py:154 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.

INFO 06-08 14:51:33 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 06-08 14:51:33 [vllm.py:624] Asynchronous scheduling is enabled.
WARNING 06-08 14:51:33 [vllm.py:662] Enforce eager set, overriding optimization level to -O0
INFO 06-08 14:51:33 [vllm.py:762] Cudagraph is disabled under eager mode

You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[2], line 8
      4 from vllm import LLM, SamplingParams
      5 
      6 # Load the pre-quantized INT8 checkpoint directly.
      7 # The dtype below is the *computation* dtype (float16); weights are 8-bit.
----> 8 llm = LLM(
      9     model="./quantized_int8",
     10     quantization="bitsandbytes",
     11     dtype="float16",

File ~/llama-2-7b-int8/jbb_common.py:108, in patch_jbb.<locals>._patched_LLM_init(self, *args, **kwargs)
    103 elif _model in _target_ids and not _quant_dir.exists():
    104     raise RuntimeError(
    105         f"INT8 checkpoint not found at {_quant_dir}. "
    106         f"Run 00_quantize.ipynb first to produce it."
    107     )
--> 108 return _orig_LLM_init(self, *args, **kwargs)

File ~/.local/lib/python3.13/site-packages/vllm/entrypoints/llm.py:334, in LLM.__init__(self, model, runner, convert, tokenizer, tokenizer_mode, skip_tokenizer_init, trust_remote_code, allowed_local_media_path, allowed_media_domains, tensor_parallel_size, dtype, quantization, revision, tokenizer_revision, seed, gpu_memory_utilization, swap_space, cpu_offload_gb, enforce_eager, enable_return_routed_experts, disable_custom_all_reduce, hf_token, hf_overrides, mm_processor_kwargs, pooler_config, structured_outputs_config, profiler_config, attention_config, kv_cache_memory_bytes, compilation_config, logits_processors, **kwargs)
    297 engine_args = EngineArgs(
    298     model=model,
    299     runner=runner,
   (...)    329     **kwargs,
    330 )
    332 log_non_default_args(engine_args)
--> 334 self.llm_engine = LLMEngine.from_engine_args(
    335     engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
    336 )
    337 self.engine_class = type(self.llm_engine)
    339 self.request_counter = Counter()

File ~/.local/lib/python3.13/site-packages/vllm/v1/engine/llm_engine.py:172, in LLMEngine.from_engine_args(cls, engine_args, usage_context, stat_loggers, enable_multiprocessing)
    169     enable_multiprocessing = True
    171 # Create the LLMEngine.
--> 172 return cls(
    173     vllm_config=vllm_config,
    174     executor_class=executor_class,
    175     log_stats=not engine_args.disable_log_stats,
    176     usage_context=usage_context,
    177     stat_loggers=stat_loggers,
    178     multiprocess_mode=enable_multiprocessing,
    179 )

File ~/.local/lib/python3.13/site-packages/vllm/v1/engine/llm_engine.py:88, in LLMEngine.__init__(self, vllm_config, executor_class, log_stats, aggregate_engine_logging, usage_context, stat_loggers, mm_registry, use_cached_outputs, multiprocess_mode)
     85     self.dp_group = None
     86 self.should_execute_dummy_batch = False
---> 88 self.input_processor = InputProcessor(self.vllm_config)
     89 self.io_processor = get_io_processor(
     90     self.vllm_config,
     91     self.model_config.io_processor_plugin,
     92 )
     94 # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).

File ~/.local/lib/python3.13/site-packages/vllm/v1/engine/input_processor.py:72, in InputProcessor.__init__(self, vllm_config, mm_registry)
     69 self.mm_registry = mm_registry
     70 self.mm_processor_cache = mm_registry.processor_cache_from_config(vllm_config)
---> 72 self.input_preprocessor = InputPreprocessor(
     73     self.model_config,
     74     vllm_config.observability_config,
     75     mm_registry,
     76     mm_processor_cache=self.mm_processor_cache,
     77 )

File ~/.local/lib/python3.13/site-packages/vllm/inputs/preprocess.py:58, in InputPreprocessor.__init__(self, model_config, observability_config, mm_registry, mm_processor_cache)
     56 self.model_config = model_config
     57 self.observability_config = observability_config
---> 58 self.renderer = renderer_from_config(model_config)
     59 self.mm_registry = mm_registry
     60 self.mm_processor_cache = mm_processor_cache

File ~/.local/lib/python3.13/site-packages/vllm/renderers/registry.py:84, in renderer_from_config(config, **kwargs)
     81 else:
     82     renderer_mode = tokenizer_mode
---> 84 return RENDERER_REGISTRY.load_renderer(
     85     renderer_mode,
     86     config,
     87     tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
     88 )

File ~/.local/lib/python3.13/site-packages/vllm/renderers/registry.py:62, in RendererRegistry.load_renderer(self, renderer_mode, config, tokenizer_kwargs)
     55 def load_renderer(
     56     self,
     57     renderer_mode: str,
     58     config: "ModelConfig",
     59     tokenizer_kwargs: dict[str, Any],
     60 ) -> RendererLike:
     61     renderer_cls = self.load_renderer_cls(renderer_mode)
---> 62     return renderer_cls.from_config(config, tokenizer_kwargs)

File ~/.local/lib/python3.13/site-packages/vllm/renderers/hf.py:489, in HfRenderer.from_config(cls, config, tokenizer_kwargs)
    483 @classmethod
    484 def from_config(
    485     cls,
    486     config: ModelConfig,
    487     tokenizer_kwargs: dict[str, Any],
    488 ) -> "RendererLike":
--> 489     return cls(config, tokenizer_kwargs)

File ~/.local/lib/python3.13/site-packages/vllm/renderers/hf.py:505, in HfRenderer.__init__(self, config, tokenizer_kwargs)
    501     tokenizer = None
    502 else:
    503     tokenizer = cast(
    504         HfTokenizer,
--> 505         cached_get_tokenizer(
    506             tokenizer_cls=CachedHfTokenizer,  # type: ignore[type-abstract]
    507             **tokenizer_kwargs,
    508         ),
    509     )
    511 self._tokenizer = tokenizer

File ~/.local/lib/python3.13/site-packages/vllm/tokenizers/registry.py:214, in get_tokenizer(tokenizer_name, tokenizer_cls, trust_remote_code, revision, download_dir, *args, **kwargs)
    211 else:
    212     tokenizer_cls_ = tokenizer_cls
--> 214 tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)
    215 if not tokenizer.is_fast:
    216     logger.warning(
    217         "Using a slow tokenizer. This might cause a significant "
    218         "slowdown. Consider using a fast tokenizer instead."
    219     )

File ~/.local/lib/python3.13/site-packages/vllm/tokenizers/hf.py:79, in CachedHfTokenizer.from_pretrained(cls, path_or_repo_id, trust_remote_code, revision, download_dir, *args, **kwargs)
     68 @classmethod
     69 def from_pretrained(
     70     cls,
   (...)     76     **kwargs,
     77 ) -> HfTokenizer:
     78     try:
---> 79         tokenizer = AutoTokenizer.from_pretrained(
     80             path_or_repo_id,
     81             *args,
     82             trust_remote_code=trust_remote_code,
     83             revision=revision,
     84             cache_dir=download_dir,
     85             **kwargs,
     86         )
     87     except ValueError as e:
     88         # If the error pertains to the tokenizer class not existing or not
     89         # currently being imported,
     90         # suggest using the --trust-remote-code flag.
     91         if not trust_remote_code and (
     92             "does not exist or is not currently imported." in str(e)
     93             or "requires you to execute the tokenizer file" in str(e)
     94         ):

File ~/.local/lib/python3.13/site-packages/transformers/models/auto/tokenization_auto.py:1175, in AutoTokenizer.from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1172 tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]
   1174 if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
-> 1175     return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
   1176 else:
   1177     if tokenizer_class_py is not None:

File ~/.local/lib/python3.13/site-packages/transformers/tokenization_utils_base.py:2113, in PreTrainedTokenizerBase.from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2110     else:
   2111         logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
-> 2113 return cls._from_pretrained(
   2114     resolved_vocab_files,
   2115     pretrained_model_name_or_path,
   2116     init_configuration,
   2117     *init_inputs,
   2118     token=token,
   2119     cache_dir=cache_dir,
   2120     local_files_only=local_files_only,
   2121     _commit_hash=commit_hash,
   2122     _is_local=is_local,
   2123     trust_remote_code=trust_remote_code,
   2124     **kwargs,
   2125 )

File ~/.local/lib/python3.13/site-packages/transformers/tokenization_utils_base.py:2151, in PreTrainedTokenizerBase._from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2148 # If one passes a GGUF file path to `gguf_file` there is no need for this check as the tokenizer will be
   2149 # loaded directly from the GGUF file.
   2150 if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None and not gguf_file:
-> 2151     slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
   2152         copy.deepcopy(resolved_vocab_files),
   2153         pretrained_model_name_or_path,
   2154         copy.deepcopy(init_configuration),
   2155         *init_inputs,
   2156         token=token,
   2157         cache_dir=cache_dir,
   2158         local_files_only=local_files_only,
   2159         _commit_hash=_commit_hash,
   2160         **(copy.deepcopy(kwargs)),
   2161     )
   2162 else:
   2163     slow_tokenizer = None

File ~/.local/lib/python3.13/site-packages/transformers/tokenization_utils_base.py:2359, in PreTrainedTokenizerBase._from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2357 # Instantiate the tokenizer.
   2358 try:
-> 2359     tokenizer = cls(*init_inputs, **init_kwargs)
   2360 except import_protobuf_decode_error():
   2361     logger.info(
   2362         "Unable to load tokenizer model from SPM, loading from TikToken will be attempted instead."
   2363         "(Google protobuf error: Tried to load SPM model with non-SPM vocab file).",
   2364     )

File ~/.local/lib/python3.13/site-packages/transformers/models/llama/tokenization_llama.py:171, in LlamaTokenizer.__init__(self, vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token, clean_up_tokenization_spaces, use_default_system_prompt, spaces_between_special_tokens, legacy, add_prefix_space, **kwargs)
    169 self.add_eos_token = add_eos_token
    170 self.use_default_system_prompt = use_default_system_prompt
--> 171 self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
    172 self.add_prefix_space = add_prefix_space
    174 super().__init__(
    175     bos_token=bos_token,
    176     eos_token=eos_token,
   (...)    187     **kwargs,
    188 )

File ~/.local/lib/python3.13/site-packages/transformers/models/llama/tokenization_llama.py:198, in LlamaTokenizer.get_spm_processor(self, from_slow)
    196 tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
    197 if self.legacy or from_slow:  # no dependency on protobuf
--> 198     tokenizer.Load(self.vocab_file)
    199     return tokenizer
    201 with open(self.vocab_file, "rb") as f:

File ~/.local/lib/python3.13/site-packages/sentencepiece/__init__.py:961, in SentencePieceProcessor.Load(self, model_file, model_proto)
    959 if model_proto:
    960   return self.LoadFromSerializedProto(model_proto)
--> 961 return self.LoadFromFile(model_file)

File ~/.local/lib/python3.13/site-packages/sentencepiece/__init__.py:316, in SentencePieceProcessor.LoadFromFile(self, arg)
    315 def LoadFromFile(self, arg):
--> 316     return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)

TypeError: not a string
```

**TODO:**
- We already have a quantized model thanks to @00_quantize.py
- OUTPUT: "Already quantized at /home/USER/llama-2-7b-int8/quantized_int8 - skipping."
- The error code above happens when the second cell of @02_fp16_model.ipynb is run
- Therefore we need to change it so that
  - It loads the INT8 model
    ```py
    # Load the pre-quantized INT8 checkpoint directly.
    # The dtype below is the *computation* dtype (float16); weights are 8-bit.
    llm = LLM(
        model="./quantized_int8",
        quantization="bitsandbytes",
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    ```
- Make the error go away
- Rename @02_fp16_model.ipynb to @02_int8_model.ipynb
- Make all cascading notebooks from beyond @02_int8_model.ipynb load the quantized model as well
- There is no need to edit @00_quantize.ipynb @01_judges.ipynb @01b_guard2.ipynb, these seemingly run ok
- If bitsandbytes does not work (which it probabily isn't based on the error), try GGUF or AWQ: https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/#quantizing-a-model
- Make the notebooks such that we only need to log into huggingface only when necessary, not every single notebook needs huggingface if it is loading the quantized model that is already here (see OUTPUT bullet above)

## DONE 2026-06-08 (Claude session)

**Root cause**: the traceback was misleading. `quantization="bitsandbytes"` wasn't
the real problem — the failure is in `sentencepiece.LoadFromFile(self.vocab_file)`
with `TypeError: not a string`, meaning `vocab_file is None`. `GPTQModel.save()`
does NOT emit `tokenizer.model` into `./quantized_int8/`, so vLLM's slow-tokenizer
path blows up when it tries to load the SPM vocab from the quantized dir.
`bitsandbytes` is also the wrong knob — `00_quantize.ipynb` produces a **GPTQ**
checkpoint (per `VARIANT_README.md`), which vLLM auto-detects from `config.json`
with no `quantization=` kwarg at all.

**Changes**:

1. `jbb_common.py`
   - `_patched_LLM_init`: when redirecting `meta-llama/Llama-2-7b-chat-hf` →
     `./quantized_int8`, also inject `tokenizer="meta-llama/Llama-2-7b-chat-hf"`
     so vLLM pulls the SPM vocab from the HF repo (cached after `00_quantize`)
     instead of the quantized dir.
   - `hf_login(required=True)`: now no-ops when `huggingface_hub.whoami()`
     succeeds (i.e. token is already cached from a prior `login(...)` or
     `huggingface-cli login`). After `00_quantize.ipynb` logs in once, every
     downstream notebook silently skips the prompt. Pass `required=False` from
     notebooks that don't need any HF downloads to skip the prompt entirely
     even on first run.

2. `02_fp16_baseline.ipynb` → renamed to `02_int8_model.ipynb`.
   - Markdown header rewritten: "Baseline INT8 Evaluation" + explains the
     GPTQ auto-detect + tokenizer-from-HF design.
   - Load cell rewritten per PLAN.md's snippet (minus the wrong
     `quantization="bitsandbytes"`):
     ```py
     llm_fp16 = LLM(
         model="./quantized_int8",
         tokenizer="meta-llama/Llama-2-7b-chat-hf",
         dtype="float16",
         max_model_len=4096,
         gpu_memory_utilization=0.85,
         tensor_parallel_size=1,
         enforce_eager=True,
     )
     ```
   - Summary CSV label updated to `"Llama-2-7B-chat (INT8 GPTQ)"`; raw.json
     gets a `checkpoint` field. Variable name `llm_fp16` kept (per
     `VARIANT_README.md`'s minimal-diff stance).

3. Downstream notebooks (`05`, `06`, `06a`, `06b`, `06c`, `07`, `08`, `09`,
   `11`, `10`) were **not edited**. They either:
   - Call `LLM(model="meta-llama/Llama-2-7b-chat-hf", ...)` or
     `LLMvLLM(model_name="llama-2-7b-chat-hf")` — both caught by the existing
     `patch_jbb()` redirect, now with the tokenizer fix.
   - Load only guard models / Llama-3-8B-Instruct — untouched by the redirect.
   - `04b_perplexity.ipynb` already loads `./quantized_int8` directly via
     `GPTQModel.load(...)` and was already correct.

**Verification** (cannot run locally — code-only changes):
- After running `00_quantize` once in a fresh kernel, every other notebook in
  the run order `01 → 01b → 02 → 04b → 05 → 06 → 07 → 08 → 09 → 11 → 10`
  should now load INT8 via the redirect (target) or HF (guards / Llama-3)
  without the SPM tokenizer error and without re-prompting for an HF token.

## Next steps

- Run `02_int8_model.ipynb` on the cluster end-to-end to confirm the
  `TypeError: not a string` is gone and the INT8 refusal rate is in the
  same ballpark as the FP16 reference (`tml/02_fp16_baseline.ipynb`, 72%).
- If vLLM ever rejects the `tokenizer=` kwarg in a future release, the
  fallback is to copy the HF tokenizer files into `./quantized_int8/` once
  with `AutoTokenizer.from_pretrained(SRC_MODEL).save_pretrained(QUANT_PATH)`
  inside `00_quantize.ipynb` (but PLAN.md says don't touch that file unless
  necessary, so prefer the in-place `tokenizer=` kwarg).
- If the cluster doesn't have a cached HF token (fresh container), the first
  `hf_login(required=True)` call per session will prompt once and then every
  later notebook silently skips via the new `whoami()` short-circuit.

## DONE 2026-06-08 (round 2 — actual root cause)

The 02 notebook hit a second error after the tokenizer fix:
`RuntimeError: No model weights found in: ./quantized_int8`. Inspection of the
on-disk dir revealed it was **a stale half-write from an earlier bnb attempt**:

- Only `config.json` + `generation_config.json`, **zero weight files**, no tokenizer.
- `config.json` had `quant_method: "bitsandbytes"` and `load_in_8bit: true` — not
  what `GPTQModel.save()` writes (which would be `quant_method: "gptq"`).
- vLLM read the bnb metadata, picked the bnb loader, and failed on the missing weights.

The idempotency check in `00_quantize.ipynb` was the amplifier: it short-circuited
on `config.json` alone, so the stale bnb stub kept surviving every "re-run."

**Changes**:

1. `00_quantize.ipynb`
   - Idempotency check now requires `config.json` AND a `*.safetensors`/`*.bin`
     shard AND `tokenizer.model`. If any are missing, the dir is `rmtree`'d and
     quantization actually runs. Stops the bnb-stub-survives-forever bug.
   - After `model.save(...)`, `AutoTokenizer.from_pretrained(SRC_MODEL).save_pretrained(QUANT_PATH)`
     drops the SPM vocab + tokenizer config into the same dir. Lets downstream
     load entirely offline with `tokenizer="./quantized_int8"`.

2. `jbb_common.py`
   - `_patched_LLM_init` redirect now sets `tokenizer=str(_quant_dir)` (was
     `tokenizer=_model` / HF id). With the tokenizer files now living in the
     quantized dir, every downstream `LLM(model="meta-llama/Llama-2-7b-chat-hf")`
     and `LLMvLLM(model_name="llama-2-7b-chat-hf")` resolves fully from local —
     no HF call for the target model anywhere.

3. `02_int8_model.ipynb`
   - LLM cell now uses `tokenizer="./quantized_int8"` (was the HF id).
   - Setup cell now calls `hf_login(required=False)` — won't prompt if the
     token isn't cached. The JBB dataset load happens via `datasets` which
     handles its own caching; if a fresh prompt is genuinely needed for the
     gated dataset, `load_dataset` will raise a clear error and the user can
     `huggingface-cli login` once.

**User must do, in this order**:

```bash
rm -rf ~/llama-2-7b-int8/quantized_int8/   # delete the stale bnb stub
# Run 00_quantize.ipynb (fresh kernel)     # actually produces GPTQ weights + tokenizer
# Run 02_int8_model.ipynb (fresh kernel)   # loads everything from ./quantized_int8 offline
```

The expected `ls ~/llama-2-7b-int8/quantized_int8/` after `00_quantize` should
include: `config.json`, one or more `*.safetensors` shards (~7 GB total at INT8),
`tokenizer.model`, `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`,
`generation_config.json`.

## Next steps

- If `00_quantize` fails partway (OOM, OOD, etc.), the new strict idempotency
  check will catch the resulting incomplete dir on the next run and force a
  clean re-quantize. No manual cleanup needed except for genuine kernel kills.
- Downstream notebooks (`05`, `06*`, `07`, `08`, `09`) still call
  `LLM(model="meta-llama/Llama-2-7b-chat-hf", ...)` or
  `LLMvLLM(model_name="llama-2-7b-chat-hf")` — they all go through the
  `patch_jbb()` redirect, which now resolves model + tokenizer locally. They
  should "just work" once `00_quantize` produces a real checkpoint.
- Guards (`01b`, `06b`, `06c`, `11`) and Llama-3-8B (`09`) still hit HF — that's
  unavoidable, those repos aren't being quantized.