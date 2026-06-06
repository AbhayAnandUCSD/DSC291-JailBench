## MAXIMUM LEVEL PRIORITY TICKET

There is a notebook that uses Llama Guard models as the kailbreak judges, to replicate JailBreakbench (2024) as discussed in AGENTS.md. As of right now, jailbreakbench.ipynb does not run the following parts

 - 4a. Pruning with WANDA
 - 4b. Perplexity calculations (which is based on comparing the perplexity of fp16, int8, int4, and pruning)
 - 6. Defense Evaluations
 - 7. Refusal Rates on Benign Behaviors
 - 8. Decoding Parameter Variation
 - 9. Model Scale Family Comparison
 - 10. Summary (Here is where the Llama Judge 2 lies)

Read jailbreakbench.ipynb for the cell outputs, many errors are usually caused by OOM (or variants of it, see known bugs and buxfixes below). As a remedy to the replication and proposal of new experiments planned for in AGENTS.md, we prpose the following modifications to the pipeline:

  - 1. Environment Setup, Table 1 Judge Selection (the library dependecies are compatible, but for the purposes of replication, the versions of all libraries used here, and put it in a separate requirements.txt that is then called in the very first cell with %%capture)
  - 2. Load the Llama-2-7b-chat-hf FP16 model
  - 3. SKIP Quantization
  - 4. SKIP Pruning
  - 4b. Perplexity (yes, get the perplexity calculations)
  - 5. Table 2 Attack Evaluation (this section should run, see notebook for cell outputs currently there)
  - 6. Table 3 Defense Evaluation (this section was not able to run due to the vllm bug below, but proposed fix below does not work. This is highly likely to be an OOM error, but we hypothesize this will be nore more if we leave out the quantized models)
  - 7. Figure 2 Benign Refusal (again, this will not run foe the same reasons as (6))
  - 8. Decoding Sweep (is dependent on all other sections and therefore cannot be run yet until you complete this new pipeline)
  - 9. Cross Model (the judge did not work becuase it required gated access, but the notebook jailbreakbench_guard.ipynb has the correct guard code. Read this and implement it in this new pipeline discussed here.)
  - 10. Summary (keep what is important)
  - 11. BONUS: Once all goes well, a copy of jailbreakbench.ipynb will be made except the new copy will load a quantized Llama-2-7b-chat-hf. Do not implement this step (11) yet. This is only for your guidance to see the scope of the project.

The notebooks have already been uploaded to git, therefore all error codes have been preserved in its current state. It is your job to read and edit jailbreakbench.ipynb with the pipeline above and information from jailbreakbench_guard.ipynb.

## Data
- Model refusals are the same with INT8, INT4 quantization experiments. This does not mean the refusal rates are the same with pruning.
  - Add pruning (and implement ALL teammate TODOs) in the notebook so that it flows in a logical order; separate part 1 (replication of JBB paper) and part 2 (new quantization experiments)
  - Llama-Guard-3 as the judge
  - The paper uses Llama-3-70B.
- Everything matches, except for the PAIR column which is a judge problem. The Guard models are very strict about what is safe and unsafe and seems to flag too many prompts as unsafe on the PAIR attacks. It basically sees a faint hint of "sus" and immediately flags it as unsafe. PAIR makes these "faint hints" so I think the Guard model here just got baited. Llama-3-70B doesn't have this I think because it's not really trained specifically to work against adversarial attacks. I could be wrong.
- Consider attacking with gradient free methods. Gradient-free methods do not require gradient access. Methods include TAP (Tree of Attacks with Pruning, https://arxiv.org/html/2312.02119v3) which builds upon PAIR (Prompt Automatic Iterative Refinement, https://arxiv.org/html/2310.08419v4).
- Because the models are robust to attacks even with quantized, add perplexity (PPL) calculations.
  - PPL quantifies the model's uncertainty
  - Lower perplexity indicates better performance in predicting the next word in a sequence
  - High perplexity suggesting the model struggles to make accurate predictions.

## COMPLETED TICKETS

- "Part 0: Environment Setup" runs with 0 issues. Complete.
- "Part 1: Table 1 - Judge Selection" runs with 0 issues. Complete.
- "Part 2: Baseline FP16 Evaluation" — **FIXED**
  - **Root cause**: `token=HF_TOKEN` was added as a kwarg to `LLM()` constructors, but vLLM's `LLM()` does not accept a `token` parameter. This caused `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'token'`.
  - **Fix**: Removed `token=` from all 9 `LLM()` calls. HuggingFace auth is handled by `login(token=HF_TOKEN)` in Cell 4, which is sufficient for vLLM to download gated models.
  - **What's preserved**: `enforce_eager=True` on all 10 `LLM()` constructors (needed for T4/V100/A10 GPU compatibility).
- **2026-06-03 build — pipeline restructure** (edits made on the Windows dev box; not yet run on cluster):
  - **Deleted Part 3 (Quantization)** — 4 cells removed (`part3-title`, `sec3-int8`, `sec3-int4`, `sec3-summary`). Per pipeline step 3 (SKIP Quantization).
  - **Deleted Part 4 (Pruning / Wanda)** — 10 cells removed (`part4-title` through `sec4-summary`). Per pipeline step 4 (SKIP Pruning). The previous TypeError on `sec4-wanda-prune` is moot — section no longer present.
  - **Simplified Part 4b (Perplexity)** to FP16-only. Deleted `sec4b-ppl-pruned-levels` (Wanda 25/50/75 PPL loop) and rewrote the title markdown to scope PPL as a sanity check + PerplexityFilter threshold input.
  - **Part 5 (Table 2)** left untouched (PLAN.md noted "this section should run"). Still uses StringClassifier judge — see "next steps" below.
  - **Replaced Part 6 (Table 3 Defense)** entirely with the all-vLLM decoupled pipeline ported from `jailbreakbench_guard.ipynb`. Cells `sec6-defense-helpers` / `sec6-load-for-defense` / `sec6-evaluate-defenses` / `sec6-results` now implement: target gen (baseline + SmoothLLM-10 + PPL filter via `prompt_logprobs` + EaC-21 variants) → free target → Guard-1 judging for SmoothLLM majority vote + EaC any-flag → free Guard-1 → Guard-3-8B final ASR. Drops the old bitsandbytes PPL model and the rule-based defense judge.
  - **Replaced Part 7 (Figure 2 Benign Refusal)** to apply the same defense apparatus to the 100 benign goals (`baseline_benign`, `smoothllm_benign`, `ppl_benign`, `erase_and_check_benign`). Refusal detection is keyword-based — Guard-3 measures *safety*, not refusal, so it's not a drop-in replacement (paper uses Llama-3-8B as a refusal judge).
  - **Replaced Part 8 (Decoding Sweep)** to bypass `LLMvLLM.query_llm`'s fixed `temperature=0.0` via `target_llm.model.generate(prompts, sp)` with a manual chat template. Judge swapped from StringClassifier to Guard-3.
  - **Replaced Part 9 (Cross-Model)** to use Guard-3 judge for PAIR ASR + keyword judge for refusal rate. Uses `tok.apply_chat_template` per-model so Llama-2 and Llama-3 each get their own chat format; both share JBB's safety system prompt for fairness.
  - **Updated Part 10 (Summary)** to drop INT8/INT4/Wanda rows; added Table 1 to the aggregate (was missing before).
  - ~~**Moved the trailing `sec1-llamaguard2` cell** into Part 1 so Section 1 is self-contained.~~ **REVERTED 2026-06-03 same session.** Moving Guard-3's `LLM(...)` construction adjacent to Part 2's FP16 load triggered the documented "Engine Core initialization failed" race (Guard-3 ~16 GB + FP16 ~12 GB > 24 GB single-GPU budget, and vLLM child workers don't release memory promptly after `del`). The original end-of-notebook placement was load-order-aware, not an artifact. Llama-Guard-3 now lives in an **Appendix** section at the very end of the notebook (cell ids `1d713f99` markdown + `d5ecc3ae` code) and appends the second row of Table 1 after every other Part has freed its target / Guard models. The env-var fix (`FLASHINFER_DISABLE_VERSION_CHECK=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`) is still set globally in `sec0-imports` and re-asserted at the top of the appendix cell — load-order plus env-vars together, not env-vars alone.
  - **Updated header TOC** to reflect the new pipeline and call out which judges are used where.
  - **Files written by Part 6 / 7 / 8 / 9 on the cluster** will land in `results_part6/`, `results_part7/`, `results_part8/`, `results_part9/` (created with `os.makedirs(..., exist_ok=True)`) — JSON dumps of generations + judge labels, plus CSV exports of the final tables.

## NEXT STEPS (build session NOT yet declared finished by the user)

1. **Run the notebook end-to-end on the cluster.** Validation order matters — at least Parts 1, 2, 4b, 6, 7 should be smoke-tested in order before tackling 8 and 9 (which iterate over more configs).
2. **Watch for vLLM worker-leak after `del`** in Parts 1 / 6 / 7 / 8 / 9. Each section deletes its model and calls `gc.collect()` + `torch.cuda.empty_cache()`, but vLLM's child workers occasionally hold GPU memory until the kernel restarts. If a later cell OOMs on load, restart the kernel and re-run from that part. The `FLASHINFER_DISABLE_VERSION_CHECK` + `VLLM_WORKER_MULTIPROC_METHOD=spawn` env-vars from the known-bug fix are *not* set in the new Part 6 / 7 / 8 / 9 cells — add them at the top of the offending cell if vLLM init fails.
3. **Verify the reproducible numbers from the guard notebook.** Specifically `PRS / PerplexityFilter ≈ 0.72` (paper 0.73) and `GCG / SmoothLLM ≈ 0.01` (paper 0). If those drift, the port from `jailbreakbench_guard.ipynb` has a regression.
4. **Part 5 judge swap to Guard-3 (optional).** Currently Part 5 still uses StringClassifier so the original outputs remain comparable. To get a Guard-3 ASR row for Table 2, refactor Part 5 to the same generate→free→Guard-3→judge pattern used in Parts 6 / 8 / 9. Expect PAIR to inflate (same Guard-3 / Llama-3-70B taxonomy gap documented in Part 6).
5. **Part 7 refusal judge upgrade (optional).** Keyword-based refusal detection over-flags responses that *mention* harmful concepts in passing. Paper uses Llama-3-8B-Instruct as a refusal judge — load it after Guard-1 is freed (≈16 GB) and prompt it to classify each `(benign_goal, response)` as refusal / compliance.
6. **Bonus step 11 (quantized Llama-2-7B-chat copy).** Per AGENTS.md, do NOT implement yet; the present notebook is the FP16 deliverable. When ready, fork `jailbreakbench.ipynb` and replace the `LLMvLLM(model_name=…)` calls with the quantized path.

## KNOWN BUGS

- `Engine Core initialization failed. See root cause above.`
  > Solution: THis is a vllm issue. It is usually fixed by clearing CUDA cache with the following snippet, notice that the following code is placed before loading anew model in the notenook (verify yourself inside jailbreakbench.ipynb)
  ```py
  import os

  # 1. Bypass the FlashInfer version mismatch check (Direct Fix for your error)
  os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

  # 2. Keep the spawn method to ensure clean process initialization
  os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

  # Clean up memory from the previous crash
  import gc
  import torch
  gc.collect()
  torch.cuda.empty_cache()

  from vllm import LLM
  ```

## 2026-06-04 — PIPELINE SPLIT (spec, not yet implemented)

### Motivation

Running the full `jailbreakbench.ipynb` end-to-end on JupyterHub disconnects or hangs the compute unit. Root cause is cumulative vLLM child-worker memory leak across ~8–10 sequential model loads in one kernel session — `gc.collect()` + `torch.cuda.empty_cache()` after each `del` is insufficient. Splitting into per-Part notebooks lets the kernel be restarted between Parts, resetting the leak. Each Part runs in its own fresh kernel and persists its results to disk; a final merger reads the artifacts and reproduces the comprehensive summary.

### File layout (flat in `tml/`, numeric-prefix sorted)

```
tml/
├── jbb_common.py              ← NEW: shared setup + helpers (no model loaders)
├── 01_judges.ipynb            ← Part 1: rule-based (StringClassifier) row of Table 1
├── 01b_guard2.ipynb           ← Part 1b: Llama Guard 2 row of Table 1
├── 02_fp16_baseline.ipynb     ← Part 2: FP16 refusal-rate baseline
├── 04b_perplexity.ipynb       ← Part 4b: FP16 PPL sanity check (Llama-2-7B only)
├── 05_attacks.ipynb           ← Part 5: Table 2 attack ASR (Llama-Guard-3-8B judge — uniform with Parts 6/8/9)
├── 06_defenses.ipynb          ← Part 6: Table 3 SmoothLLM / PerplexityFilter / Erase-and-Check
├── 07_benign_refusal.ipynb    ← Part 7: Figure 2 refusal rates on 100 benign goals
├── 08_decoding.ipynb          ← Part 8: temperature × top-p sweep
├── 09_cross_model.ipynb       ← Part 9: Llama-2-7B-chat vs Llama-3-8B-Instruct
├── 10_summary.ipynb           ← Merger: pure-CPU, reads CSVs, joins Table 1 rows, writes summary.md
├── 11_appendix_guard3.ipynb   ← Llama-Guard-3-8B row of Table 1
├── results_part1/, results_part1b/, results_part2/ ... results_part11/   ← per-Part `summary.csv` + `raw.json`
├── final_results/summary.md              ← written by 10_summary.ipynb
├── jailbreakbench.ipynb         ← UNCHANGED (kept as reference + cached error outputs)
└── jailbreakbench_guard.ipynb   ← UNCHANGED (kept as reference for Guard-3 code)
```

### `jbb_common.py` contract

Exposes the following functions. **No model loaders** — each notebook keeps its `LLM(...)` / `AutoModelForCausalLM.from_pretrained(...)` call visible in-cell so the GPU-memory-sensitive kwargs (`gpu_memory_utilization`, `max_model_len`, `enforce_eager`) are auditable when debugging OOMs.

- `setup_vllm_env()` — sets `FLASHINFER_DISABLE_VERSION_CHECK=1` and `VLLM_WORKER_MULTIPROC_METHOD=spawn` (per KNOWN BUGS).
- `patch_jbb()` — applies the two monkey-patches: (a) `vllm.LLM.__init__` default kwargs override so `LLMvLLM` doesn't crash on `token=` etc., and (b) `jailbreakbench.defenses.defense_hparams.register_hparams` override so the PerplexityFilter points at the HF identifier `meta-llama/Llama-2-7b-chat-hf` instead of the researcher's hardcoded local path.
- `hf_login()` — reads `HF_TOKEN` from env, falls back to `getpass()` if missing, calls `huggingface_hub.login(token=...)`.
- `load_jbb_dataset()` — returns `(goals, benign_goals, judge_df)`. `goals` from `jbb.read_dataset().goals`; `benign_goals` from the `benign` split of `JailbreakBench/JBB-Behaviors`; `judge_df` from the `judge_comparison` test split (rows used by Parts 1 + 1b + 11).
- `judge_metrics(y_true, y_pred)` — returns `dict(agreement, fpr, fnr)`. Same logic as the original `sec1-helpers` cell.
- `refusal_keyword_check(text)` — boolean; same keyword set as Part 7's `REFUSAL_KEYWORDS`.
- `format_llama2_chat(user_prompt, system_prompt=None)` — applies Llama-2 chat template with JBB's default safety system prompt.
- `format_llama3_chat(user_prompt, system_prompt=None)` — Llama-3 chat template equivalent (used in Part 9 + 11).

### Per-notebook shape

Every notebook follows the same three-cell skeleton:

```python
# Cell 1 — setup
import sys; sys.path.insert(0, '.')
from jbb_common import (setup_vllm_env, patch_jbb, hf_login,
                        load_jbb_dataset, judge_metrics,
                        refusal_keyword_check, format_llama2_chat)
setup_vllm_env()
patch_jbb()
hf_login()
goals, benign_goals, judge_df = load_jbb_dataset()

# Cell 2 — Part-specific model load + experiment
# (visible LLM(...) / AutoModelForCausalLM.from_pretrained(...) calls)

# Cell 3 — persist results
os.makedirs(f"results_part{N}", exist_ok=True)
summary_df.to_csv(f"results_part{N}/summary.csv", index=False)
json.dump(raw_payload, open(f"results_part{N}/raw.json", "w"))
```

### Artifact format

Each Part writes to `tml/results_partN/`:
- `summary.csv` — the row(s) that Part contributes to the final tables. Columns vary per Part (e.g., Part 1: `Classifier, Agreement, FPR, FNR`; Part 6: `Attack, Defense, ASR`; Part 8: `Temperature, TopP, Attack, ASR`).
- `raw.json` — raw target generations + judge labels per prompt, sufficient to re-judge later without re-running the target. Shape is Part-specific.

Re-runs **unconditionally overwrite** existing files (`to_csv(...)`, `json.dump(...)`). Backups via git, not via file-name versioning.

### Merge step (`10_summary.ipynb`)

Pure-CPU notebook, no model loads. Reads every `results_partN/summary.csv`, joins Part 1 (rule-based) + Part 1b (Llama Guard 2) + Part 11 (Llama Guard 3) rows into a single Table 1, and renders all summary tables inline as pandas DataFrames. Writes `tml/final_results/summary.md` as the human-readable hand-in. **Tolerant of missing artifacts**: if `results_partN/summary.csv` doesn't exist, the cell prints `"Part N: not yet run"` and continues — so a partial run still produces a partial summary.

### Run order

Parts 01–09, 1b, and 11 are mutually independent (each reloads its own models, each loads the JBB dataset via `jbb_common`). They can be run in any order, including out-of-paper-order. **Run 10 last**. On the cluster, the recommended order is the natural numeric one: 01 → 01b → 02 → 04b → 05 → 06 → 07 → 08 → 09 → 11 → 10, restarting the kernel between each so the vLLM leak resets.

### Heavy-Part risk + fallback

Parts 6, 7, and 9 each still load **3 models sequentially in one kernel** (e.g., Part 6 = FP16 target → Guard-1 → Guard-3). In a fresh kernel this is hopefully fine, but `feedback_guard3_end_of_notebook.md` documents a vLLM engine-core race when Guard-3 is loaded adjacent to a recently-freed FP16 vLLM target on a 24 GB GPU.

If Part 6 / 7 / 9 still OOM or race on the cluster, the fallback is to split each into sub-notebooks:
- `06a_defenses_gen.ipynb` — FP16 target only; dumps baseline + SmoothLLM-perturbed + EaC-21 generations to `results_part6/gens.json`. Frees target.
- `06b_defenses_guard1.ipynb` — loads Guard-1-7B-4bit; reads `gens.json`; runs Guard-1 over SmoothLLM votes + EaC any-flag; dumps to `results_part6/guard1.json`. Frees Guard-1.
- `06c_defenses_guard3.ipynb` — loads Guard-3-8B; reads `gens.json`; runs final-ASR judging; writes `results_part6/summary.csv`.

Same template for 07 → `07a_gen` / `07b_guard1` and 09 → `09a_gen_llama2` / `09b_gen_llama3` / `09c_guard3`. **Do not pre-emptively split.** Ship 11 files first; only split if the cluster actually fails.

### 2026-06-04 amendment — Llama Guard 2 in Table 1; Part 5 uses Guard-3

- **Added `01b_guard2.ipynb`.** Llama Guard 2 row of Table 1, mirroring `11_appendix_guard3.ipynb` but with `meta-llama/Meta-Llama-Guard-2-8B`. Lives in its own kernel for the same load-order reason as Guard-3 (8 B model on a 24 GB GPU). `10_summary.ipynb` now joins three rows for Table 1: Part 1 + 1b + 11.
- **Rewrote `05_attacks.ipynb` to use Llama-Guard-3-8B as judge** (was `StringClassifier`). Now follows the same gen → free target → load Guard-3 → judge → free Guard-3 pattern as Parts 6 / 8 / 9. Uniform judge across the entire deliverable (Table 1 has three judges by design; every other table uses Guard-3). Expect PAIR ASR to inflate vs the old StringClassifier number — same Guard-3 / Llama-3-70B taxonomy gap documented in Part 6.
- **Run order** is now: 01 → 01b → 02 → 04b → 05 → 06 → 07 → 08 → 09 → 11 → 10.

### What the existing notebooks become

- `tml/jailbreakbench.ipynb` — UNCHANGED. Stays as a reference + cached error outputs are useful for debugging.
- `tml/jailbreakbench_guard.ipynb` — UNCHANGED. Stays as the source of the Guard-3 code that Part 6 / 7 / 8 / 9 / 11 ported.
- The `results_part6/` etc. directories mentioned in the 2026-06-03 NEXT STEPS are **re-homed under `tml/`** in this new design (`tml/results_part6/` instead of `./results_part6/` at notebook CWD). No path collision because the old notebook is no longer the runtime.

### Implementation order (when greenlit)

1. Write `tml/jbb_common.py` — exposes the functions above. Validate by importing it from a scratch notebook on the dev box (no GPU needed for the import).
2. Scaffold `01_judges.ipynb` and `11_appendix_guard3.ipynb` first — they're the lightest (judge dataset only, one model each) and exercise the `jbb_common` setup contract end-to-end.
3. Scaffold `02_fp16_baseline.ipynb` and `04b_perplexity.ipynb` — single-model Parts, low risk.
4. Scaffold `05_attacks.ipynb` — single-model Part (target only; StringClassifier judge is CPU).
5. Scaffold `06_defenses.ipynb`, `07_benign_refusal.ipynb` — heavy Parts; carry forward the all-vLLM decoupled pipeline already in the current `jailbreakbench.ipynb` Section 6 / 7.
6. Scaffold `08_decoding.ipynb`, `09_cross_model.ipynb`.
7. Scaffold `10_summary.ipynb` — implement the missing-file-tolerant aggregation.
8. Smoke-test the import-and-setup cell of every notebook on the dev box (no GPU work — just confirm `jbb_common` calls don't crash and the dataset loads).
9. Hand back to user for cluster run.
