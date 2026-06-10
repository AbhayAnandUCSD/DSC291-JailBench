# AGENTS.md

## Overview

Reproduce JailbreakBench (Chao et al., NeurIPS 2024). The paper's contributions are: (1) the JBB-Behaviors dataset of 100 misuse behaviors (and 100 matched benign counterparts) aligned with OpenAI's usage policies; (2) an evolving repository of jailbreak artifacts for PAIR, GCG, JailbreakChat, and prompt-with-random-search; (3) a standardized red-teaming and defense-evaluation pipeline with a clearly specified threat model, system prompts, and chat templates; (4) a rigorous judge-selection study comparing six classifiers against expert human annotation, ultimately recommending Llama-3-70B as judge; and (5) a public leaderboard tracking attack/defense performance across Vicuna-13B, Llama-2-7B, GPT-3.5, and GPT-4.

Beyond reproducing the benchmarks, we want to investigate whether practical post-training modifications routinely applied during deployment i.e., quantization, pruning, parameter-efficient fine-tuning, and decoding adjustments, silently degrade the safety alignment scores from JailbreakBench.

## Implementation

> The following section contains ALL implemention ideas to be implemented. Reference the paper listed in ## References to align yourself with the correct tables.

> `jailbreakbench.py` has been created for easier LLM parsing of the notebook `jailbreakbench.ipynb`, however be warned that the .py file may not be up to date. Proceed with caution. The is not the main deliverable, the notebook is.

### Part 1: Reproduction

We will reproduce the core experimental results of JailbreakBench on the two open-weight models supported by the official pipeline:

- Table 2 (Attack evaluation): Attack success rate, average queries, and average tokens for PAIR, GCG, JailbreakChat (AIM template), and Prompt-with-RS on Llama-2-7B-chat. We will use the released artifacts with the prompt from Table 10 of the paper.
- Table 3 (Defense evaluation): Transfer-attack ASR against the same attacks on three defenses: SmoothLLM (N=10, q=10% swap), Perplexity Filter (Llama-2-7B perplexity), and Erase-and-Check (erase length 20).
- Figure 2 (Refusal rates): Refusal rates on the 100 benign JBB-Behaviors with and without each defense, using Llama-3-8B as a refusal judge.
- Table 1 (Judge selection, partial): We will reproduce agreement / FPR / FNR for Rule-based, and Llama Guard 2 on the 300-prompt judge dataset, omitting GPT-4 due to API cost.

We will skip the closed-source columns (GPT-3.5, GPT-4) of Tables 2 and 3 unless API budget permits a small-scale spot check, since the paper itself notes that those numbers drift over time as providers patch attacks.

### Part 2: New Experiments — Robustness Under Model Modifications

Our novel contribution is a systematic study of how common deployment-time modifications affect jailbreak susceptibility. We will study five axes of modification, holding the JailbreakBench evaluation pipeline fixed:

- Quantization. Rule-base LLM (baseline), Llama 2 7B and 13B, Qwen 2.5 quantized using llama.cpp or anything that works on DSMLP. Hypothesis: aggressive low-bit quantization perturbs the activation patterns underlying refusal behavior, increasing ASR.
- Pruning. Unstructured magnitude pruning at 25%, 50%, and 75% sparsity using Wanda or SparseGPT. Hypothesis: refusal directions are concentrated in a small set of weights and are disproportionately damaged at high sparsity.
- Decoding parameters. Temperature in {0.0, 0.7, 1.0} and top-p in {0.9, 1.0}, deviating from JailbreakBench's deterministic-greedy default. Hypothesis: higher-entropy decoding makes attacks more effective at the cost of output quality.
- Model scale / family. Cross-model comparison across Llama-2-7B, and (compute permitting) Llama-3-8B-Instruct.

For each (model, modification, intensity) cell, we will run all four attack artifacts from the JailbreakBench repository and report ASR using a Rule-based LLM, Qwen-2.5, Llama-2-7B and 13B. The deliverables are (i) a heatmap of ASR over the (modification axis, intensity) grid, (ii) a comparison table contrasting modified-model ASR against the Rule-base LLM baseline, and (iii) a refusal-rate sanity check on benign behaviors to ensure that observed ASR increases reflect genuine alignment degradation rather than across-the-board over-compliance.

## New Proposed Notebook Pipieline (For Your Reference)

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

## Contraints

- Use only the notebook `jailbreakbench.ipynb` and `jailbreakbench_guard.ipynb`
- Before starting the build, read `PLAN.md`; do not start until it has been read
- After completing the build, append 'PLAN.md` with your next steps, the session is not finished until `PLAN.md` has been updated
- The session is not finished until the user declares that it is
- `vllm` is the quanitzation library, do not use `auto-gptq`
- Because this is a dev build, do not run the notebook; the notebook will be run on a cluster where LLM access is forbidden
- Human feedback will be provided in the debugging session
- More constraints will be added as needed

## Deliverables

- (1) `jailbreakbench.ipynb`

## Repositories

- https://github.com/JailbreakBench/jailbreakbench

## References

- [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://arxiv.org/pdf/2404.01318)
- [WANDA. A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/html/2306.11695v3)