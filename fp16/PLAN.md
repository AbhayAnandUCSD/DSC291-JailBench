# PLAN

## Overview

Reproduce JailbreakBench (Chao et al., NeurIPS 2024) core results and extend to study
deployment-time modification effects on safety alignment.

## Pipeline Parts

1. **Part 1 — Table 1 (Judge Selection)** — `01_judges.ipynb`, `01b_guard2.ipynb`
2. **Part 2 — Baseline FP16** — `02_fp16_baseline.ipynb`
3. *(Parts 3, 4 SKIP — Quantization & Pruning in sibling notebooks)*
4. **Part 4b — Perplexity** — `04b_perplexity.ipynb`
5. **Part 5 — Table 2 (Attack Evaluation)** — `05_attacks.ipynb`
6. **Part 6 — Table 3 (Defense Evaluation)** — `06_defenses.ipynb`, `06a_defenses_gen.ipynb`, `06b_defenses_guard1.ipynb`, `06c_defenses_guard3.ipynb`
7. **Part 7 — Figure 2 (Benign Refusal)** — `07_benign_refusal.ipynb`
8. **Part 8 — Decoding Sweep** — `08_decoding.ipynb`
9. **Part 9 — Cross-Model** — `09_cross_model.ipynb`
10. **Part 10 — Summary** — `10_summary.ipynb`
11. **Part 11 — Appendix Guard-3** — `11_appendix_guard3.ipynb`

## Deliverables

- `jailbreakbench.ipynb` — monolithic reproduction notebook
- `jailbreakbench_guard.ipynb` — decoupled defense-evaluation pipeline using Guard-3 as judge

## Recent Changes (2026-06-08)

- Created `requirements.txt` covering all packages imported across every notebook in the folder.
- Updated `jailbreakbench_guard.ipynb` and `jailbreakbench.ipynb` first cells to install via `-r requirements.txt`.
- Key pins: `vllm==0.15.0`, `jailbreakbench>=1.0.0`, `litellm<1.50`, `setuptools<70.0.0`.
- Install uses `--ignore-requires-python` to work around jailbreakbench's `<3.12` constraint on Python 3.13.

## Known Issues

- `bitsandbytes` may not support Windows (cluster is Linux — OK).
- vLLM 0.15.0 must be built from source on some CUDA versions; use the pre-built wheel where available.

## Next Steps

- Verify the notebooks load correctly on the cluster with the new `requirements.txt`.
- Run Parts 1–11 in order per `jbb_common.py` docstring with kernel restarts between each.
- Once pipeline is verified, create quantized model variant notebook (step 11 BONUS).
