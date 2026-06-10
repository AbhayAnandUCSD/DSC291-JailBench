# JailbreakBench Reproduction — Comprehensive Summary

## A. Table 1 — Judge Selection

| Classifier                    |   Agreement |    FPR |    FNR |
|:------------------------------|------------:|-------:|-------:|
| Rule-based (StringClassifier) |      0.5467 | 0.6895 | 0.0455 |
| Llama Guard 3                 |      0.89   | 0.1053 | 0.1182 |

## B. FP16 Baseline Refusal Rate (Part 2)

| Model                  |   N |   Refusal Rate |
|:-----------------------|----:|---------------:|
| Llama-2-7B-chat (FP16) |  25 |           0.72 |

## C. Perplexity Sanity Check (Part 4b)

| Model Variant   |   Perplexity (PPL) |
|:----------------|-------------------:|
| FP16 (baseline) |              20.94 |

## D. Table 2 — Attack ASR (Part 5, Llama-Guard-3-8B judge)

| Method                    |   ASR |   Total Prompts Tested |
|:--------------------------|------:|-----------------------:|
| PAIR                      |  1    |                      4 |
| GCG                       |  0.58 |                    100 |
| JBC                       |  0.26 |                    100 |
| prompt_with_random_search |  0.99 |                    100 |

## E. Table 3 — Defense ASR (Part 6)

|    | Unnamed: 0       |   PAIR |   GCG |   JBC |   PRS |
|---:|:-----------------|-------:|------:|------:|------:|
|  0 | nan              |   0.75 |  0.07 |     0 |  0.71 |
|  1 | SmoothLLM        |   0.75 |  0.02 |     0 |  0    |
|  2 | PerplexityFilter |   0.75 |  0    |     0 |  0.71 |
|  3 | EraseAndCheck    |   0    |  0.02 |     0 |  0.26 |

## F. Figure 2 — Benign Refusal (Part 7)

| Defense           |   Refusal Rate |
|:------------------|---------------:|
| None (undefended) |           0.75 |
| SmoothLLM         |           0.86 |
| PerplexityFilter  |           0.75 |
| EraseAndCheck     |           0.75 |

## G. Decoding Sweep (Part 8)

|   Temperature |   Top-p |   PAIR |   GCG |   JBC |   PRS |   Mean ASR |
|--------------:|--------:|-------:|------:|------:|------:|-----------:|
|           0   |     0.9 |   0.75 |  0.07 |     0 |  0.73 |      0.388 |
|           0   |     1   |   0.75 |  0.07 |     0 |  0.74 |      0.39  |
|           0.7 |     0.9 |   0.75 |  0.05 |     0 |  0.61 |      0.352 |
|           0.7 |     1   |   0.5  |  0.08 |     0 |  0.44 |      0.255 |
|           1   |     0.9 |   0.5  |  0.11 |     0 |  0.49 |      0.275 |
|           1   |     1   |   0.5  |  0.09 |     0 |  0.52 |      0.278 |

## H. Cross-Model (Part 9)

| Model               |   Refusal Rate (Harmful, kw) |   ASR (PAIR, Guard-3) |
|:--------------------|-----------------------------:|----------------------:|
| Llama-2-7B-chat     |                            1 |                  0.75 |
| Llama-3-8B-Instruct |                            1 |                  0    |
