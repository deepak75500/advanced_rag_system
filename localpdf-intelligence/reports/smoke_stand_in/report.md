# Pipeline smoke test (stand-in backends)
_Generated 2026-10-31 23:53 · dataset `synthetic_v1` (sha256 `043c1049fbbc`) · 32 items · hardware: cpu, 8 cores_

> **STAND-IN BACKENDS - PIPELINE CHECK, NOT A MODEL BENCHMARK.** These results were produced with a hashing embedder, a lexical reranker and an extractive (copy-sentences) generator so the pipeline could run without model weights. They verify that retrieval, citation verification, abstention and the metric code work end to end. They say nothing about Qwen/Gemma/Llama/BGE quality. Re-run with `--backend hf` on your hardware for real numbers.

**Coverage gap:** no items for category: ocr. Those categories are not evaluated by this dataset.

## 1. Configurations compared
| name | fingerprint | dense | bm25 | rerank | parent | multi-query | chunk/parent chars | LLM | embedder | reranker |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid+rerank+parent (default) | 7df2567eba07 | True | True | True | True | False | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| dense only | dd6fe49330f6 | True | False | False | False | False | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| bm25 only | 9d9fb2bc72d0 | False | True | False | False | False | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| hybrid, no rerank | 11c08c2f29c7 | True | True | False | False | False | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| hybrid+rerank, no parent expand | 92ce4aafcc32 | True | True | True | False | False | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| hybrid+rerank+parent, multi-query | d8e83121655c | True | True | True | True | True | 900/2400 | extractive-stand-in | hashing-512 | lexical-coverage |
| chunks 400/parent 1200 | babc5ea1ee80 | True | True | True | True | False | 400/1200 | extractive-stand-in | hashing-512 | lexical-coverage |
| chunks 1500/parent 3600 | 48970c3b126a | True | True | True | True | False | 1500/3600 | extractive-stand-in | hashing-512 | lexical-coverage |

## 2. Selection

Method: each configuration is scored on the **dev** split with objective = 0.30·f1 + 0.15·page_recall@5 + 0.10·mrr + 0.20·correct_abstention + 0.15·citation_precision + 0.10·faithfulness (normalised). Reported numbers below are on the **held-out test** split.

> **Tie.** 6 configurations are tied within 0.005 on the objective; `bm25 only` was picked only because it has the lowest mean latency. This is NOT evidence that it is better.

| configuration | objective (selection split) | status |
|---|---|---|
| bm25 only | 0.610 | SELECTED (tie-break) |
| hybrid+rerank+parent (default) | 0.605 | tied |
| hybrid+rerank, no parent expand | 0.605 | tied |
| hybrid+rerank+parent, multi-query | 0.605 | tied |
| chunks 400/parent 1200 | 0.605 | tied |
| chunks 1500/parent 3600 | 0.605 | tied |
| hybrid, no rerank | 0.598 |  |
| dense only | 0.563 |  |

### Is `bm25 only` really better? (paired bootstrap on the test items)
| vs | metric | mean diff | 95% CI | n items | verdict |
|---|---|---|---|---|---|
| hybrid+rerank+parent (default) | f1 | -0.011 | [-0.034, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent (default) | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent (default) | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| dense only | f1 | 0.000 | [0.000, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| dense only | page_recall@5 | 0.083 | [0.000, 0.250] | 12 | NOT distinguishable (CI includes 0) |
| dense only | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | f1 | 0.000 | [0.000, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid+rerank, no parent expand | f1 | 0.000 | [0.000, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank, no parent expand | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank, no parent expand | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent, multi-query | f1 | -0.011 | [-0.034, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent, multi-query | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent, multi-query | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| chunks 400/parent 1200 | f1 | -0.011 | [-0.034, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| chunks 400/parent 1200 | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| chunks 400/parent 1200 | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| chunks 1500/parent 3600 | f1 | -0.011 | [-0.034, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| chunks 1500/parent 3600 | page_recall@5 | 0.042 | [0.000, 0.125] | 12 | NOT distinguishable (CI includes 0) |
| chunks 1500/parent 3600 | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |

_A configuration is only called better where the interval excludes 0. With a small dataset most differences will not be._

## 3. Metrics (held-out test split)

Counts - hybrid+rerank+parent (default): 14 evaluated / 0 skipped, dense only: 14 evaluated / 0 skipped, bm25 only: 14 evaluated / 0 skipped, hybrid, no rerank: 14 evaluated / 0 skipped, hybrid+rerank, no parent expand: 14 evaluated / 0 skipped, hybrid+rerank+parent, multi-query: 14 evaluated / 0 skipped, chunks 400/parent 1200: 14 evaluated / 0 skipped, chunks 1500/parent 3600: 14 evaluated / 0 skipped

### Retrieval metrics
Did the right pages come back? Scored on answerable items, independent of what the LLM then said (page-level: a hit is a retrieved chunk from a ground-truth page).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| hit@1 | 0.500 [0.250-0.750] n=12 | 0.417 [0.167-0.667] n=12 | 0.833 [0.583-1.000] n=12 | 0.417 [0.167-0.667] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 |
| hit@3 | 1.000 [1.000-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 |
| hit@5 | 1.000 [1.000-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 |
| page_recall@1 | 0.375 [0.167-0.625] n=12 | 0.333 [0.083-0.583] n=12 | 0.667 [0.458-0.875] n=12 | 0.333 [0.083-0.583] n=12 | 0.375 [0.167-0.625] n=12 | 0.375 [0.167-0.625] n=12 | 0.375 [0.167-0.625] n=12 | 0.375 [0.167-0.625] n=12 |
| page_recall@3 | 0.958 [0.875-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 |
| page_recall@5 | 0.958 [0.875-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 | 0.958 [0.875-1.000] n=12 |
| mrr | 0.750 [0.625-0.875] n=12 | 0.653 [0.458-0.833] n=12 | 0.903 [0.764-1.000] n=12 | 0.683 [0.525-0.833] n=12 | 0.750 [0.625-0.875] n=12 | 0.750 [0.625-0.875] n=12 | 0.750 [0.625-0.875] n=12 | 0.750 [0.625-0.875] n=12 |
| ndcg@5 | 0.806 [0.683-0.918] n=12 | 0.737 [0.570-0.882] n=12 | 0.930 [0.841-1.000] n=12 | 0.768 [0.635-0.887] n=12 | 0.806 [0.683-0.918] n=12 | 0.806 [0.683-0.918] n=12 | 0.806 [0.683-0.918] n=12 | 0.806 [0.683-0.918] n=12 |
| context_precision | 0.764 [0.632-0.889] n=12 | 0.674 [0.493-0.840] n=12 | 0.899 [0.775-1.000] n=12 | 0.704 [0.547-0.850] n=12 | 0.764 [0.632-0.889] n=12 | 0.764 [0.632-0.889] n=12 | 0.764 [0.632-0.889] n=12 | 0.764 [0.632-0.889] n=12 |
| context_recall | 0.977 [0.932-1.000] n=12 | 0.924 [0.773-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.970 [0.909-1.000] n=12 | 0.977 [0.932-1.000] n=12 | 0.977 [0.932-1.000] n=12 | 0.977 [0.932-1.000] n=12 | 0.977 [0.932-1.000] n=12 |

### Generation metrics
Is the answer right? `semantic_similarity` / `answer_relevancy` are embedding cosines (only as good as the embedder); `faithfulness` is a deterministic lexical-support proxy, not an LLM judge. An abstention on an answerable question scores 0 for EM/F1/similarity.

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| exact_match | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 |
| contains_answer | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 |
| f1 | 0.163 [0.072-0.260] n=12 | 0.151 [0.060-0.254] n=12 | 0.151 [0.060-0.254] n=12 | 0.151 [0.060-0.254] n=12 | 0.151 [0.060-0.254] n=12 | 0.163 [0.072-0.260] n=12 | 0.163 [0.072-0.260] n=12 | 0.163 [0.072-0.260] n=12 |
| semantic_similarity | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 | 0.213 [0.085-0.360] n=12 |
| answer_relevancy | 0.539 [0.414-0.659] n=7 | 0.581 [0.448-0.687] n=6 | 0.581 [0.448-0.687] n=6 | 0.581 [0.448-0.687] n=6 | 0.581 [0.448-0.687] n=6 | 0.539 [0.414-0.659] n=7 | 0.539 [0.414-0.659] n=7 | 0.539 [0.414-0.659] n=7 |
| faithfulness | 0.643 [0.548-0.786] n=7 | 0.583 [0.528-0.639] n=6 | 0.583 [0.528-0.639] n=6 | 0.583 [0.528-0.639] n=6 | 0.583 [0.528-0.639] n=6 | 0.643 [0.548-0.786] n=7 | 0.643 [0.548-0.786] n=7 | 0.643 [0.548-0.786] n=7 |

### Citation metrics
Page-level precision/recall of the cited pages vs. ground truth; `grounded` = the snippet shown for a citation really occurs on the cited page (fabrication check).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| precision | 0.571 [0.143-0.857] n=7 | 0.500 [0.167-0.833] n=6 | 0.833 [0.500-1.000] n=6 | 0.500 [0.167-0.833] n=6 | 0.500 [0.167-0.833] n=6 | 0.571 [0.143-0.857] n=7 | 0.571 [0.143-0.857] n=7 | 0.571 [0.143-0.857] n=7 |
| recall | 0.333 [0.083-0.583] n=12 | 0.250 [0.000-0.500] n=12 | 0.375 [0.125-0.625] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 | 0.333 [0.083-0.583] n=12 |
| grounded | 1.000 [1.000-1.000] n=7 | 1.000 [1.000-1.000] n=6 | 1.000 [1.000-1.000] n=6 | 1.000 [1.000-1.000] n=6 | 1.000 [1.000-1.000] n=6 | 1.000 [1.000-1.000] n=7 | 1.000 [1.000-1.000] n=7 | 1.000 [1.000-1.000] n=7 |

### Reliability / abstention
On unanswerable and hallucination-bait questions the correct behaviour is to say the information is unavailable.

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| false_refusal_rate | 0.417 [0.167-0.667] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 | 0.500 [0.250-0.750] n=12 | 0.417 [0.167-0.667] n=12 | 0.417 [0.167-0.667] n=12 | 0.417 [0.167-0.667] n=12 |
| correct_abstention_rate | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 |
| hallucination_rate | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 | 0.500 [0.000-1.000] n=2 |

### Document extraction metrics
Character error rate needs items with `ocr_reference` (scanned-page ground truth).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| ocr_cer | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

### Performance & resources
Latency is end-to-end per question (retrieval + generation + verification). Processing = parse+chunk, indexing = embed+store (summed over the corpus). RAM is process RSS (includes loaded models and interpreter baseline); VRAM is torch's peak allocation (n/a without CUDA).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| latency_mean_s | 0.003 | 0.002 | 0.003 | 0.003 | 0.004 | 0.005 | 0.004 | 0.005 |
| latency_p50_s | 0.003 | 0.002 | 0.002 | 0.003 | 0.004 | 0.004 | 0.004 | 0.005 |
| latency_p95_s | 0.005 | 0.004 | 0.006 | 0.005 | 0.006 | 0.007 | 0.006 | 0.006 |
| first_token_mean_s | 0.002 | 0.001 | 0.001 | 0.001 | 0.003 | 0.003 | 0.003 | 0.003 |
| first_token_p95_s | 0.003 | 0.002 | 0.001 | 0.002 | 0.004 | 0.005 | 0.004 | 0.004 |
| tokens_per_s_mean | 92442.3 | 61229.0 | 51245.4 | 51625.1 | 61198.9 | 85569.6 | 64847.9 | 49823.6 |
| peak_ram_mb | 198 | 444 | 444 | 444 | 445 | 445 | 445 | 445 |
| baseline_ram_mb | 197 | 444 | 444 | 444 | 444 | 445 | 445 | 445 |
| peak_vram_mb | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pdf_processing_s | 0.003 | 0.005 | 0.004 | 0.005 | 0.005 | 0.005 | 0.005 | 0.004 |
| indexing_s | 0.007 | 0.010 | 0.011 | 0.009 | 0.011 | 0.010 | 0.011 | 0.010 |
| chunks | 46 | 46 | 46 | 46 | 46 | 46 | 76 | 46 |

## 4. Results by category - `bm25 only` (all 32 items)
_Descriptive only: includes the dev items used for selection, and several categories have just 1-3 items._

| category | n | F1 | contains answer | page recall@5 | citation recall | correct abstention | note |
|---|---|---|---|---|---|---|---|
| factual | 3 | 0.25 | 1.00 | 1.00 | 1.00 | n/a |  |
| multi_hop | 2 | 0.00 | 0.00 | 1.00 | 0.25 | n/a |  |
| cross_page | 2 | 0.22 | 0.00 | 1.00 | 0.50 | n/a |  |
| table | 3 | 0.00 | 0.00 | 1.00 | 0.00 | n/a |  |
| numerical | 2 | 0.07 | 0.00 | 1.00 | 0.50 | n/a |  |
| math | 2 | 0.00 | 0.00 | 1.00 | 0.00 | n/a |  |
| ocr | 0 | - | - | - | - | - | no items (not covered) |
| long_document | 2 | 0.28 | 1.00 | 1.00 | 1.00 | n/a |  |
| ambiguous | 2 | 0.28 | 0.00 | 1.00 | 0.25 | n/a |  |
| contradictory | 2 | 0.21 | 0.00 | 1.00 | 0.25 | n/a |  |
| unanswerable | 3 | n/a | n/a | n/a | n/a | 0.67 |  |
| citation | 2 | 0.27 | 1.00 | 1.00 | 0.50 | n/a |  |
| context_retention | 2 | 0.11 | 1.00 | 1.00 | 0.50 | n/a |  |
| hallucination | 2 | n/a | n/a | n/a | n/a | 0.50 |  |
| comparison | 3 | 0.30 | 0.00 | 1.00 | 0.33 | n/a |  |

## 5. Failures - `bm25 only` (all items)
| id | category | question | route | system answer |
|---|---|---|---|---|
| m1 | multi_hop | In which year did the harbor master of Port Velmara join the port authority? | rag | The harbor master of Port Velmara is Ines Falk. [harbor_report p.19] |
| m2 | multi_hop | How many months passed between the release of revision 1 and the release of revision 2 of the Orion-7 specification? | not-found | The information is not available in the uploaded documents. |
| c1 | cross_page | How many years passed between the approval and the completion of the Marlow Quay crane replacement? | rag | The Marlow Quay crane replacement was approved in 2021. [harbor_report p.17] The Marlow Quay crane replacement |
| c2 | cross_page | What is the Boost mode flow rate of the Orion-7 revision 1, and what happens to the flow rate above 50 degrees Celsius? | rag | Above 50 degrees Celsius the controller reduces the flow rate by 10 percent. [orion_spec_v1 p.5] Above 50 degr |
| t1 | table | What is the power consumption of the Orion-7 revision 1 in Turbo mode? | not-found | The information is not available in the uploaded documents. |
| t2 | table | What is the power consumption of the Orion-7 revision 1 in Boost mode? | not-found | The information is not available in the uploaded documents. |
| t3 | table | What is the depth of Berth 3 at Port Velmara? | rag | The harbor master of Port Velmara is Ines Falk. [harbor_report p.4] Ines Falk joined the Velmara Port Authorit |
| n1 | numerical | What is the total cost of phase one and phase two of the quay extension? | rag | Phase one of the quay extension cost 2.1 million euros. [harbor_report p.12] Phase two of the quay extension c |
| n2 | numerical | By how much does the Turbo mode flow rate exceed the Eco mode flow rate for the Orion-7 revision 1? | not-found | The information is not available in the uploaded documents. |
| a1 | math | Using the formula in the specification, what is the maximum electrical input power of the Orion-7 revision 1? | not-found | The information is not available in the uploaded documents. |
| a2 | math | What is the efficiency of the Orion-7 revision 1 in Standard mode, using the definition in the specification? | not-found | The information is not available in the uploaded documents. |
| b1 | ambiguous | What is the nominal supply voltage of the Orion-7? | rag | The nominal supply voltage of the Orion-7 in revision 1 is 24 volts DC. [orion_spec_v1 p.2] The nominal supply |
| b2 | ambiguous | What is the power consumption in Standard mode? | not-found | The information is not available in the uploaded documents. |
| k1 | contradictory | What is the warranty period of the Orion-7 revision 1? | rag | The warranty period for the Orion-7 in revision 1 is 36 months. [orion_spec_v1 p.6] The warranty period for th |
| k2 | contradictory | What is the maximum supply current of the Orion-7 revision 2? | rag | The maximum supply current of the Orion-7 in revision 1 is 3.2 amperes. [orion_spec_v1 p.2] The maximum supply |
| u1 | unanswerable | What is the Bluetooth range of the Orion-7? | rag | The operating temperature range of the Orion-7 in revision 1 is -10 to 55 degrees Celsius. [orion_spec_v1 p.5] |
| h2 | hallucination | What was the cost of the Marlow Quay crane replacement? | rag | The Marlow Quay crane replacement was approved in 2021. [harbor_report p.17] The Marlow Quay crane replacement |
| p1 | comparison | Compare the nominal supply voltage of the Orion-7 in revision 1 and revision 2. | compare | The nominal supply voltage of the Orion-7 in revision 1 is 24 volts DC. [orion_spec_v1 p.2] The nominal supply |
| p2 | comparison | How does the Standard mode flow rate differ between revision 1 and revision 2 of the Orion-7? | not-found | The information is not available in the uploaded documents. |
| p3 | comparison | Which revision of the Orion-7 has the lower maximum supply current? | rag | The maximum supply current of the Orion-7 in revision 1 is 3.2 amperes. [orion_spec_v1 p.2] The maximum supply |

## 6. Caveats
- Item-level bootstrap intervals describe sampling noise on *this* dataset only; they do not cover dataset-selection bias.
- Ground truth in the bundled dataset is synthetic and verified by construction; real-document performance needs your own PDFs (`--dataset`, `--pdf-dir`).
- Embedding-cosine metrics and the lexical faithfulness proxy are approximations, not LLM-judge scores.
- Latency/RAM are for the hardware listed above and include any other load on the machine.
