# Retrieval-configuration sweep (hf)
_Generated 2026-11-01 00:50 · dataset `synthetic_v1` (sha256 `043c1049fbbc`) · 32 items · hardware: cpu, 8 cores_

**Coverage gap:** no items for category: ocr. Those categories are not evaluated by this dataset.

## 1. Configurations compared
| name | fingerprint | dense | bm25 | rerank | parent | multi-query | chunk/parent chars | LLM | embedder | reranker |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid+rerank+parent (default) | d15640492411 | True | True | True | True | False | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| dense only | b8e63e47aa13 | True | False | False | False | False | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| bm25 only | 096413f85da0 | False | True | False | False | False | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| hybrid, no rerank | f85e8ab7bfa8 | True | True | False | False | False | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| hybrid+rerank, no parent expand | d4fcfbe35eca | True | True | True | False | False | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| hybrid+rerank+parent, multi-query | 717d48f729c9 | True | True | True | True | True | 900/2400 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| chunks 400/parent 1200 | 1b026183ade3 | True | True | True | True | False | 400/1200 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| chunks 1500/parent 3600 | 84ab6078f881 | True | True | True | True | False | 1500/3600 | Qwen/Qwen2.5-0.5B-Instruct | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-6-v2 |

## 2. Selection

Method: each configuration is scored on the **dev** split with objective = 0.30·f1 + 0.15·page_recall@5 + 0.10·mrr + 0.20·correct_abstention + 0.15·citation_precision + 0.10·faithfulness (normalised). Reported numbers below are on the **held-out test** split.

> **Tie.** 2 configurations are tied within 0.005 on the objective; `dense only` was picked only because it has the lowest mean latency. This is NOT evidence that it is better.

| configuration | objective (selection split) | status |
|---|---|---|
| dense only | 0.736 | SELECTED (tie-break) |
| hybrid, no rerank | 0.736 | tied |
| bm25 only | 0.719 |  |
| hybrid+rerank+parent, multi-query | 0.713 |  |
| hybrid+rerank+parent (default) | 0.712 |  |
| hybrid+rerank, no parent expand | 0.712 |  |
| chunks 1500/parent 3600 | 0.712 |  |
| chunks 400/parent 1200 | 0.712 |  |

### Is `dense only` really better? (paired bootstrap on the test items)
| vs | metric | mean diff | 95% CI | n items | verdict |
|---|---|---|---|---|---|
| hybrid+rerank+parent (default) | f1 | 0.069 | [0.017, 0.131] | 12 | distinguishable |
| hybrid+rerank+parent (default) | page_recall@5 | 0.125 | [-0.167, 0.375] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent (default) | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| bm25 only | f1 | 0.014 | [-0.030, 0.071] | 12 | NOT distinguishable (CI includes 0) |
| bm25 only | page_recall@5 | -0.125 | [-0.333, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| bm25 only | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | f1 | 0.002 | [-0.017, 0.023] | 12 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | page_recall@5 | -0.042 | [-0.125, 0.000] | 12 | NOT distinguishable (CI includes 0) |
| hybrid, no rerank | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid+rerank, no parent expand | f1 | 0.069 | [0.017, 0.131] | 12 | distinguishable |
| hybrid+rerank, no parent expand | page_recall@5 | 0.125 | [-0.167, 0.375] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank, no parent expand | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent, multi-query | f1 | 0.069 | [0.017, 0.131] | 12 | distinguishable |
| hybrid+rerank+parent, multi-query | page_recall@5 | 0.125 | [-0.167, 0.375] | 12 | NOT distinguishable (CI includes 0) |
| hybrid+rerank+parent, multi-query | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| chunks 400/parent 1200 | f1 | 0.069 | [0.017, 0.131] | 12 | distinguishable |
| chunks 400/parent 1200 | page_recall@5 | 0.125 | [-0.167, 0.375] | 12 | NOT distinguishable (CI includes 0) |
| chunks 400/parent 1200 | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |
| chunks 1500/parent 3600 | f1 | 0.069 | [0.017, 0.131] | 12 | distinguishable |
| chunks 1500/parent 3600 | page_recall@5 | 0.125 | [-0.167, 0.375] | 12 | NOT distinguishable (CI includes 0) |
| chunks 1500/parent 3600 | correct_abstention | 0.000 | [0.000, 0.000] | 2 | NOT distinguishable (CI includes 0) |

_A configuration is only called better where the interval excludes 0. With a small dataset most differences will not be._

## 3. Metrics (held-out test split)

Counts - hybrid+rerank+parent (default): 14 evaluated / 0 skipped, dense only: 14 evaluated / 0 skipped, bm25 only: 14 evaluated / 0 skipped, hybrid, no rerank: 14 evaluated / 0 skipped, hybrid+rerank, no parent expand: 14 evaluated / 0 skipped, hybrid+rerank+parent, multi-query: 14 evaluated / 0 skipped, chunks 400/parent 1200: 14 evaluated / 0 skipped, chunks 1500/parent 3600: 14 evaluated / 0 skipped

### Retrieval metrics
Did the right pages come back? Scored on answerable items, independent of what the LLM then said (page-level: a hit is a retrieved chunk from a ground-truth page).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| hit@1 | 0.667 [0.417-0.917] n=12 | 0.667 [0.417-0.917] n=12 | 0.833 [0.583-1.000] n=12 | 0.667 [0.417-0.917] n=12 | 0.667 [0.417-0.917] n=12 | 0.667 [0.417-0.917] n=12 | 0.667 [0.417-0.917] n=12 | 0.667 [0.417-0.917] n=12 |
| hit@3 | 0.750 [0.500-1.000] n=12 | 0.833 [0.583-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.833 [0.583-1.000] n=12 | 0.750 [0.500-1.000] n=12 | 0.750 [0.500-1.000] n=12 | 0.750 [0.500-1.000] n=12 | 0.750 [0.500-1.000] n=12 |
| hit@5 | 0.833 [0.583-1.000] n=12 | 0.917 [0.750-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.833 [0.583-1.000] n=12 | 0.833 [0.583-1.000] n=12 | 0.833 [0.583-1.000] n=12 | 0.833 [0.583-1.000] n=12 |
| page_recall@1 | 0.500 [0.292-0.750] n=12 | 0.542 [0.292-0.792] n=12 | 0.667 [0.458-0.875] n=12 | 0.542 [0.292-0.792] n=12 | 0.500 [0.292-0.750] n=12 | 0.500 [0.292-0.750] n=12 | 0.500 [0.292-0.750] n=12 | 0.500 [0.292-0.750] n=12 |
| page_recall@3 | 0.625 [0.417-0.833] n=12 | 0.750 [0.500-0.958] n=12 | 0.958 [0.875-1.000] n=12 | 0.792 [0.542-1.000] n=12 | 0.625 [0.417-0.833] n=12 | 0.625 [0.417-0.833] n=12 | 0.625 [0.417-0.833] n=12 | 0.625 [0.417-0.833] n=12 |
| page_recall@5 | 0.750 [0.541-0.958] n=12 | 0.875 [0.667-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.917 [0.792-1.000] n=12 | 0.750 [0.541-0.958] n=12 | 0.750 [0.541-0.958] n=12 | 0.750 [0.541-0.958] n=12 | 0.750 [0.541-0.958] n=12 |
| mrr | 0.711 [0.478-0.933] n=12 | 0.753 [0.533-0.944] n=12 | 0.903 [0.764-1.000] n=12 | 0.783 [0.592-0.958] n=12 | 0.711 [0.478-0.933] n=12 | 0.711 [0.478-0.933] n=12 | 0.711 [0.478-0.933] n=12 | 0.711 [0.478-0.933] n=12 |
| ndcg@5 | 0.675 [0.468-0.871] n=12 | 0.763 [0.567-0.927] n=12 | 0.930 [0.841-1.000] n=12 | 0.800 [0.628-0.944] n=12 | 0.675 [0.468-0.871] n=12 | 0.675 [0.468-0.871] n=12 | 0.675 [0.468-0.871] n=12 | 0.675 [0.468-0.871] n=12 |
| context_precision | 0.704 [0.484-0.919] n=12 | 0.711 [0.517-0.885] n=12 | 0.899 [0.775-1.000] n=12 | 0.761 [0.589-0.917] n=12 | 0.704 [0.484-0.919] n=12 | 0.704 [0.484-0.919] n=12 | 0.704 [0.484-0.919] n=12 | 0.704 [0.484-0.919] n=12 |
| context_recall | 0.791 [0.564-0.977] n=12 | 0.895 [0.719-1.000] n=12 | 1.000 [1.000-1.000] n=12 | 0.976 [0.929-1.000] n=12 | 0.791 [0.564-0.977] n=12 | 0.791 [0.564-0.977] n=12 | 0.791 [0.564-0.977] n=12 | 0.791 [0.564-0.977] n=12 |

### Generation metrics
Is the answer right? `semantic_similarity` / `answer_relevancy` are embedding cosines (only as good as the embedder); `faithfulness` is a deterministic lexical-support proxy, not an LLM judge. An abstention on an answerable question scores 0 for EM/F1/similarity.

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| exact_match | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 | 0.000 [0.000-0.000] n=12 |
| contains_answer | 0.250 [0.000-0.500] n=12 | 0.417 [0.167-0.667] n=12 | 0.500 [0.250-0.750] n=12 | 0.417 [0.167-0.667] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 |
| f1 | 0.280 [0.133-0.444] n=12 | 0.349 [0.218-0.490] n=12 | 0.335 [0.201-0.483] n=12 | 0.347 [0.221-0.489] n=12 | 0.280 [0.133-0.444] n=12 | 0.280 [0.133-0.444] n=12 | 0.280 [0.133-0.444] n=12 | 0.280 [0.133-0.444] n=12 |
| semantic_similarity | 0.561 [0.371-0.739] n=12 | 0.680 [0.540-0.786] n=12 | 0.633 [0.445-0.784] n=12 | 0.681 [0.541-0.788] n=12 | 0.561 [0.371-0.739] n=12 | 0.561 [0.371-0.739] n=12 | 0.561 [0.371-0.739] n=12 | 0.561 [0.371-0.739] n=12 |
| answer_relevancy | 0.816 [0.745-0.880] n=9 | 0.842 [0.777-0.906] n=11 | 0.825 [0.755-0.891] n=10 | 0.831 [0.771-0.892] n=11 | 0.816 [0.745-0.880] n=9 | 0.816 [0.745-0.880] n=9 | 0.816 [0.745-0.880] n=9 | 0.816 [0.745-0.880] n=9 |
| faithfulness | 0.796 [0.648-0.944] n=9 | 0.742 [0.591-0.879] n=11 | 0.767 [0.633-0.917] n=10 | 0.742 [0.591-0.879] n=11 | 0.796 [0.648-0.944] n=9 | 0.796 [0.648-0.944] n=9 | 0.796 [0.648-0.944] n=9 | 0.796 [0.648-0.944] n=9 |

### Citation metrics
Page-level precision/recall of the cited pages vs. ground truth; `grounded` = the snippet shown for a citation really occurs on the cited page (fabrication check).

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| precision | 0.778 [0.444-1.000] n=9 | 0.727 [0.455-1.000] n=11 | 0.800 [0.500-1.000] n=10 | 0.727 [0.455-1.000] n=11 | 0.778 [0.444-1.000] n=9 | 0.778 [0.444-1.000] n=9 | 0.778 [0.444-1.000] n=9 | 0.778 [0.444-1.000] n=9 |
| recall | 0.417 [0.208-0.625] n=12 | 0.500 [0.292-0.750] n=12 | 0.542 [0.292-0.792] n=12 | 0.500 [0.292-0.750] n=12 | 0.417 [0.208-0.625] n=12 | 0.417 [0.208-0.625] n=12 | 0.417 [0.208-0.625] n=12 | 0.417 [0.208-0.625] n=12 |
| grounded | 1.000 [1.000-1.000] n=9 | 0.909 [0.727-1.000] n=11 | 0.900 [0.700-1.000] n=10 | 0.909 [0.727-1.000] n=11 | 1.000 [1.000-1.000] n=9 | 1.000 [1.000-1.000] n=9 | 1.000 [1.000-1.000] n=9 | 1.000 [1.000-1.000] n=9 |

### Reliability / abstention
On unanswerable and hallucination-bait questions the correct behaviour is to say the information is unavailable.

| metric | hybrid+rerank+parent (default) | dense only | bm25 only | hybrid, no rerank | hybrid+rerank, no parent expand | hybrid+rerank+parent, multi-query | chunks 400/parent 1200 | chunks 1500/parent 3600 |
|---|---|---|---|---|---|---|---|---|
| false_refusal_rate | 0.250 [0.000-0.500] n=12 | 0.083 [0.000-0.250] n=12 | 0.167 [0.000-0.417] n=12 | 0.083 [0.000-0.250] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 | 0.250 [0.000-0.500] n=12 |
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
| latency_mean_s | 10.947 | 9.667 | 9.522 | 10.232 | 9.905 | 18.197 | 9.657 | 10.752 |
| latency_p50_s | 10.152 | 9.436 | 9.191 | 10.027 | 8.956 | 18.040 | 8.803 | 9.569 |
| latency_p95_s | 16.268 | 12.503 | 15.393 | 15.283 | 15.041 | 24.110 | 14.032 | 15.425 |
| first_token_mean_s | 8.044 | 6.949 | 6.060 | 7.296 | 7.450 | 15.654 | 7.242 | 8.167 |
| first_token_p95_s | 10.125 | 8.865 | 7.395 | 9.518 | 8.921 | 18.889 | 8.702 | 9.837 |
| tokens_per_s_mean | 1012.1 | 429.2 | 520.8 | 703.9 | 828.5 | 260.1 | 738.0 | 485.5 |
| peak_ram_mb | 2326 | 2480 | 2252 | 2272 | 2465 | 2333 | 2364 | 2481 |
| baseline_ram_mb | 1298 | 2446 | 1048 | 2214 | 2397 | 2306 | 2144 | 2412 |
| peak_vram_mb | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pdf_processing_s | 0.006 | 0.006 | 0.006 | 0.006 | 0.005 | 0.005 | 0.005 | 0.005 |
| indexing_s | 4.540 | 3.018 | 3.479 | 3.105 | 2.888 | 2.778 | 3.287 | 2.930 |
| chunks | 46 | 46 | 46 | 46 | 46 | 46 | 76 | 46 |

## 4. Results by category - `dense only` (all 32 items)
_Descriptive only: includes the dev items used for selection, and several categories have just 1-3 items._

| category | n | F1 | contains answer | page recall@5 | citation recall | correct abstention | note |
|---|---|---|---|---|---|---|---|
| factual | 3 | 0.56 | 1.00 | 1.00 | 1.00 | n/a |  |
| multi_hop | 2 | 0.14 | 0.00 | 0.50 | 0.00 | n/a |  |
| cross_page | 2 | 0.25 | 0.00 | 1.00 | 0.25 | n/a |  |
| table | 3 | 0.16 | 0.67 | 1.00 | 0.67 | n/a |  |
| numerical | 2 | 0.16 | 0.50 | 1.00 | 0.75 | n/a |  |
| math | 2 | 0.06 | 0.00 | 1.00 | 0.75 | n/a |  |
| ocr | 0 | - | - | - | - | - | no items (not covered) |
| long_document | 2 | 0.33 | 0.50 | 1.00 | 0.00 | n/a |  |
| ambiguous | 2 | 0.30 | 0.00 | 1.00 | 0.50 | n/a |  |
| contradictory | 2 | 0.22 | 0.00 | 1.00 | 0.50 | n/a |  |
| unanswerable | 3 | n/a | n/a | n/a | n/a | 1.00 |  |
| citation | 2 | 0.62 | 1.00 | 1.00 | 1.00 | n/a |  |
| context_retention | 2 | 0.29 | 0.50 | 1.00 | 0.50 | n/a |  |
| hallucination | 2 | n/a | n/a | n/a | n/a | 0.50 |  |
| comparison | 3 | 0.33 | 0.00 | 0.83 | 0.17 | n/a |  |

## 5. Failures - `dense only` (all items)
| id | category | question | route | system answer |
|---|---|---|---|---|
| m1 | multi_hop | In which year did the harbor master of Port Velmara join the port authority? | unverified | The information is not available in the uploaded documents. |
| m2 | multi_hop | How many months passed between the release of revision 1 and the release of revision 2 of the Orion-7 specification? | rag | 36 months [orion_spec_v2 p.6] |
| c1 | cross_page | How many years passed between the approval and the completion of the Marlow Quay crane replacement? | unverified | The information is not available in the uploaded documents. |
| c2 | cross_page | What is the Boost mode flow rate of the Orion-7 revision 1, and what happens to the flow rate above 50 degrees Celsius? | rag | The Boost mode flow rate of the Orion-7 revision 1 is 70 L/min. [orion_spec_v1 p.5] Above 50 degrees Celsius,  |
| t3 | table | What is the depth of Berth 3 at Port Velmara? | unverified | The information is not available in the uploaded documents. |
| n1 | numerical | What is the total cost of phase one and phase two of the quay extension? | rag | The total cost of phase one of the quay extension is 2.1 million euros, and the total cost of phase two is 3.4 |
| a1 | math | Using the formula in the specification, what is the maximum electrical input power of the Orion-7 revision 1? | rag | The maximum electrical input power of the Orion-7 revision 1 is 96 watts. [orion_spec_v1 p.2] |
| a2 | math | What is the efficiency of the Orion-7 revision 1 in Standard mode, using the definition in the specification? | rag | The efficiency of the Orion-7 revision 1 in Standard mode is 0.9. [orion_spec_v1 p.5] |
| l1 | long_document | In which year was the Marlow Quay crane replacement approved? | unverified | The information is not available in the uploaded documents. |
| b1 | ambiguous | What is the nominal supply voltage of the Orion-7? | rag | The nominal supply voltage of the Orion-7 is 24 volts DC. [orion_spec_v1 p.2] |
| b2 | ambiguous | What is the power consumption in Standard mode? | rag | The power consumption in Standard mode is 90 W. [orion_spec_v2 p.3] |
| k1 | contradictory | What is the warranty period of the Orion-7 revision 1? | rag | The warranty period of the Orion-7 revision 1 is 36 months. [orion_spec_v1 p.6] |
| k2 | contradictory | What is the maximum supply current of the Orion-7 revision 2? | rag | 2.1 amperes [orion_spec_v2 p.2] |
| r2 | context_retention | When did she join the port authority? | unverified | The information is not available in the uploaded documents. |
| h2 | hallucination | What was the cost of the Marlow Quay crane replacement? | rag | The cost of the Marlow Quay crane replacement was 2.1 million euros. [harbor_report p.17] |
| p1 | comparison | Compare the nominal supply voltage of the Orion-7 in revision 1 and revision 2. | compare | The nominal supply voltage of the Orion-7 in revision 1 is 24 volts DC, while in revision 2 it is 48 volts DC. |
| p2 | comparison | How does the Standard mode flow rate differ between revision 1 and revision 2 of the Orion-7? | compare | The Standard mode flow rate differs between revision 1 and revision 2 of the Orion-7 by 10%. [orion_spec_v1 p. |
| p3 | comparison | Which revision of the Orion-7 has the lower maximum supply current? | unverified | The information is not available in the uploaded documents. |

## 6. Caveats
- Item-level bootstrap intervals describe sampling noise on *this* dataset only; they do not cover dataset-selection bias.
- Ground truth in the bundled dataset is synthetic and verified by construction; real-document performance needs your own PDFs (`--dataset`, `--pdf-dir`).
- Embedding-cosine metrics and the lexical faithfulness proxy are approximations, not LLM-judge scores.
- Latency/RAM are for the hardware listed above and include any other load on the machine.
