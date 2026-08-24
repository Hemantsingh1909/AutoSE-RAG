# AutoSE-RAG Experimental Results & Data Dictionary

This directory contains persistent, machine-readable JSON artifacts produced by the automated benchmark suite (`scripts/run_experiments.py`). 

> **Important:** These files are programmatically generated raw research artifacts. Do not manually edit them.

---

## Directory Structure

```
results/
├── raw/
│   ├── R01.json                 # Requirement R01 raw model outputs and execution trace
│   ├── ...
│   └── R15.json                 # Requirement R15 raw model outputs and execution trace
├── retrieval_results.json       # Experiment 1 IR metrics (Sparse vs Dense vs Hybrid RRF)
├── generation_results.json      # Experiment 2 software metrics (M0 vs M1 vs M2)
└── statistical_results.json     # Paired t-tests, Cohen's d_z, and Holm-adjusted p-values
```

---

## File Schema & Data Dictionary

### 1. `retrieval_results.json`
Contains aggregate summary statistics and per-query evaluation metrics for the three retrieval modalities across all 15 queries:
- `summary`: Contains `Precision@1`, `Precision@3`, `Precision@4`, `Recall@4`, `MRR`, and `NDCG@4` with `mean`, `std`, `ci_95` (95% bootstrap confidence intervals, $B=1000$), and formatted display strings.
- `raw_metrics`: Per-query metric dictionaries for `tfidf_sparse`, `faiss_dense`, and `hybrid_rrf`.

### 2. `generation_results.json`
Contains aggregate summary statistics for the three generation paradigms:
- `M0: Raw LLM` (Ungrounded Baseline)
- `M1: Standard RAG` (Single-pass retrieval-grounded generation)
- `M2: Agentic RAG` (Multi-agent loop with TestGen, Critic, Refiner, and HITL)
Metrics recorded:
- `AST Syntactic Correctness (%)`
- `Constraint Coverage (%)` ($C_1 \dots C_k$)
- `Initial Oracle Pass Rate (%)`
- `Final Oracle Pass Rate (%)`
- `Hallucination Rate (UGCR) (%)`
- `Traceability Completeness (0-1)`
- `Revision Success Rate (%)`
- `Regression Rate (%)`

### 3. `statistical_results.json`
Contains paired statistical comparisons across matched queries:
- `traceability_m2_vs_m1`: Paired $t$-test statistic, Cohen's $d_z$ effect size, raw $p$-value, and step-down Holm-Bonferroni adjusted $p$-value.
- `traceability_m2_vs_m0`: Paired $t$-test statistic, Cohen's $d_z$ effect size, raw $p$-value, and Holm-adjusted $p$-value.
- `traceability_m1_vs_m0`: Paired $t$-test statistic, Cohen's $d_z$ effect size, raw $p$-value, and Holm-adjusted $p$-value.

### 4. `raw/RXX.json`
Contains the raw input prompt, AST parse status, initial and final code, generated tests, and diagnostic traces for each specific benchmark requirement:
- `requirement_id`: e.g. `R05`
- `category`: e.g. `Safety & SPoF`
- `m0_result`: Code generation, coverage, and oracle pass status under M0.
- `m1_result`: Code generation, coverage, and oracle pass status under M1.
- `m2_result`: Initial draft, critic classification ($F_1 - F_7$), revision code, and final pass status under M2.
- `diagnostic_trace`: Per-step critic diagnosis, revision action, and self-healing flag.
