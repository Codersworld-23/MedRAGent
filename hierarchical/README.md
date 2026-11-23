# Track A: Hierarchical Medical Retrieval (RAPTOR-Style & Multi-Level)

This directory contains the complete implementation, benchmark scripts, and evaluation results for **Track A: Hierarchical Retrieval** of the **MedRAGent** project.

---

## 1. Overview & Architecture

Standard medical RAG typically operates on flat, single-depth chunks, which causes failures when answers require broad cross-section context or multi-passage synthesis. **Track A** addresses this by structuring medical textbooks into a hierarchical summary tree (inspired by the RAPTOR framework, ICLR 2024).

```
                 [ User Clinical Query ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [ Tree Traversal ]        [ Collapsed Tree ]
     (Top-Down: L1 -> L0)      (Joint L0 + L1 Search)
               │                         │
     Layer 1: Thematic Summaries (100-120 clusters)
               │
     Layer 0: Raw Leaf Chunks (4,057 DSM-5 passages)
               │
               ▼
     [ Grounded Evidence with Passage Citations ]
```

### Components:
1. **Layer 0 (Leaf Nodes):** 4,057 raw Psichiatry DSM-5 chunks indexed in ChromaDB (`psichiatry_dsm5`).
2. **Layer 1 (Thematic Cluster Summaries):** Clustered higher-order summary nodes formed via semantic clustering (`MiniBatchKMeans`) on MiniLM embeddings (`psichiatry_dsm5_hierarchical`).
3. **Retrieval Strategies:**
   * **Tree Traversal (Algorithm 1):** Descends from summary root/clusters into relevant child leaf branches. Best for complex, multi-section clinical queries.
   * **Collapsed Tree (Algorithm 2):** Pools summaries and leaf chunks into a single index searched flat. Achieves high hit rates when paired with an evidence verifier.
   * **Flat Dense Baseline:** Direct nearest-neighbor search over Layer 0 leaves (fast path for direct factoid questions).

---

## 2. Directory Structure

```
hierarchical/
├── hierarchical_retrieval_eval.py   # RAPTOR tree builder, search algorithms, and multi-metric evaluator
├── run_track_a_benchmark.py         # Benchmark runner for 78 DSM-5 questions
├── hierarchy_parser.py              # Textbook parser for chapters, sections, and subsections
├── reports/
│   ├── hierarchical_evaluation_report.json   # Multi-metric evaluation output (Faithfulness, Relevance, BLEU, etc.)
│   └── track_a_5_parameters_results.json    # Official 5-parameter benchmark results (Hit, Recall, MRR, LLM Judge)
└── README.md                        # Documentation
```

---

## 3. Benchmark Results (Midsem Milestones)

### A. Shared 78 DSM-5 Benchmark (Exact 5 Parameters)

| Metric | Flat Dense Baseline | Hierarchical Collapsed Tree | Hierarchical Tree Traversal |
| :--- | :---: | :---: | :---: |
| **Hit (Source Found)** | 0.782 | **0.769** | 0.372 |
| **Recall of Sources** | 0.665 | **0.652** | 0.314 |
| **MRR** | 0.523 | **0.516** | 0.101 |
| **Correctness (1–5)** | 3.96 | **3.92** | 3.88 |
| **Faithfulness (1–5)** | 3.96 | **3.88** | 3.88 |

### B. Retrieval Quality & Faithfulness Comparison

| Strategy | Mean Faithfulness | Hallucination Rate | Mean Relevance | Latency (sec) |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Dense Baseline** | 0.78 | 0.22 | 0.146 | 1.69s |
| **RAPTOR Collapsed Tree** | 0.76 | 0.24 | 0.147 | 1.11s |
| **RAPTOR Tree Traversal** | 0.72 | 0.28 | 0.151 | 3.35s |

---

## 4. How to Run

### Build Tree & Run Evaluation:
```bash
python hierarchical/hierarchical_retrieval_eval.py
```

### Run 78 DSM-5 Benchmark Evaluation:
```bash
python hierarchical/run_track_a_benchmark.py
```
