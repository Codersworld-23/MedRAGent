# MedRAGent — Current Project Status

**Project:** Medical RAG Optimization / Agentic Hierarchical RAG for Medical Textbooks  
**Working name:** **MedRAGent — Medical Hierarchical Evidence Retrieval Agent**  
**Proposed paper title:** *MedRAGent: An Agentic Hierarchical RAG Framework for Medical Textbook Question Answering*

---

## 1. Project Objective

The project investigates whether **agentic, hierarchical, and iterative retrieval** can improve medical question answering over conventional single-step RAG.

### Core research question

> Can autonomous, hierarchical and iterative retrieval make medical textbook RAG more accurate, reliable, and interpretable than conventional single-step RAG?

### Problem statement

Existing medical RAG systems primarily rely on static, single-step retrieval. This can fail when medical questions require multi-hop reasoning, evidence from different textbook sections, complementary evidence, or filtering of noisy/redundant context.

The proposed system will develop an **agentic hierarchical RAG framework for medical textbooks** that can:

1. decompose complex questions,
2. perform multi-stage retrieval,
3. verify and refine evidence,
4. iteratively reason,
5. generate grounded answers with supporting evidence.

The system will be compared against conventional RAG using retrieval, answer-quality, faithfulness, and hallucination-related metrics.

---

## 2. Research Strategy

The project will **not jump directly to agentic RAG**.

Planned progression:

```text
Medical Textbook Corpus
        ↓
Preprocessing
        ↓
Chunking Experiments
        ↓
Baseline Retrieval
   ├── BM25
   ├── Dense Retrieval
   └── Hybrid Retrieval
        ↓
Embedding Experiments
        ↓
Top-k Experiments
        ↓
Vanilla RAG Generation
        ↓
Baseline Evaluation
        ↓
Failure Analysis
        ↓
Agentic + Hierarchical RAG
        ↓
Iterative Retrieval / Verification
        ↓
Ablation Studies
        ↓
Final Comparison
```

The baseline-first approach is important so that the final contribution can demonstrate **which components actually improve performance**.

---

## 3. Current Dataset Collection

Current major datasets:

```text
datasets/
├── BioASQ/
├── MedMCQA/
├── MedQA/
│   ├── questions/
│   └── textbooks/
│       └── en/
│       └── zh_paragraph/
│       └── zh_sentence/
├── MMLU-Medical/
└── PubMedQA/
```

---

## 3.1 MedQA

### MedQA Status: PRESENT / USABLE

Structure:

```text
datasets/MedQA/
├── questions/
│   ├── Mainland/
│   ├── Taiwan/
│   └── US/
└── textbooks/
    └── en/
```

Example record:

```json
{
  "question": "...",
  "answer": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "...",
    "E": "..."
  },
  "meta_info": "step1",
  "answer_idx": "D"
}
```

**Intended role:** Primary medical textbook QA benchmark, especially MedQA-US.

---

## 4. Medical Textbook Corpus

Canonical corpus:

```text
datasets/MedQA/textbooks/en/
```

There are currently **18 English medical textbooks**:

| Textbook | Size (bytes) |
| --- | ---: |
| Anatomy_Gray.txt | 2,286,967 |
| Biochemistry_Lippincott.txt | 1,353,650 |
| Cell_Biology_Alberts.txt | 4,895,912 |
| First_Aid_Step1.txt | 672,786 |
| First_Aid_Step2.txt | 1,038,243 |
| Gynecology_Novak.txt | 5,647,726 |
| Histology_Ross.txt | 3,054,197 |
| Immunology_Janeway.txt | 3,329,538 |
| InternalMed_Harrison.txt | 22,375,808 |
| Neurology_Adams.txt | 8,386,547 |
| Obstentrics_Williams.txt | 6,585,140 |
| Pathology_Robbins.txt | 3,810,423 |
| Pathoma_Husain.txt | 399,974 |
| Pediatrics_Nelson.txt | 3,006,793 |
| Pharmacology_Katzung.txt | 5,141,125 |
| Physiology_Levy.txt | 3,067,415 |
| Psichiatry_DSM-5.txt | 2,905,014 |
| Surgery_Schwartz.txt | 11,478,117 |

No chunking or vectorization has been performed yet.

---

## 5. MedMCQA

### MedMCQA Status: PRESENT / USABLE

```text
datasets/MedMCQA/
├── dev.json
├── test.json
└── train.json
```

Current record counts:

| Split | Records |
| --- | ---: |
| Train | 182,822 |
| Dev | 4,183 |
| Test | 6,150 |
| **Total** | **193,155** |

Typical fields:

```text
question
exp
cop
opa
opb
opc
opd
subject_name
topic_name
id
choice_type
```

### Important correction

A folder previously assumed to be **MMLU-Medical** was actually MedMCQA. It has now been correctly placed at:

```text
datasets/MedMCQA/
```

The old miscellaneous folder was moved to:

```text
datasets/archive/MedMCQA_misc/
```

and retained.

---

## 6. MMLU-Medical

### MMLU-Medical Status: NOW COMPLETE

The actual MMLU-Medical dataset has now been downloaded.

Six subjects:

1. anatomy
2. clinical_knowledge
3. college_biology
4. college_medicine
5. medical_genetics
6. professional_medicine

Current counts:

| Subject | Test | Validation | Dev |
| --- | ---: | ---: | ---: |
| Anatomy | 135 | 14 | 5 |
| Clinical Knowledge | 265 | 29 | 5 |
| College Biology | 144 | 16 | 5 |
| College Medicine | 173 | 22 | 5 |
| Medical Genetics | 100 | 11 | 5 |
| Professional Medicine | 272 | 31 | 5 |
| **Total** | **1,089** | **123** | **30** |

The **1,089 test questions** correspond to the six medical MMLU subjects used in the MIRAGE/MEDRAG medical RAG evaluation setup.

Expected structure:

```text
datasets/MMLU-Medical/
├── anatomy/
├── clinical_knowledge/
├── college_biology/
├── college_medicine/
├── medical_genetics/
└── professional_medicine/
```

---

## 7. PubMedQA

### PubMedQA Status: COMPLETE

```text
datasets/PubMedQA/
├── ori_pqaa.json
├── ori_pqal.json
└── ori_pqau.json
```

Verified file sizes:

```text
ori_pqaa.json → 533,377,829 bytes
ori_pqal.json →   2,619,959 bytes
ori_pqau.json → 151,920,084 bytes
```

**Intended role:** Biomedical literature QA and generalization beyond textbook-only QA.

---

## 8. BioASQ

### BioASQ Status: PRESENT / USABLE

```text
datasets/BioASQ/
├── BioASQ-training13b/
│   └── training13b.json
└── Task13BGoldenEnriched/
    ├── 13B1_golden.json
    ├── 13B2_golden.json
    ├── 13B3_golden.json
    └── 13B4_golden.json
```

Training file: approximately **5,389 questions**.

The four golden files contain 85 questions each.

**Intended role:** Additional biomedical QA evaluation and generalization.

---

## 9. Dataset Strategy

The datasets will not all be treated equally.

| Dataset | Main purpose |
| --- | --- |
| **MedQA** | Primary medical textbook QA |
| **MedMCQA** | Medical-domain robustness |
| **MMLU-Medical** | Cross-benchmark medical generalization |
| **PubMedQA** | Biomedical literature QA |
| **BioASQ** | Additional biomedical research QA |

The main research story should remain centered around **medical textbook question answering**.

---

## 10. Research Papers Collected

Important papers currently available:

### RAG Evaluation

- **RAGChecker** — fine-grained RAG evaluation; separates retrieval and generation errors and supports claim-level analysis.

### RAG Background

- **RAG survey** — general RAG architectures, retrieval strategies, and evaluation.

### Medical RAG

- **MKRAG** — medical knowledge retrieval and multi-stage retrieval ideas.
- **i-MedRAG** — iterative retrieval for complex medical QA.
- **Med.ai ASK** — agentic biomedical QA, dynamic retrieval, and tool orchestration.

### Long-context / Multi-step RAG

- **BriefContext** — long-context medical RAG and lost-in-the-middle issues.
- **RaR** — multi-step retrieval and reasoning for radiology.

### Dataset / Benchmark

- **MedQA** — medical examination QA dataset.
- **MIRAGE / MEDRAG** — medical RAG benchmark covering multiple medical QA datasets and retrieval strategies.

---

## 11. Planned Baseline System

Before the agentic architecture, establish a strong vanilla RAG baseline.

## Preprocessing

- clean textbook text,
- remove irrelevant formatting,
- preserve headings/section boundaries,
- normalize whitespace,
- retain textbook/section metadata.

## Chunking

Initial configuration:

```text
500 words
50-word overlap
```

Experiments:

| Chunk size | Overlap |
| ---: | ---: |
| 250 | 50 |
| 500 | 50 |
| 500 | 100 |
| 1000 | 100 |
| 1500 | 200 |

## Retrieval

### BM25

Sparse lexical retrieval.

### Dense Retrieval

Initial embedding:

```text
BAAI/bge-base-en-v1.5
```

Potential later comparisons:

- smaller BGE models,
- MiniLM,
- medical-domain embeddings such as MedCPT.

### Hybrid

Combination of:

```text
BM25 + Dense Retrieval
```

## Top-k

Candidate values:

```text
1, 3, 5, 10, 15
```

---

## 12. Planned Generation

Baseline pipeline:

```text
Question
   +
Retrieved textbook evidence
   ↓
LLM
   ↓
Grounded answer + evidence/citations
```

An Ollama-hosted model such as `llama3.1:8b` can be used initially. Final model choice remains open.

---

## 13. Planned Evaluation

## Retrieval metrics

- Recall@k
- Precision@k
- MRR
- potentially nDCG

## Generation metrics

- answer accuracy
- answer relevance
- faithfulness
- hallucination rate

## Fine-grained evaluation

RAGChecker is a candidate framework for diagnosing:

```text
Retrieval errors
        vs.
Generation errors
```

This distinction is important because a wrong answer can occur either because the correct evidence was not retrieved or because the LLM failed despite receiving correct evidence.

---

## 14. Proposed Agentic Hierarchical RAG

Potential architecture:

```text
                         User Question
                              │
                              ▼
                    ┌──────────────────┐
                    │  Query Analyzer  │
                    └────────┬─────────┘
                             │
                    Simple / Complex?
                             │
              ┌──────────────┴──────────────┐
              │                             │
           Simple                        Complex
              │                             │
              ▼                             ▼
       Direct Retrieval             Question Decomposition
                                            │
                                            ▼
                                   Sub-question Retrieval
                                            │
                                            ▼
                                  Hierarchical Retrieval
                              ┌─────────────┴─────────────┐
                              │                           │
                        Broad retrieval             Fine retrieval
                              │                           │
                              └─────────────┬─────────────┘
                                            ▼
                                    Evidence Aggregation
                                            │
                                            ▼
                                     Evidence Verifier
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                           Sufficient?                 Not enough
                              │                           ▼
                             YES                    Query Refinement
                              │                           │
                              │                      Retrieve Again
                              │                           │
                              └──────────────┬────────────┘
                                             ▼
                                      Final Answer
                                             │
                                             ▼
                                   Evidence/Citations
```

Potential agent/modules:

1. **Query Analyzer** — determines question complexity and medical concepts.
2. **Query Decomposer** — splits complex questions into retrieval targets.
3. **Retriever Agent** — selects BM25, dense, or hybrid retrieval.
4. **Evidence Verifier** — checks whether evidence supports intermediate conclusions.
5. **Query Refiner** — reformulates queries when evidence is insufficient.
6. **Answer Synthesizer** — produces the final grounded answer.
7. **Stop/No-answer Mechanism** — avoids hallucinating when evidence is insufficient.

---

## 15. Hierarchical Retrieval

The textbook corpus naturally has multiple levels:

```text
Textbook
   ↓
Chapter
   ↓
Section
   ↓
Subsection
   ↓
Paragraph
   ↓
Chunk
```

Potential retrieval strategy:

```text
Question
   ↓
Relevant textbook
   ↓
Relevant chapter
   ↓
Relevant section
   ↓
Relevant passages
```

This is one of the central research directions.

---

## 16. Iterative Retrieval

For difficult questions:

```text
Question
   ↓
Initial retrieval
   ↓
Evidence analysis
   ↓
Missing information detected
   ↓
Query refinement
   ↓
Second retrieval
   ↓
Evidence verification
   ↓
Enough evidence?
   ├── No → retrieve again
   └── Yes → answer
```

A maximum iteration limit should be used.

---

## 17. Potential Novel Contributions

The project should investigate:

### A. Hierarchical retrieval

Does textbook hierarchy improve retrieval relevance and efficiency?

### B. Query decomposition

Does decomposing complex medical questions improve multi-hop QA?

### C. Iterative retrieval

Does retrieve → evaluate → refine → retrieve improve difficult questions?

### D. Evidence verification

Does explicit evidence verification reduce hallucinations?

### E. Adaptive retrieval

Can the system decide when simple questions need one retrieval step and complex questions need multiple steps?

### F. Retrieval strategy selection

Can an agent dynamically choose between BM25, dense, and hybrid retrieval?

### G. Confidence / stop mechanism

Can the system detect insufficient evidence and avoid unsupported answers?

---

## 18. Planned Ablation Studies

The final paper should isolate individual contributions rather than only comparing vanilla RAG with the full system.

| System | Hierarchical | Iterative | Verification | Decomposition |
| --- | ---: | ---: | ---: | ---: |
| Vanilla RAG | ✗ | ✗ | ✗ | ✗ |
| + Hierarchy | ✓ | ✗ | ✗ | ✗ |
| + Iteration | ✓ | ✓ | ✗ | ✗ |
| + Verification | ✓ | ✓ | ✓ | ✗ |
| Full MedRAGent | ✓ | ✓ | ✓ | ✓ |

This will make the research contribution more defensible.

---

## 19. Completed Work

- [x] Research topic selected
- [x] Research problem defined
- [x] Agentic + hierarchical RAG direction established
- [x] Baseline-first methodology established
- [x] Medical textbook corpus collected
- [x] 18 English medical textbooks verified
- [x] MedQA present
- [x] MedMCQA present and correctly identified
- [x] BioASQ present
- [x] PubMedQA completed
- [x] Actual MMLU-Medical downloaded
- [x] Six MMLU medical subjects downloaded
- [x] Dataset roles defined
- [x] Relevant RAG/medical-RAG research papers collected
- [x] Initial baseline architecture designed
- [x] Initial agentic architecture designed
- [x] **PHASE 1 COMPLETE**: Dataset audit, normalization, duplicate detection, and leakage checks finished.

---

## 20. Remaining Work

## Phase 1 — Dataset Finalization

- [x] Audit every dataset
- [x] Verify record counts
- [x] Verify train/dev/test splits
- [x] Validate answer fields
- [x] Detect malformed records
- [x] Normalize datasets into a common schema
- [x] Detect duplicate questions
- [x] Check cross-dataset overlap
- [x] Investigate question/textbook leakage
- [x] Decide final evaluation subsets
- [x] Create dataset manifest

## Phase 2 — Textbook Preprocessing

- [x] Clean textbook text
- [x] Preserve chapter/section metadata
- [x] Design hierarchical document representation
- [x] Implement chunking
- [x] Run chunk-size experiments

## Phase 3 — Baseline Retrieval

- [x] Implement BM25
- [x] Implement dense retrieval
- [x] Implement hybrid retrieval
- [x] Implement embedding model comparison runner
- [x] Implement top-k experiments
- [x] Build retrieval evaluation pipeline

Phase 3 implementation is complete. Run `python scripts/run_phase3.py` for BM25,
or use `--retriever all --models BAAI/bge-small-en-v1.5` for dense and hybrid
experiments. Chunk-level recall metrics require optional `relevant_chunk_ids`
labels in the question records; unlabeled runs still save ranked evidence.

## Phase 4 — Vanilla RAG

- [x] Implement prompt
- [x] Implement LLM generation
- [x] Add evidence citations
- [x] Evaluate accuracy
- [x] Evaluate faithfulness
- [x] Evaluate relevance
- [x] Analyze hallucinations

## Phase 5 — Agentic RAG

- [ ] Query complexity detection
- [ ] Query decomposition
- [ ] Hierarchical retrieval
- [ ] Iterative retrieval
- [ ] Evidence verification
- [ ] Query refinement
- [ ] Adaptive retrieval
- [ ] Stop/no-answer mechanism

## Phase 6 — Research Evaluation

- [ ] Baseline vs MedRAGent
- [ ] Ablation experiments
- [ ] Error analysis
- [ ] Retrieval-vs-generation error analysis
- [ ] Efficiency/cost analysis
- [ ] Statistical significance where appropriate

---

## 21. Research Novelty & Utility Roadmap

To ensure the project is highly novel and practically useful:

**Novelty Focus:**
We will move beyond simple semantic search (Vanilla RAG). The core novelty will be an **Adaptive Hierarchical Agent** that dynamically scopes its retrieval. Instead of just chunk retrieval, it will navigate the textbook structure (Textbook -> Chapter -> Section -> Paragraph) and autonomously decide if a question requires iterative multi-hop reasoning. We will also incorporate a novel **Confidence/Stop Mechanism** that forces the agent to admit ignorance if the verified evidence is insufficient, directly tackling medical hallucination.

**Utility Focus:**
The resulting system will be a highly interpretable RAG framework where medical professionals can trace exactly *why* an answer was given and see the precise textbook section it came from. The ablation studies will definitively prove the value of each agentic component over a standard RAG pipeline, providing a valuable open-source architecture for medical QA.

---

## 22. Immediate Next Step

**Phase 2 (Textbook Preprocessing) is complete.**

The immediate workflow is now:

```text
BASELINE BM25 RETRIEVAL
        ↓
BASELINE DENSE RETRIEVAL
        ↓
HYBRID RETRIEVAL SETUP
        ↓
TOP-K EVALUATION PIPELINE
```

The next steps involve implementing **Phase 3: Baseline Retrieval**, specifically setting up the initial retrieval infrastructure using the newly generated chunks.

---

## 23. Current One-Line Status

> **Phase 2 is complete! The medical textbooks have been successfully parsed into a hierarchy and chunked using both Baseline and Hierarchical strategies. The project is now ready for Phase 3: Baseline Retrieval Implementation.**
