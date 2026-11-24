 

# MedRAGent: Research Workflow and Publication Roadmap

**Working title:** *MedRAGent: Evidence-Grounded Retrieval for Subjective Medical Question Answering Across Multiple Textbooks*

**Repository:** `Codersworld-23/MedRAGent`

**Current status (24 November 2026):** The reproducible MCQ baseline is
complete, including one frozen test evaluation. The 18-textbook corpus
(125,847 chunks) and a source-cited subjective-answer contract are
implemented. A model-assisted, textbook-mapped 250-question BioASQ
subjective benchmark (`bioasq13b_summary_model_reviewed_v1.jsonl`) is frozen
for engineering work (72 partial-support, 178 expected-abstention). A
full-featured iterative-retrieval pipeline (BM25, dense, hybrid, RRF fusion,
LLM query refinement, up to 3 rounds, scope refusal, audit traces) is in
`iterative_retrival/` with its own DSM-5 evaluation (78 generated + 91 MedQA
psychiatry questions); it is **not yet integrated** with the 18-textbook
subjective pipeline. Dense FAISS index, corrective retrieval,
agent/controller routing, chunking ablations, fusion experiments, the
dashboard, and external validation are pending. The project is not clinically
validated or ready for clinical use.

---

## 0. Current project status at a glance

| Final objective | Current evidence | Status | What remains before a final claim |
| --- | --- | --- | --- |
| 1. Show why RAG improves over an LLM-only baseline | Frozen MedQA-US/DSM-5 test: LLM-only 45.64%, BM25 48.15%, dense 47.92%, hybrid RRF 48.63% on 1,273 MCQ questions (all RAG gains significant after Holm correction). Iterative RAG (DSM-5 only): closed-book 91.2% → rag_r1 92.3% → rag_r3 92.3% on 91 psychiatry MCQ; on 78 generated DSM-5 Qs, 3-round recall 0.735 vs. 1-round 0.693, correctness 4.47 vs. 4.22. | **MCQ baseline complete; iterative DSM-5 prototype evaluated** | Repeat on the frozen 18-textbook subjective benchmark to prove gains transfer to free-form answers. |
| 2. Iterative and hierarchical retrieval | DSM-5 hierarchical prototype exists. `iterative_retrival/` has a full iterative pipeline: BM25/Chroma dense/hybrid retrieval, LLM query refinement (planner), RRF fusion, up to 3 rounds, scope refusal, audit traces, and evaluation over generated + MedQA Qs. 100% out-of-scope refusal rate, 100% citation validity. | **Iterative fully implemented for DSM-5; not integrated with 18-textbook corpus** | Refactor & integrate iterative pipeline with `src/` package and 18-book indexes; run the same frozen subjective questions. |
| 3. Chunking, storage, and indexing optimization | 18 books, 125,847 chunks, corpus manifest, and persisted full-corpus BM25 index are present. No chunking ablations, no full dense index, no storage comparison yet. | **Partially complete** | Build full 18-book dense FAISS index; run chunking ablations (fixed-window, sentence-aware, parent-child); compare storage backends; measure index size/memory/reload time. |
| 4. Corrective/agentic workflow and broader validation | Structured subjective responses and abstention policy implemented. Iterative retrieval's scope-check and planner approach is a precursor but not yet a corrective/controller workflow. | **Not yet a workflow** | Add bounded corrective retrieval (1 retry on evidence-quality failure) and deterministic controller routing (flat / iterative / hierarchical / abstain). Validate on reviewed subjective data + external datasets (PubMedQA, MEDIQA-AnS). |
| 5. Retrieval fusion and UI | BM25 + dense RRF fusion implemented in both `src/hybrid.py` and `iterative_retrival/medx_retrieval.py`. No UI dashboard. | **Fusion baseline complete; UI not started** | Tune/ablate fusion weights, test optional reranker; build source-inspection dashboard. |
| Subjective transition | BioASQ 13b exported as 1,283 summary Qs; 250-Q model-reviewed subset frozen (`bioasq13b_summary_model_reviewed_v1.jsonl`: 72 partial-support, 178 expected-abstention). **Option A adopted:** 18-textbook benchmark generator created (`scripts/generate_textbook_subjective_questions.py`) to produce ~400 in-corpus Qs with ground-truth chunk IDs across all 18 books, combining with 50 BioASQ items for abstention controls. | **Engineering-ready; Option A in progress** | Run Option A generator across 18 textbooks to freeze `textbook_subjective_benchmark_v1.jsonl`; retain BioASQ for abstention/external evaluation. |

### 0.1 Completed, reproducible assets

- Frozen MedQA-US test outputs for LLM-only, BM25, dense, and hybrid RRF in
  `experiments/results/test_frozen_2026-11-23/`.
- Paired McNemar tests, paired bootstrap confidence intervals, and
  Holm-Bonferroni correction in `paired_statistics.json`.
- Validated 18-textbook corpus manifest: 125,847 chunks and 97,752,383
  characters in `experiments/manifests/full_textbook_corpus_manifest.json`.
- Persisted full-corpus BM25 index in `indexes/full_textbooks/`.
- Full iterative RAG pipeline in `iterative_retrival/`:
  - `medx_retrieval.py` — BM25Index, ChromaRetriever (dense),
    HybridRetriever, GeminiEmbeddingFunction, LocalEmbeddingFunction,
    HashingEmbeddingFunction; RRF fusion with configurable weights;
    `iterative_retrieve()` with up to N rounds, LLM or keyword query
    refinement, scope checking, and grounded answer generation.
  - `medx_llm.py` — Multi-provider LLM client (NVIDIA, Groq, Gemini,
    OpenRouter, Ollama) with rate limiting, disk caching, and automatic
    fallback.
  - `medx_eval.py` — Full evaluation harness: MedQA filter-to-psychiatry,
    closed-book / rag_r1 / rag_r3 MCQ comparison, generated DSM-5 Q
    evaluation with retrieval metrics (hit, recall, MRR), judge-LLM
    grading (correctness, completeness, faithfulness), citation validation,
    scope-refusal tracking, and markdown/JSON summary report.
  - `generate_rag_eval_dd.py` — DSM-5 question generator for evaluation.
  - `Gemini_RAG_Pipeline.ipynb` — Notebook prototype.
  - Evaluation results (`eval_results/summary.json`):
    - MedQA psychiatry (91 Qs): closed-book 91.2%, rag_r1 92.3%,
      rag_r3 92.3%. 100% out-of-scope refusal (30 controls).
    - Generated DSM-5 (78 Qs): 1-round recall 0.693, hit 0.779;
      3-round recall 0.735 (+6.1%), hit 0.795, correctness 4.47/5,
      faithfulness 4.54/5, 100% citation validity.
  - These results are exploratory DSM-5-only and must not be presented as
    clinically validated 18-textbook results.
- Subjective JSON response contract with citations, evidence status,
  limitations, and abstention in `src/subjective.py`.
- BioASQ summary export in
  `datasets/subjective/bioasq13b_summary_unreviewed.jsonl`: 1,283 records,
  each with a question, ideal/reference answer, original PubMed evidence for
  provenance, and blank textbook-gold evidence fields.
- Model-assisted technical review set in
  `datasets/subjective/bioasq13b_summary_model_reviewed_v1.jsonl`: 250
  deterministic questions mapped by a transparent streaming lexical method
  (`create_model_reviewed_subjective_subset.py`); 72 have partial-support
  chunk mappings and 178 default to expected abstention. This removes the
  development blocker completely — no external human review is needed.
- Full 18-textbook candidate attachment in
  `datasets/subjective/bioasq13b_summary_textbook_candidates.jsonl`: 1,283
  records with top-k textbook candidate chunks attached.

### 0.2 Important boundaries

- MedQA-US and MedMCQA remain MCQ regression/ablation datasets only.
- BioASQ reference evidence is PubMed-based. It cannot be reported as gold
  evidence for the 18 textbook corpus.
- Automatic textbook retrieval candidates are never clinician gold evidence.
  The versioned model-assisted mapping may be used for technical development;
  human review is optional, and required only for claims of clinical-quality
  validation.
- Citation IDs that are structurally valid are not, by themselves, proof of
  factual correctness or clinical safety.

---

## 1. Project goal

MedRAGent will develop and evaluate a medical retrieval-augmented generation
system that:

1. retrieves evidence from all 18 medical textbooks in `datasets/MedRAG/chunk/`;
2. answers free-form, subjective medical questions;
3. cites the exact textbook passages used to support each answer;
4. distinguishes well-supported, partially supported, and unsupported answers;
5. presents answers through a dashboard; and
6. compares its quantitative results with published RAG systems in the
   medical field.

The 18-textbook full-corpus system is the primary research goal. The MedQA-US
MCQ study is an early, controlled baseline that establishes the retrieval
methods and evaluation infrastructure.

### 1.1 Main final research question

> Does evidence-grounded retrieval over all 18 medical textbooks improve
> answer accuracy, faithfulness, citation quality, completeness, and
> abstention behavior for subjective medical questions compared with
> LLM-only and published medical RAG baselines?

### 1.2 Secondary questions

- Which retrieval strategy—BM25, dense, hybrid, iterative, corrective, or
  agent-controlled—works best across all textbooks?
- Does retrieving across multiple textbooks improve coverage over DSM-5 only?
- Do answers remain grounded in retrieved passages?
- Can the system identify insufficient evidence and abstain?
- How do results compare with quantitative results reported by existing papers?
- What is the latency, cost, context size, and index size of each system?

---

## 2. End-to-end project architecture

### 2.1 Objective baseline

The objective baseline uses multiple-choice MedQA-US questions:

- MedQA-US development: 1,272 questions;
- MedQA-US test: 1,273 questions;
- Evidence corpus for this baseline: Psychiatry DSM-5 only;
- Answer output: one option letter;
- Purpose: controlled comparison of retrieval strategies.

This baseline must not be described as the final subjective system.

### 2.2 Final subjective system

The final system will use:

- a question entered through a dashboard;
- retrieval over all 18 textbook corpora;
- a bounded retrieval and generation pipeline;
- a free-form response;
- source chunk identifiers and passage excerpts;
- an evidence status and abstention reason;
- an answer with uncertainty or coverage limitations when appropriate.

The final response should be structured while remaining readable:

```text
Answer
Evidence status: supported / partial / unsupported
Key explanation
Sources
Limitations
Abstention reason, if applicable
```

The dashboard will show:

- the answer;
- retrieved chunks;
- source textbook and title;
- relevance score;
- citation snippets;
- retrieval strategy;
- latency;
- evidence coverage;
- optional model and configuration details.

---

## 3. Evidence corpus

### 3.1 The 18 textbooks

`datasets/MedQA/textbooks/en/` contains 18 textbooks:

1. Anatomy — Gray
2. Biochemistry — Lippincott
3. Cell Biology — Alberts
4. First Aid — Step 1
5. First Aid — Step 2
6. Gynecology — Novak
7. Histology — Ross
8. Immunology — Janeway
9. Internal Medicine — Harrison
10. Neurology — Adams
11. Obstetrics — Williams
12. Pathology — Robbins
13. Pathoma — Husain
14. Pediatrics — Nelson
15. Pharmacology — Katzung
16. Physiology — Levy
17. Psychiatry — DSM-5
18. Surgery — Schwartz

The processed repository also contains 125,847 chunks across these books under
`datasets/MedRAG/chunk/`.

Each system must use the exact same chunk boundaries and source files during a
comparison. Chunking, hierarchy, or storage changes require separate
experiments.

### 3.2 Corpus metadata to add

Before final retrieval, the corpus will require a versioned manifest containing:

- textbook name and edition;
- source file hash;
- chunk ID;
- chunk text;
- title;
- chapter or section metadata only when trustworthy;
- corpus version;
- preprocessing version;
- license or permitted-use status.

Hierarchy-aware retrieval may begin only when trustworthy chapter or section
metadata exists. Chunk order or the `title` field must not be treated as
chapter hierarchy.

---

## 4. Subjective QA datasets

The project must stop evaluating only MedQA-US for its final claims.

### 4.1 BioASQ

BioASQ is a primary candidate for subjective evaluation because it contains:

- factoid questions;
- list questions;
- yes/no questions;
- summary questions;
- source documents;
- snippets;
- ideal answers.

Use BioASQ only for the task types that match the final user interface. Summary
and list questions are especially relevant. Report the exact training, test,
year, task, and gold-answer version used.

### 4.2 PubMedQA

PubMedQA is a complementary evidence-grounding candidate. The labeled set
contains 1,000 expert-labeled questions and the contextual set is larger.

It is suitable for:

- yes/no with rationale;
- evidence-grounded decision-making;
- abstention and disagreement analysis.

It is not automatically a subjective free-text benchmark. A separate human
protocol is required before judging generated explanations.

### 4.3 MedMCQA and MedQA-US

These datasets remain useful for:

- reproducible objective validation;
- retriever and prompt ablations;
- cross-dataset generalization;
- frozen test comparisons.

They must not be used as the sole evidence for subjective answer quality.

### 4.4 Required subjective evaluation set

Create a new, versioned subjective benchmark from an approved combination of:

- BioASQ ideal answers;
- PubMedQA rationales;
- clinician-authored or domain-expert-reviewed questions;
- questions spanning all 18 textbook domains;
- open-ended diagnostic, management, pharmacological, and educational queries.

Every record should contain:

```text
question_id
question
domain
reference_answer
supporting_chunk_ids
question_type
expected_abstention
clinician_difficulty
reviewer
annotation_version
```

Do not convert MCQ options into subjective questions without documenting the
transformation. Do not claim a subjective answer is gold-standard if it has only
an automatically generated reference.

### 4.5 In-Corpus Textbook Subjective Benchmark (Option A)

#### The Problem with Solely Relying on BioASQ
BioASQ questions and their ideal reference answers are authored exclusively
from PubMed journal articles. In contrast, MedRAGent retrieves strictly from
18 medical textbooks. This mismatch results in ~71% of BioASQ questions having
no direct supporting evidence in the 18 textbooks (e.g., 178 of 250 in the v1
review set default to expected abstention). While excellent for measuring
abstention / over-answering, relying solely on BioASQ limits true RAG quality
evaluation to a narrow slice of questions.

#### Option A Resolution
MedRAGent addresses this by generating questions **directly from the 18 textbooks**:
1. **Source-grounded synthesis:** `scripts/generate_textbook_subjective_questions.py`
   samples single and multi-chunk passages across all 18 textbooks to formulate
   free-form medical questions with guaranteed supporting chunk IDs.
2. **Quality judging:** A separate judge LLM verifies groundedness, standalone
   clarity, and retrieval testing value.
3. **Abstention benchmark pairing:** ~50 BioASQ questions with no textbook support
   are included as negative controls to evaluate abstention and out-of-corpus handling.
4. **Result:** An end-to-end benchmark (`datasets/subjective/textbook_subjective_benchmark_v1.jsonl`)
   spanning all 18 medical specialties where ground truth retrieval targets are
   rigorously known, coupled with an external abstention suite.

---

## 5. Evaluation protocol

## 5.1 Objective metrics

For MCQ tasks:

- exact option-letter accuracy;
- Wilson or bootstrap confidence intervals;
- failures and invalid-output rate;
- paired McNemar test;
- paired bootstrap difference;
- Holm-Bonferroni correction;
- accuracy by textbook, specialty, and scope;
- retrieval Recall@k, Precision@k, MRR, and nDCG.

Retrieval metrics use only human-reviewed non-empty evidence labels.

### 5.2 Subjective metrics

For free-form answers:

- answer correctness verified by domain reviewers;
- factual support against retrieved passages;
- citation precision and citation recall;
- unsupported-claim rate;
- completeness;
- specificity;
- coherence;
- relevance;
- harmful or unsafe recommendation rate;
- abstention coverage;
- correct-abstention rate;
- over-answering unsupported questions;
- agreement between reviewers;
- latency;
- number of retrieved chunks;
- prompt and completion token counts;
- retrieval cost and index cost.

Automated metrics may assist analysis, but they cannot replace human review.

### 5.3 Paired evaluation

All systems must answer the same questions with the same frozen prompt,
generator, context budget, and retrieval policy. Compare:

- LLM-only;
- BM25;
- dense;
- hybrid RRF;
- hierarchy-aware retrieval;
- bounded iterative retrieval;
- corrective retrieval;
- agent/controller routing.

Use paired significance testing and practical effect sizes. Do not compare
independent confidence intervals as though they were paired data.

### 5.4 Human review

At least a representative subset of:

- supported answers;
- unsupported answers;
- out-of-corpus answers;
- correct answers;
- incorrect answers;
- abstentions;
- high-risk medical domains.

The review guide will require reviewers to record:

- whether the answer is clinically correct;
- whether each cited passage supports the claim;
- whether unsupported claims are present;
- whether abstention was appropriate;
- whether the source is relevant and sufficient;
- confidence;
- reviewer identity;
- disagreement resolution.

No medical safety or factuality claim may be treated as established without
human review.

---

## 6. Baseline results currently completed

The existing development results are from MedQA-US `dev.jsonl` and the DSM-5
corpus only:

| System     |   Correct | Accuracy | Mean latency |
| ---------- | --------: | -------: | -----------: |
| LLM-only   | 670/1,272 |   52.67% |       268 ms |
| BM25 RAG   | 728/1,272 |   57.23% |     1,308 ms |
| Dense RAG  | 722/1,272 |   56.76% |       722 ms |
| Hybrid RAG | 730/1,272 |   57.39% |     1,030 ms |

These are development results. They are not frozen test results.

The current development data show:

- BM25: +4.56 percentage points over LLM-only;
- Dense: +4.09 percentage points over LLM-only;
- Hybrid: +4.72 percentage points over LLM-only.

The paired bootstrap and exact McNemar outputs are available in
`experiments/analysis/baseline_paired_statistics.json`.

The 128-row psychiatry candidate template and claim-level grounding templates
remain human-review artifacts, not completed expert annotations.

---

## 7. Remaining baseline work

The first deliverable is to finish the original MedQA/DSM-5 controlled baseline.
It must finish before the architecture is expanded.

### Step 1 — Complete manual scope and evidence review

**Status: waived for technical progress; expert review remains a future publication enhancement**

- Optional future work: complete `medqa_scope_manual_dev_template.jsonl`.
- Assign one of:
  - `psychiatry_relevant_supported`;
  - `psychiatry_relevant_unsupported`;
  - `out_of_corpus`.
- Add real DSM-5 chunk IDs for supported rows.
- Record reviewer identity and confidence.
- Freeze `manual_scope_v1` only after review and validation.

Lexical labels are development candidates, not expert scope decisions.

### Step 2 — Validate gold evidence

**Status: implemented and tested**

- Validate every question ID.
- Validate every supporting chunk ID.
- Require non-empty evidence for supported labels.
- Require empty evidence for unsupported and out-of-corpus labels.
- Check reviewer and annotation version metadata.
- Export coverage and label counts.

### Step 3 — Run the frozen development comparison

**Status: technical rerun completed; expert-reviewed rerun is optional future work**

- Freeze split IDs, corpus hash, model tags, temperature, top-k, RRF constant,
  prompt version, and context budget.
- Verify all four development files contain identical ordered question IDs.
- Recompute common reports with reviewed evidence.
- Report Recall@1, Recall@3, Precision@1, Precision@3, MRR, and nDCG.
- Compute exact McNemar tests and paired bootstrap intervals.
- Apply Holm-Bonferroni correction.
- Do not alter the test split.

### Step 4 — Conduct claim-level grounding review

**Status: tooling implemented; human claim review is waived for the current technical phase**

- Review the generated grounding templates.
- Record correctness, evidence support, unsupported claims, abstention, and
  citation relevance.
- Add inter-reviewer agreement and disagreement resolution.
- Separate supported, unsupported, and out-of-corpus findings.

### Baseline gate command

After reviewer files are complete, run:

```powershell
& .\.venv\Scripts\python.exe scripts\run_frozen_dev_baseline.py `
  --manual-annotations experiments\annotations\medqa_scope_manual_dev.jsonl `
  --grounding-review experiments\analysis\grounding_review_completed.jsonl `
  --verify-only
```

### Step 5 — Run the frozen MedQA-US test evaluation

**Status: technical frozen test run completed with deterministic lexical scope labels**

The human-review gate is waived for this phase. All test claims are technical
claims only and must not be described as clinically validated.

Commands:

```powershell
# Preflight only
& .\.venv\Scripts\python.exe scripts\run_frozen_test_baseline.py

# One immutable execution; no retries or re-tuning
& .\.venv\Scripts\python.exe scripts\run_frozen_test_baseline.py --execute
```

The `--execute` command refuses to reuse a non-empty test directory and stops on
the first failed system rather than retrying or changing the configuration.

- Run LLM-only, BM25, dense, and hybrid once.
- Use no tuning after the run starts.
- Preflight split ID hashes and model tags.
- Save raw JSONL outputs and reports.
- Mark test results as immutable.
- Do not infer from test data into a changed system.

### Step 6 — Baseline publication package

**Status: technical package complete; expert-review fields remain outside the package**

**Status: package builder implemented; population occurs after the frozen test run**

After the immutable test run, build the package with:

```powershell
& .\.venv\Scripts\python.exe scripts\build_publication_package.py
```

The builder refuses to create publication metadata before all frozen test
outputs exist and marked `execute_status: completed`.

Prepare:

- source code and exact commit;
- environment and dependency specification;
- model tags and generation settings;
- corpus and benchmark manifests;
- prompt versions;
- raw output JSONL;
- aggregate reports;
- review guide;
- statistical analysis;
- limitations and ethical-use statement.

---

## 8. 18-textbook subjective system

This phase replaces the narrow DSM-5 baseline with the complete 18-textbook
corpus.

### Step 7 — Build and validate the complete 18-textbook index

**Status: complete for persisted BM25; dense/hybrid builders implemented and ready for a one-time full embedding build**

- Load all 18 chunk JSONL files.
- Preserve source textbook metadata.
- Check chunk IDs, hashes, duplicates, empty text, and malformed fields.
- Create the corpus manifest.
- Build BM25 and dense indexes.
- Generate reproducible hybrid rankings.
- Record index size, build time, memory use, and reload time.

No textbook is to be silently omitted.

### Step 8 — Implement source-aware retrieval

**Status: implemented in the full-corpus retrieval runner**

Retrieved output must include:

- chunk ID;
- textbook name;
- title;
- source file;
- chunk text;
- rank;
- retrieval method;
- raw score;
- fused rank, where applicable.

### Step 9 — Implement the subjective response contract

**Status: implemented for source-aware structured responses**

The system must distinguish:

- an answer grounded in retrieved passages;
- an answer that combines several passages;
- a partial answer;
- an unsupported answer;
- an answer requiring abstention.

The generator must return source identifiers, but structural citation validity
must not be reported as medical factuality.

The implementation is in `src/subjective.py` and
`scripts/run_subjective_qa.py`. The response contract includes:

- `answer`
- `evidence_status`
- `citations`
- `limitations`
- `abstention_reason`

### Step 10 — Add insufficient-evidence behavior

**Status: implemented for BM25, dense, and hybrid retrieval**

The default documented policy is:

- BM25: abstain when the top adequate passage has a score at or below zero;
- dense: abstain below cosine similarity `0.25`;
- hybrid RRF: abstain below fused score `0.015`;
- every method: require at least 100 characters of passage text;
- return structured abstention instead of inventing citations.

The policy is configurable in `scripts/run_subjective_qa.py`.

Implement a separate abstention experiment:

- define a retrieval-confidence or evidence-quality rule;
- distinguish invalid output from valid abstention;
- report coverage and abstention rate;
- report correct abstention;
- report over-answering unsupported questions;
- never count citation presence as evidence support.

This behavior experiment must not rewrite frozen baseline outputs.

### Step 11 — Add bounded advanced retrieval

**Status: hierarchy prototype exists for DSM-5; iterative retrieval is fully
implemented in `iterative_retrival/` with BM25/dense/hybrid retrieval, LLM
query refinement, RRF fusion, up to 3 rounds, scope refusal, and audit
traces (evaluated on 78 generated + 91 MedQA psychiatry Qs, DSM-5 only);
corrective retrieval and controller routing are not yet implemented; no
advanced workflow has been evaluated on the frozen 18-textbook subjective
protocol.**

Integrate and test these as separate systems:

1. hierarchy-aware retrieval;
2. one-round bounded iterative retrieval;
3. corrective retrieval with at most one correction;
4. agent/controller routing.

Do not implement unbounded loops.

Each system must log:

- initial query;
- follow-up query;
- retrieved chunks;
- scores;
- stop reason;
- generation calls;
- tokens;
- latency;
- cost;
- final citation set.

The context budget and retrieval budget must be comparable with the baselines.

---

## 9. Subjective benchmark and dashboard

### Step 12 — Prepare the final subjective benchmark

**Status: BioASQ 13b export and a 250-question model-assisted textbook
mapping are complete for engineering evaluation; clinical/expert review is an
optional publication enhancement, not a project blocker.**

- BioASQ 13b is available locally; its 1,283 summary questions were exported
  to `datasets/subjective/bioasq13b_summary_unreviewed.jsonl`.
- Each row has a subjective question, ideal/reference answer, and external
  PubMed evidence for provenance, but no approved textbook evidence labels.
- Use `scripts/build_bioasq_subjective_benchmark.py` to rebuild the export,
  `scripts/attach_textbook_candidates.py` for review candidates, and
  `scripts/validate_subjective_benchmark.py` to validate the final labels.
- The v1 technical subset is
  `datasets/subjective/bioasq13b_summary_model_reviewed_v1.jsonl`; it uses
  `model_reviewed` status, low/medium confidence, explicit evidence mapping,
  and expected-abstention labels. It is schema-validated against all 18 books.
- Freeze this v1 set for development. Human review may later upgrade selected
  records to `expert_reviewed`, but is not required to run the final system or
  technical ablations.

### Step 13 — Run human-annotated evaluation

- Run objective and subjective evaluation on identical questions.
- Conduct blinded or dual-review claim-level assessment.
- Report Cohen's kappa or another appropriate agreement statistic.
- Resolve disagreements and preserve original reviews.
- Produce supported, partial, unsupported, and abstained subsets.

### Step 14 — Build the dashboard

The dashboard must allow a user to:

- enter a subjective question;
- select all 18 textbooks or a textbook subset;
- select retrieval strategy;
- set a retrieval budget;
- view the final answer;
- view citations and source passages;
- see evidence status;
- view abstention reason;
- inspect retrieval latency and token usage;
- export the response and source list.

The dashboard should display a clear non-clinical-use notice:

> This system supports medical information research. It does not replace
> licensed medical judgment or clinical decision-making.

---

## 10. Quantitative comparison with published medical RAG research

### Step 15 — Create a verified literature table

Build `experiments/literature/comparison_table.jsonl` with one entry per study:

```text
study_id
paper_title
authors
year
venue
doi_or_url
dataset
evidence_corpus
retrieval_method
generation_model
split
metric_name
reported_value
standard_error_or_ci
sample_size
comparison_category
notes
source_quote_or_table_location
```

Every metric must come from a verifiable paper table, supplementary material,
or officially released reproduction. No result may be copied from another
summary if the original could not be checked.

### Step 16 — Match comparison categories

Published systems may only be compared in categories with compatible evidence:

- exact same dataset and split;
- same or explained evaluation protocol;
- same evidence corpus or clearly documented corpus;
- same answer format;
- same metric definition;
- comparable subject or domain subset.

Do not combine incompatible numbers into a single ranking.

### Step 17 — Add MedRAG paper comparisons

Search and verify the published results of relevant:

- MedQA and MedMCQA papers;
- PubMedQA and BioASQ research;
- medical RAG studies;
- dense and BM25 medical retrieval studies;
- hybrid RAG studies;
- citation-grounded medical QA studies;
- agentic or corrective RAG medical studies;
- textbook-based medical QA systems.

Record whether each study is:

- directly comparable;
- partially comparable;
- non-comparable;
- a different task or corpus.

### Step 18 — Produce the final comparison table

Separate tables by task:

1. MedQA-US MCQ;
2. MedMCQA;
3. PubMedQA;
4. BioASQ;
5. the final 18-textbook subjective benchmark.

For MedRAGent, report:

- median or mean result;
- confidence interval or paired difference;
- sample size;
- retrieval method;
- model;
- corpus;
- split;
- human-review rate;
- latency;
- evidence citation metric.

Clearly label the comparison as:

- direct reproduced baseline;
- reported by the original paper;
- independent reproduction;
- non-exact contextual comparison.

---

## 11. Advanced research extensions

Only after the 18-textbook subjective baseline is frozen may these extensions
be added:

1. Hierarchy-aware retrieval.
2. Bounded iterative retrieval.
3. Corrective and verified retrieval.
4. Agent/controller routing.
5. Retrieval reranking and query expansion.
6. Chunking ablations.
7. Persistent storage and indexing comparisons.
8. External-dataset generalization.

These are separate experiments. They must not be presented as improvements to
the frozen baseline unless they use the frozen questions and protocol.

---

## 12. Project completion checklist

A complete project should include:

- reproducible baseline code;
- 18-textbook corpus manifest;
- subjective benchmark and review guide;
- frozen MedQA-US test evaluation;
- frozen subjective 18-textbook evaluation;
- paired statistical tests;
- human grounding review;
- abstention study;
- dashboard;
- source-citation interface;
- verified published-literature comparison;
- source code and exact commit;
- reproducible commands;
- dependency and environment versions;
- raw outputs and aggregate reports;
- limitations and ethical-use statement.

---

## 13. Publication claims allowed

After the final evaluation:

- state which systems differ on the exact tested datasets;
- state whether the 18-textbook system outperforms the baseline;
- report subjective evaluation with uncertainty and reviewer agreement;
- report published comparison only for compatible categories;
- state that citation validity is structural or reviewed, as appropriate;
- state that human review is limited to the reported sample;
- state that the system is not clinically validated or deployable.

Do not claim:

- universal medical accuracy;
- superiority over a paper without an exact compatible evaluation;
- clinical safety;
- factuality solely from automatic citation checks;
- hierarchy support without trustworthy hierarchy metadata;
- subjective quality from MCQ data alone.

---

## 14. Final execution plan — clear actionable steps

The frozen MCQ baseline is finished. Do not re-tune it using its frozen test
split. All remaining work is directed toward the final 18-textbook,
subjective-answering system. The five final goals are:

1. **Show why RAG improves** over an LLM-only baseline.
2. **Iterative and hierarchical retrieval** — multiple retrieval rounds.
3. **Chunking, storage, and indexing optimization** — ablations and comparisons.
4. **Corrective/agentic workflow** — bounded self-correction and routing.
5. **Retrieval fusion and UI** — fusion tuning and a source-inspection dashboard.

Plus: subjective-ready RAG, and chunking/storage/retrieval explorations and
optimization.

The human-expert-review blocker is **removed**: the model-reviewed v1
benchmark (`bioasq13b_summary_model_reviewed_v1.jsonl`, 250 Qs) is available for
engineering experiments, and Option A provides an in-corpus textbook-grounded
benchmark with guaranteed gold chunk IDs. Expert clinical review is an optional
publication enhancement, not a project gate.

---

### Phase A — Subjective benchmark & corpus-grounded test sets

#### Phase A.1 — BioASQ Summary Transition Dataset ✅ DONE
- Exported 1,283 BioASQ summary questions to `datasets/subjective/bioasq13b_summary_unreviewed.jsonl`.
- Attached candidate chunks across 18 textbooks to `bioasq13b_summary_textbook_candidates.jsonl`.
- Built and validated deterministic model-reviewed subset in `bioasq13b_summary_model_reviewed_v1.jsonl` (72 partial support, 178 expected abstention).
- *Role:* Primary benchmark for abstention verification and out-of-corpus stress testing.

#### Phase A.2 — 18-Textbook Subjective Benchmark (Option A) 🚀 IN PROGRESS
- **Generator script:** `scripts/generate_textbook_subjective_questions.py` created.
- **Methodology:**
  1. Proportionally samples single and multi-chunk passage groups across all 18 textbooks.
  2. Synthesizes questions across 6 medical categories (factual, explanation, comparison, clinical reasoning, integration, safety).
  3. Uses a judge LLM to filter by groundedness, standalone clarity, and retrieval value.
  4. Generates ~20-25 questions per textbook (~400 questions total).
  5. Injects ~50 BioASQ out-of-corpus questions with `expected_abstention: true` as negative controls.
- **Output:** `datasets/subjective/textbook_subjective_benchmark_v1.jsonl` (~450 questions total).
- *Role:* Primary benchmark for factual retrieval metrics (Recall@k, MRR, nDCG) and subjective answer generation where textbook evidence is guaranteed to exist.

| Deliverable | File | Status |
|---|---|---|
| BioASQ 13b summary export (1,283 Qs) | `datasets/subjective/bioasq13b_summary_unreviewed.jsonl` | ✅ |
| BioASQ candidate attachment (1,283 Qs) | `datasets/subjective/bioasq13b_summary_textbook_candidates.jsonl` | ✅ |
| BioASQ model-reviewed subset (250 Qs) | `datasets/subjective/bioasq13b_summary_model_reviewed_v1.jsonl` | ✅ |
| 18-Textbook generator script | `scripts/generate_textbook_subjective_questions.py` | ✅ |
| 18-Textbook subjective benchmark (~450 Qs) | `datasets/subjective/textbook_subjective_benchmark_v1.jsonl` | ⏳ Ready to run |

---

### Phase B — Integrate iterative retrieval & build full-corpus subjective baseline

**Status: IN PROGRESS — this is the immediate next priority.**

**Goal:** unify the proven `iterative_retrival/` pipeline with the main
`src/` package, then run all system variants on the corpus-grounded
subjective benchmark (`textbook_subjective_benchmark_v1.jsonl`) and the
BioASQ abstention subset over all 18 textbooks.

#### Step B.1 — Integrate iterative retrieval into main package
- [ ] Refactor the reusable components from `iterative_retrival/medx_retrieval.py`
  into `src/`:
  - `BM25Index`, `ChromaRetriever`, `HybridRetriever` → adapt to work with
    the existing `src/bm25.py`, `src/dense.py`, `src/hybrid.py` chunk schema
    (which uses the `datasets/MedRAG/chunk/` JSONL with `id`, `content`/`contents`,
    `title` fields from all 18 books, not just DSM-5).
  - `iterative_retrieve()` function → new `src/iterative.py` module that
    accepts the existing retriever interfaces.
  - `check_scope()` → new `src/scope_check.py` or integrate with existing
    `src/scope.py`.
  - `build_context()`, `grounded_answer()`, response-to-dict serialization.
  - Multi-provider LLM client (`medx_llm.py`) → `src/llm.py` or reuse with
    an import.
- [ ] Ensure all integrated code uses the same chunk ID format, metadata
  schema, and textbook naming as the main project.
- [ ] Add unit tests for the integrated iterative retriever.
- [ ] Keep `iterative_retrival/` as-is for reference; mark it as the DSM-5
  prototype in a README.

#### Step B.2 — Build and persist the full 18-textbook dense index
- [ ] Choose embedding model: local MiniLM-L6 (free, ~384-dim) or Gemini
  embeddings (768-dim, quota-limited). Document the choice.
- [ ] Embed all 125,847 chunks. Record: model name, dimensions, build time,
  index file size, peak memory, reload time.
- [ ] Persist the index under `indexes/full_textbooks_dense/`.
- [ ] Verify index integrity: reload and query a known chunk.

#### Step B.3 — Run the frozen subjective baseline (4 systems)
- [ ] On the 250 frozen questions, run:
  1. **LLM-only** — no retrieval, just the generator model.
  2. **BM25 RAG** — persisted full-corpus BM25 index, top-k=5.
  3. **Dense RAG** — full-corpus dense index, top-k=5.
  4. **Hybrid RRF** — BM25 + dense, RRF fusion, top-k=5.
- [ ] Use identical: generator model, prompt template, context budget (9000
  chars), abstention thresholds, and temperature=0.2.
- [ ] Save per-question JSONL with: answer text, retrieved chunk IDs, raw
  scores, citations, evidence status, latency, tokens.
- [ ] Evaluate: answer correctness vs. BioASQ reference answer, citation
  precision/recall, claim support, abstention coverage, correct abstention,
  over-answering rate.
- [ ] Report retrieval Recall@1/@3/@5, Precision@k, MRR, nDCG.
- [ ] Compute paired McNemar and bootstrap intervals; apply Holm-Bonferroni.

**Completion gate:** 4 frozen subjective baseline output files in
`experiments/results/subjective_baseline/`.

---

### Phase C — Chunking, storage, indexing, and metadata optimization

**Status: NOT STARTED**

**Goal:** determine whether better evidence representation improves the
frozen subjective baseline.

#### Step C.1 — Define and build chunking variants
- [ ] **Variant 1 (current):** existing MedRAG chunks (125,847).
- [ ] **Variant 2:** fixed 512-token windows with 64-token overlap.
- [ ] **Variant 3:** sentence-boundary-aware chunks (~300–500 tokens).
- [ ] **Variant 4:** parent-child chunks (large parent = section, small
  child = paragraph; retrieve child, expand to parent for context).
- [ ] For each: preserve source textbook, stable chunk IDs, source offsets.

#### Step C.2 — Measure and compare each variant
- [ ] Metrics per variant: chunk count, duplicate rate, avg/p90 token length,
  disk size, BM25 build time, dense embed time, dense index size, reload
  time, query latency (p50/p95), peak memory.
- [ ] Run each variant through the frozen subjective baseline (BM25 + dense +
  hybrid) and measure Recall@k, citation precision, answer quality.
- [ ] Record results in a comparison table.

#### Step C.3 — Storage and index backend comparison
- [ ] Compare: in-memory BM25 + FAISS, persisted BM25 + FAISS, ChromaDB
  (as used in `iterative_retrival/`).
- [ ] Only add a metadata-filter-capable vector store (e.g., Chroma with
  textbook metadata filters) if it materially improves retrieval or
  filtering.
- [ ] Measure: query latency, build time, disk usage, memory, concurrent
  query throughput.

#### Step C.4 — Metadata enrichment
- [ ] Add defensible metadata to chunks: textbook name, edition, specialty,
  chunk byte offsets, corpus version.
- [ ] Only add chapter/section metadata if it can be reliably extracted (do
  not infer hierarchy from title fields alone).
- [ ] Test metadata-filtered retrieval: e.g., retrieve only from Internal
  Medicine + Pharmacology for a drug question.

#### Step C.5 — Select frozen optimized configuration
- [ ] Choose the best chunk/index configuration using development data only.
- [ ] Freeze it and evaluate once on the subjective test set.

**Completion gate:** documented chunk/index ablation table in
`experiments/analysis/chunking_ablation.json` and a selected versioned
configuration.

---

### Phase D — Retrieval exploration and advanced workflows

**Status: NOT STARTED (iterative prototype exists, needs integration)**

**Goal:** evaluate five retrieval directions as bounded, auditable systems,
all compared against the frozen Phase B baseline.

#### Step D.1 — Fusion exploration
- [ ] Compare: BM25-only, dense-only, RRF (k=60), weighted RRF (vary
  BM25:dense ratio), and optional cross-encoder reranker.
- [ ] Tune weights and top-k on development split only.
- [ ] Log both source rankings and the fused ranking for every question.
- [ ] Record quality uplift vs. baseline, latency overhead.

#### Step D.2 — Iterative retrieval (integrated)
- [ ] Using the integrated `iterative_retrieve()` from Step B.1:
  - Run 1-round (baseline) and 3-round (ablation) on the frozen questions.
  - Use LLM query refinement (planner) with keyword fallback.
  - Log: initial query, every follow-up query, refiner decision, stop reason,
    per-round evidence sets, fused final sources.
- [ ] Compare: Recall@k, citation quality, answer correctness, latency,
  token cost, number of LLM calls.
- [ ] Determine if the gains observed on DSM-5 (recall +6.1%) transfer to
  18-textbook retrieval.

#### Step D.3 — Hierarchical retrieval
- [ ] First build trustworthy section metadata for at least 3 textbooks
  (DSM-5, Harrison's, Robbins) — either from reliable TOC extraction or
  manual annotation.
- [ ] Implement hierarchy-aware retrieval: collapsed-tree and traversal modes.
- [ ] Compare against flat retrieval using the same context and retrieval
  budgets.

#### Step D.4 — Corrective retrieval
- [ ] After the first retrieval, run a bounded evidence-quality check:
  - If the top-k evidence scores are below the abstention threshold, OR
  - If the evidence doesn't cover the key entities in the question,
  - Then: reformulate the query and retrieve once more (1 retry max).
- [ ] Log: original evidence quality, correction trigger reason, corrected
  query, corrected evidence quality, whether correction improved the answer.

#### Step D.5 — Agent/controller routing
- [ ] Implement a small, deterministic controller that:
  - Classifies the question (factual/complex/multi-part/out-of-scope).
  - Routes to: flat retrieval, iterative retrieval, hierarchical retrieval,
    or direct abstention.
  - Has a hard call budget (max 3 retrieval calls + 2 LLM calls).
  - Exposes its routing decision in the audit trace.
- [ ] No unbounded autonomous loops.

#### Step D.6 — Comparative evaluation
- [ ] Evaluate all variants against the same frozen questions, same generator,
  same review protocol, same context budget.
- [ ] Report: quality, grounding, abstention, latency, token/cost,
  statistical significance for each workflow.
- [ ] Retain a workflow only if its gain justifies its added cost and
  complexity.

**Completion gate:** ablation report in `experiments/analysis/retrieval_ablation.json`.

---

### Phase E — External validation, dashboard, and final package

**Status: NOT STARTED**

**Goal:** show the system works beyond one internal annotation set, build
the UI, and prepare the final deliverable.

#### Step E.1 — External validation
- [ ] Evaluate on a held-out BioASQ subset (separate from the 250 development
  questions) against ideal answers.
- [ ] Download and evaluate on PubMedQA (1,000 expert-labeled) for
  evidence/rationale and abstention stress testing.
- [ ] Optionally evaluate on MEDIQA-AnS for patient-facing
  clarity/completeness.
- [ ] Keep scores in separate tables (different corpora and answer formats).

#### Step E.2 — Build the dashboard
- [ ] Framework: Streamlit or Gradio (choose whichever gives faster iteration).
- [ ] Features:
  - Text input for a subjective question.
  - Textbook selector (all 18 or a subset).
  - Retrieval strategy selector (BM25 / dense / hybrid / iterative).
  - Retrieval budget slider (top-k, max rounds).
  - Answer display with:
    - Generated answer text.
    - Evidence status badge (supported / partial / unsupported / abstention).
    - Citation list with clickable passage excerpts.
    - Source textbook and chunk ID.
    - Retrieval latency and token count.
    - Limitations section.
  - Non-clinical-use notice banner.
  - Export button (JSON audit record: query, config version, chunks, answer,
    citations, timestamp).

#### Step E.3 — Literature comparison table
- [ ] Build `experiments/literature/comparison_table.jsonl`:
  - One entry per published study with: paper title, authors, year, venue,
    dataset, retrieval method, model, metric, reported value, CI, sample size.
  - Only compare systems on compatible datasets, splits, and metrics.
  - Label non-identical comparisons as contextual.
- [ ] Include comparisons with: MedRAG, MedQA leaderboard entries, BioASQ
  challenge results, medical RAG studies.

#### Step E.4 — Final reproducibility package
- [ ] Source code and exact commit hash.
- [ ] Environment specification (`requirements.txt`, Python version).
- [ ] Model tags, generation settings, prompt versions.
- [ ] Corpus and benchmark manifests.
- [ ] Raw output JSONL for every experiment.
- [ ] Aggregate reports and statistical analyses.
- [ ] Limitations and ethical-use statement.
- [ ] Dashboard demo instructions.

**Completion gate:** auditable dashboard demo, external validation report,
reproducibility package, limitations statement, and final presentation of
the five objectives.

---

### Recommended immediate next actions (priority order)

1. **Step B.1** — Integrate iterative retrieval into `src/`. This unblocks
   everything else.
2. **Step B.2** — Build the full 18-textbook dense index.
3. **Step B.3** — Run the frozen subjective baseline (4 systems).
4. **Step D.2** — Run iterative retrieval on the subjective benchmark.
5. **Step C.1–C.2** — Chunking ablations.
6. **Step D.4–D.5** — Corrective and controller routing.
7. **Step E.2** — Dashboard.

No step depends on external human review. All experiment labels are
model-assisted and clearly documented as such.
