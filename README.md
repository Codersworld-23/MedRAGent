<<<<<<< Updated upstream
# MedRAGent
=======
# MedX.ai

Iterative, LLM-refined retrieval and RAG testing over the included
`Psichiatry_DSM-5.jsonl` corpus.

## Flow

```text
question
  -> round 1: hybrid search with the original question
       (local MiniLM or Gemini embeddings in Chroma + BM25, fused with reciprocal rank fusion)
  -> the planner LLM reads the fused evidence and returns
       { sufficient, missing_information, next_queries }
  -> if not sufficient: search each new query, fuse with all earlier results
       (original question weighted highest so later rounds cannot drift the ranking)
  -> repeat until sufficient, no new queries, or the --rounds limit
  -> sources: each query's best chunk + top fused chunks -> answer LLM with inline source ids
```

Embeddings default to Chroma's bundled ONNX `all-MiniLM-L6-v2` model: local, free,
no quota. `--embedder gemini` uses `gemini-embedding-001`, but the free tier allows
only 1,000 embedded chunks per day (this corpus has 4,057); ingestion resumes where
it stopped, so a free-tier Gemini index can be built over several days.

Without any LLM key the pipeline still runs: IDF-weighted keyword expansion
replaces LLM query planning, and an extractive answer replaces LLM generation.

## LLM Setup (free tiers)

All LLM calls go through one OpenAI-compatible client (`medx_llm.py`), so the provider is
configuration, not code. Install dependencies and add whichever keys you have to `.env`
(see `.env.example`):

```powershell
pip install -r requirements.txt
```

| Provider | Key | Default role |
|---|---|---|
| NVIDIA build.nvidia.com | `NVIDIA_API_KEY` | planner + answer: `nvidia/nemotron-3-super-120b-a12b`; judge: `z-ai/glm-5.3-flash` |
| Google Gemini | `GEMINI_API_KEY` | automatic fallback when NVIDIA fails |
| Groq / OpenRouter | `GROQ_API_KEY` / `OPENROUTER_API_KEY` | optional |
| Local Ollama | `OLLAMA_BASE_URL` | optional, unlimited, slow on CPU |

Several models in NVIDIA's catalog return 404 on a free account (e.g. Mistral Large, Gemma 3,
Kimi K2.6); the defaults above were benchmarked and respond in about 6-25 seconds.

Override any role with `MEDX_PLANNER_LLM`, `MEDX_ANSWER_LLM`, `MEDX_JUDGE_LLM` in `.env`, or
`--planner-llm` / `--answer-llm` on the command line, as `provider` or `provider:model`.
The judge uses a different model family than the answer model so evaluation is not a model
grading itself. If a call fails outright, the next provider with a key is tried and the output
records which model answered; set `MEDX_FALLBACK=0` to disable that during evaluations.

Check what is configured, and that each role responds:

```powershell
python medx_llm.py ping
```

Every LLM reply is cached in `.llm_cache/`, so rerunning a question or an evaluation costs no
quota and gives identical output. Requests are paced per provider to stay under free-tier
per-minute limits, and a daily-quota error fails fast instead of waiting for hours.

## Run

First embed the JSONL chunks into Chroma (a few minutes on CPU; resumable if
interrupted, and rebuilt automatically if the corpus text changes):

```powershell
python medx_retrieval.py ingest
```

Ask a question:

```powershell
python medx_retrieval.py ask "How are panic attacks considered with anxiety disorders?" --rounds 3
```

Useful flags:

| Flag | Choices | Default |
|---|---|---|
| `--retriever` | `hybrid`, `chroma`, `bm25` | `hybrid` |
| `--embedder` | `auto`, `local`, `gemini`, `hash` | `auto` (= `local`) |
| `--refiner` | `auto`, `llm`, `keywords` | `auto` |
| `--generator` | `auto`, `llm`, `extractive` | `auto` |
| `--planner-llm`, `--answer-llm` | `provider` or `provider:model` | from `.env` |
| `--rounds` | max retrieval rounds | `3` |
| `--top-k` | chunks per query | `5` |
| `--sources` | fused chunks sent to the generator | `8` |

Fully offline run: `--embedder hash --refiner keywords --generator extractive`.

## Notebook API

```python
import medx_retrieval

retriever = medx_retrieval.build_retriever(retriever_name="hybrid", embedder="local")

response = medx_retrieval.answer_question(
    "What criteria are associated with a major depressive episode?",
    retriever,
    rounds=3,
)

print(response.answer)
for round_ in response.rounds:
    print(round_.round_number, round_.refiner, round_.queries)
for source in response.sources:
    print(medx_retrieval.format_result(source))
```

## Batch Testing

Put questions in JSONL:

```json
{"question": "How are panic attacks considered with anxiety disorders?", "expected_doc_ids": ["Psichiatry_DSM-5_1084"]}
```

Then run:

```powershell
python medx_retrieval.py test --questions test_questions.jsonl --output rag_test_results.jsonl --retriever hybrid --generator llm
```

The output file contains the answer, sources, each round's queries, retrieval
trace, and whether an expected source was retrieved.

Use `--generator extractive` when you want to test retrieval without calling the
LLM API.
>>>>>>> Stashed changes
