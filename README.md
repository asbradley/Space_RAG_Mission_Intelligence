# Space RAG — Mission Intelligence

A retrieval-augmented generation system over NASA mission reports, built to
answer questions about them with passage-level citations — and, unusually
for a project this size, with a real evaluation harness that measures
whether the retrieval pipeline actually works.

Everything runs locally. No API keys, no hosted inference, no vector-database
service: Postgres with `pgvector` is the vector store, `sentence-transformers`
provides embeddings and reranking, and Ollama runs the LLM.

```
$ curl -s localhost:8001/ask -d '{"question":"Which ship recovered the Apollo 11 crew?"}'

{
  "answer": "The U.S.S. Hornet [2].",
  "sources": [
    {"n": 1, "title": "Apollo 11 lunar landing mission - Press kit", "cited": false, ...},
    {"n": 2, "title": "Apollo 11 mission report",                    "cited": true,  ...}
  ]
}
```

## Pipeline

```
NTRS API → PDF → text extraction → chunking → embeddings → Postgres/pgvector
                                                                    │
question ──────────────────────────────────────────────────────────┤
                                                                    ▼
              ┌──────────────────┐   ┌─────────────────┐   ┌────────────────┐
              │ vector search    │──▶│ Reciprocal Rank │──▶│ cross-encoder  │
              │ keyword search   │   │ Fusion          │   │ rerank         │
              └──────────────────┘   └─────────────────┘   └────────────────┘
                                                                    │
                                          per-document cap ─────────┤
                                                                    ▼
                                            prompt → Ollama → answer + citations
```

Retrieval runs in three selectable stages, so each can be measured against
the others:

| mode | what it does |
|---|---|
| `vector` | pgvector cosine similarity only |
| `hybrid` | + Postgres full-text search, merged by Reciprocal Rank Fusion |
| `reranked` | + cross-encoder scoring over a wider candidate pool *(default)* |

Results are capped at 2 chunks per document so a single source can't fill
every slot, and the answer's inline `[n]` markers are parsed back to mark
which excerpts were actually used.

## Stack

| layer | choice | why |
|---|---|---|
| Vector store | PostgreSQL 16 + `pgvector` | One database for relational data and vectors; no separate service to run |
| Keyword search | Postgres full-text (`tsvector`, GIN) | Already there; no Elasticsearch dependency |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Local, ~80 MB, fast on CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Ships with `sentence-transformers` — no new dependency |
| LLM | `llama3.2:3b` via Ollama | Local and free |
| API | FastAPI + SQLAlchemy | |
| Frontend | Vite + React + TypeScript | |

## Setup

```bash
# 1. Start Postgres (with pgvector) via Docker.
#    Runs on host port 5433, not 5432 — adjust in docker-compose.yml /
#    backend/.env if 5433 is also taken on your machine.
docker compose up -d

# 2. Python environment
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Create the schema (pgvector extension, tables, GIN index)
python -m scripts.init_db

# 4. Ingest documents from the NASA Technical Reports Server
python -m app.ingestion.ingest "apollo 11 mission report" --limit 10

# 5. Pull the LLM (requires Ollama: https://ollama.com)
ollama pull llama3.2:3b

# 6. Run the API.
#    Port 8001, not 8000 — pick your own if 8001 is also taken locally
#    (check with `lsof -nP -iTCP:8001 -sTCP:LISTEN`).
uvicorn app.main:app --reload --port 8001
```

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was the primary objective of the Apollo 11 mission?"}'
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (backend must be running on `:8001` — CORS is
already configured for the Vite dev server in `backend/app/main.py`).

## API

| endpoint | purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /documents` | list ingested documents |
| `POST /ask` | answer a question; returns the answer plus the numbered excerpts behind it, each flagged `cited` or not |

## Evaluation

Retrieval quality is measured rather than asserted. `backend/eval/eval_set.json`
holds 15 hand-curated questions written by reading actual chunks out of the
database, each anchored to ground truth by `source_id` plus a distinctive text
snippet — never by chunk id, which is DB-assigned and would shift on
re-ingestion.

```bash
python -m scripts.evaluate --compare            # all three modes, side by side
python -m scripts.evaluate --retrieval-only     # skip generation (fast)
python -m scripts.evaluate --verify-anchors     # re-check ground truth resolves
```

### Results

15 questions, 21 anchors, 1411 chunks across 3 Apollo 11 documents,
`top_k=5`, generation pinned to temperature 0 for reproducibility.

| metric | vector | hybrid | reranked |
|---|---|---|---|
| hit@5 | 0.600 | 0.600 | 0.600 |
| MRR | 0.377 | 0.402 | **0.461** |
| recall@5 | 0.544 | 0.544 | 0.544 |
| facts_present | **0.678** | 0.622 | 0.589 |
| answered_with_citation | 0.533 | 0.400 | **0.600** |
| median latency | 25.2 ms | 22.5 ms | 99.3 ms |

All three modes tie exactly on hit@5 and recall@5; reranking finds the same
passages and orders them better. `reranked` is the default on that basis.

### What the evaluation found

The harness was built to test claims made in earlier phases from single
hand-picked examples. It did not confirm them:

- **Reranking is a modest ordering improvement, not the win it was credited
  with.** It leads on MRR by about one question's worth and costs 77 ms.
- **Hybrid retrieval is not distinguishable from vector-only on this set** —
  though every question here is a specific factoid lookup, which is keyword
  search at its best. Broad questions are unrepresented.
- **The bottleneck is upstream, in PDF extraction.** One of the three
  documents extracts with words run together (`"waslaunchedfrom
  KennedySpaceCenter"`), which defeats *both* halves of hybrid search at
  once: Postgres can't tokenize it and its embedding is poor. It also causes
  the cross-encoder to systematically demote otherwise-correct passages,
  because it reads passage text directly while rank fusion does not.

Two methodological limits are worth stating plainly, because they bound what
the numbers above can support:

- **Answer metrics were originally noise.** Generation ran at Ollama's default
  temperature with nothing pinned, and three *identical* runs produced
  citation rates of 0.800, 0.733 and 0.600 — a 0.20 spread. The harness now
  pins temperature 0 and repeat runs are identical.
- **The set is small.** At n=15 one question is worth 0.067, so differences
  under roughly 0.15 are unresolved. The retrieval ties are exact and real;
  most answer-metric orderings are not established.

## Project structure

```
backend/
  app/
    ingestion/     NTRS client, PDF extraction, chunking, storage
    embeddings.py  local sentence-transformers embeddings
    reranker.py    local cross-encoder
    rag.py         retrieval, prompt construction, citation parsing
    main.py        FastAPI endpoints
  eval/            evaluation set + reference runs
  scripts/         init_db, backfill_embeddings, evaluate
frontend/          Vite + React question UI
```

## Known limitations

- **PDF extraction** damages one of the three documents (run-together text).
  The highest-value fix; `pypdf` is the likely cause.
- **Chunking** slices on fixed 1000-character boundaries, so chunks begin and
  end mid-word. The API trims fragments for display, but the stored chunks —
  which are what get embedded — are still cut arbitrarily.
- **Evaluation set** is 15 questions, all specific factoid lookups, each
  anchored to one location per fact even where the corpus repeats it.
- **No automated tests.** Every phase was verified by running it.
- **Corpus is 3 documents.** Nothing here has been exercised at scale.
