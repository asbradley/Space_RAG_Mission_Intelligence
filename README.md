# Space_RAG_Mission_Intelligence

A RAG system over NASA mission reports and technical documentation.

Stack: React → FastAPI (Python) → PostgreSQL + pgvector → LLM.

## Phase 1: NASA document ingestion

Pulls documents from the [NASA Technical Reports Server (NTRS)](https://ntrs.nasa.gov),
stores raw PDFs on disk, extracts text, chunks it, and stores document
metadata + chunks in Postgres. Embeddings and retrieval come in Phase 2.

### Setup

```bash
# 1. Start Postgres (with pgvector) via Docker
docker compose up -d

# 2. Set up the Python environment
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Create the schema
python -m scripts.init_db

# 4. Ingest some documents
python -m app.ingestion.ingest "apollo 11 mission report" --limit 10

# 5. Run the API and check what landed
uvicorn app.main:app --reload
curl http://localhost:8000/documents
```

### Frontend

Minimal Vite + React + TypeScript app that lists ingested documents from
`GET /documents`. Lives in `frontend/`, sibling to `backend/`.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (backend must be running on `:8000` — CORS is
already configured for the Vite dev server in `backend/app/main.py`).

### Notes

- `app/ingestion/ntrs_client.py` maps NTRS's API response into our schema
  defensively, but the exact field names weren't verified against a live
  response — run the module directly (`python -m app.ingestion.ntrs_client
  "apollo 11"`) to dump a raw result and confirm the mapping before a real
  ingestion run.
- Raw PDFs are stored on local disk (`data/raw/`) for now; swap
  `app/ingestion/storage.py` for an S3-backed implementation later without
  touching the rest of the ingestion pipeline.
- The `chunks.embedding` column exists already (pgvector) but stays `NULL`
  until Phase 2 wires up an embedding model.
