# Space_RAG_Mission_Intelligence

A RAG system over NASA mission reports and technical documentation.

Stack: React → FastAPI (Python) → PostgreSQL + pgvector → LLM.

## Phase 1: NASA document ingestion

Pulls documents from the [NASA Technical Reports Server (NTRS)](https://ntrs.nasa.gov),
stores raw PDFs on disk, extracts text, chunks it, and stores document
metadata + chunks in Postgres. Embeddings and retrieval come in Phase 2.

### Setup

```bash
# 1. Start Postgres (with pgvector) via Docker.
#    Runs on host port 5433, not 5432 — adjust in docker-compose.yml /
#    backend/.env if 5433 is also taken on your machine.
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

# 5. Run the API and check what landed.
#    Port 8001, not 8000 — pick your own if 8001 is also taken locally
#    (check with `lsof -nP -iTCP:8001 -sTCP:LISTEN`).
uvicorn app.main:app --reload --port 8001
curl http://127.0.0.1:8001/documents
```

### Frontend

Minimal Vite + React + TypeScript app that lists ingested documents from
`GET /documents`. Lives in `frontend/`, sibling to `backend/`.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (backend must be running on `:8001` — CORS is
already configured for the Vite dev server in `backend/app/main.py`).

