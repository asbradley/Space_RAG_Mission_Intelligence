from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document

app = FastAPI(title="Space RAG API")

# Allow the Vite dev server to call the API. Tighten/replace this for
# non-local environments once the frontend is actually deployed somewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    """List ingested documents. Just enough to confirm Phase 1 ingestion
    landed rows in Postgres — retrieval/RAG endpoints come in Phase 2."""
    docs = db.execute(select(Document).order_by(Document.ingested_at.desc())).scalars().all()
    return [
        {
            "id": d.id,
            "title": d.title,
            "source_id": d.source_id,
            "source_url": d.source_url,
            "ingested_at": d.ingested_at,
        }
        for d in docs
    ]
