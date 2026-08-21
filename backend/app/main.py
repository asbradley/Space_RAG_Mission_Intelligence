from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import llm, rag
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


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    """Retrieve relevant chunks, ask the local LLM to answer using only
    those excerpts, and return the evidence behind the answer.

    Each source is one distinct passage (not one per document), numbered
    to match the [n] markers the answer cites. `cited` marks the ones the
    answer actually referenced — best-effort, since a small local model
    doesn't always cite reliably."""
    chunks = rag.retrieve(req.question, db)
    if not chunks:
        return {"answer": "No ingested documents to search yet.", "sources": []}

    prompt = rag.build_prompt(req.question, chunks)
    answer = llm.generate(prompt)
    cited = rag.parse_citations(answer, len(chunks))

    return {
        "answer": answer,
        "sources": [
            {
                "n": i,
                "chunk_id": c.chunk_id,
                "title": c.document_title,
                "source_url": c.document_source_url,
                "excerpt": c.text,
                "cited": i in cited,
            }
            for i, c in enumerate(chunks, start=1)
        ],
    }
