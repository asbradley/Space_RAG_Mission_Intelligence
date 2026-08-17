"""Phase 2: basic retrieval-augmented answering.

retrieve() does a pgvector cosine-similarity search over chunks;
build_prompt() stuffs the results into a prompt for app.llm.generate().
Reranking, hybrid search, and citation tracking are later phases — this
is intentionally the simplest version that works end to end.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_text
from app.models import Chunk, Document

TOP_K = 5


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_title: str
    document_source_url: str | None
    text: str


def retrieve(question: str, db: Session, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Return the top_k chunks most similar to the question, by cosine distance."""
    query_vector = embed_text(question)

    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    rows = db.execute(stmt).all()

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_title=document.title,
            document_source_url=document.source_url,
            text=chunk.text,
        )
        for chunk, document in rows
    ]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Source: {c.document_title}]\n{c.text}" for c in chunks
    )
    return (
        "You are answering questions using only the NASA document excerpts "
        "below. If the excerpts don't contain the answer, say so — don't "
        "make anything up.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
