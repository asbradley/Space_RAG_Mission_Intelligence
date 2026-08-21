"""Phase 3: hybrid (vector + keyword) retrieval-augmented answering.

retrieve() runs two independent searches over chunks — pgvector cosine
similarity and Postgres full-text keyword search — and merges their
rankings with Reciprocal Rank Fusion (RRF). Vector search alone misses
exact-term matches (acronyms, part numbers) when the surrounding meaning
isn't a close embedding match; keyword search alone misses paraphrases.
Combining both catches more of each. Reranking and citation tracking are
later phases.

build_prompt() stuffs the merged results into a prompt for
app.llm.generate().
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embed_text
from app.models import Chunk, Document

TOP_K = 5
CANDIDATE_POOL_SIZE = 10  # how many results to pull from each search before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_title: str
    document_source_url: str | None
    text: str


def _vector_search(query_vector: list[float], db: Session, limit: int) -> list[int]:
    """Return chunk ids ranked by cosine similarity to query_vector."""
    stmt = (
        select(Chunk.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def _keyword_search(question: str, db: Session, limit: int) -> list[int]:
    """Return chunk ids ranked by Postgres full-text search relevance."""
    tsquery = func.plainto_tsquery("english", question)
    stmt = (
        select(Chunk.id)
        .where(Chunk.text_search.op("@@")(tsquery))
        .order_by(func.ts_rank(Chunk.text_search, tsquery).desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def _reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge multiple ranked id lists into one, by Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank) per item; scores from lists an item
    doesn't appear in are simply 0. This avoids having to calibrate cosine
    distance and ts_rank onto a shared scale.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(question: str, db: Session, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Return the top_k chunks for a question, combining vector similarity
    and keyword search via Reciprocal Rank Fusion."""
    query_vector = embed_text(question)
    vector_ids = _vector_search(query_vector, db, CANDIDATE_POOL_SIZE)
    keyword_ids = _keyword_search(question, db, CANDIDATE_POOL_SIZE)

    fused_ids = _reciprocal_rank_fusion([vector_ids, keyword_ids])[:top_k]
    if not fused_ids:
        return []

    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(fused_ids))
    ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}

    results = []
    for chunk_id in fused_ids:
        chunk, document = by_id[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_title=document.title,
                document_source_url=document.source_url,
                text=chunk.text,
            )
        )
    return results


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
