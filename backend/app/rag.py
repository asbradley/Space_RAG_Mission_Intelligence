"""Phase 5: hybrid retrieval + cross-encoder reranking + cited evidence.

retrieve() runs two independent searches over chunks — pgvector cosine
similarity and Postgres full-text keyword search — merges their rankings
with Reciprocal Rank Fusion (RRF), then reranks a wider candidate pool
with a cross-encoder before trimming to the final top_k. RRF only knows
where each candidate landed in two rank orderings; the cross-encoder
actually scores each candidate's text against the question, catching
cases where a good rank position didn't mean strong relevance.

build_prompt() numbers the final results into a prompt for
app.llm.generate(), and parse_citations() reads back which of those
numbered excerpts the answer actually cited (Phase 5).
"""

import re
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import reranker
from app.embeddings import embed_text
from app.models import Chunk, Document

TOP_K = 5
CANDIDATE_POOL_SIZE = 10  # how many results to pull from each search before fusion
RERANK_POOL_SIZE = 20  # how many fused candidates to hand to the reranker
RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant

# Which pipeline stages retrieve() should run -- see its docstring.
RetrievalMode = Literal["vector", "hybrid", "reranked"]


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


def retrieve(
    question: str,
    db: Session,
    top_k: int = TOP_K,
    mode: RetrievalMode = "reranked",
) -> list[RetrievedChunk]:
    """Return the top_k chunks for a question.

    `mode` selects how much of the pipeline runs, so the stages added in
    Phases 2-4 can be measured against each other (see
    scripts/evaluate.py). The default reproduces current behaviour:

      vector   - pgvector cosine similarity only            (Phase 2)
      hybrid   - + keyword search, merged by RRF            (Phase 3)
      reranked - + cross-encoder over the fused candidates  (Phase 4)
    """
    if mode not in ("vector", "hybrid", "reranked"):
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    # Pull at least top_k from each search, so callers asking for a deeper
    # top_k than the default pool size still get a full result set.
    pool_size = max(CANDIDATE_POOL_SIZE, top_k)
    query_vector = embed_text(question)
    vector_ids = _vector_search(query_vector, db, pool_size)

    if mode == "vector":
        candidate_ids = vector_ids[:top_k]
    else:
        keyword_ids = _keyword_search(question, db, pool_size)
        fused_ids = _reciprocal_rank_fusion([vector_ids, keyword_ids])
        # Reranking reorders a wider pool; hybrid alone takes the fused order.
        limit = max(RERANK_POOL_SIZE, top_k) if mode == "reranked" else top_k
        candidate_ids = fused_ids[:limit]

    if not candidate_ids:
        return []

    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(candidate_ids))
    ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}
    candidates = [by_id[cid] for cid in candidate_ids if cid in by_id]

    if mode == "reranked":
        scores = reranker.rerank(question, [chunk.text for chunk, _ in candidates])
        candidates = [
            pair
            for pair, _score in sorted(
                zip(candidates, scores), key=lambda pair: pair[1], reverse=True
            )
        ]

    results = []
    for chunk, document in candidates[:top_k]:
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
    """Build the answering prompt, numbering excerpts so the model can cite
    them inline as [1], [2], ... — see parse_citations()."""
    context = "\n\n".join(
        f"[{i}] {c.document_title}\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    return (
        "You are answering questions using only the NASA document excerpts "
        "below. If the excerpts don't contain the answer, say so — don't "
        "make anything up.\n\n"
        "Cite the excerpts that support your answer inline, using their "
        "bracketed numbers (for example: \"The CSM is the Command and "
        "Service Module [1].\").\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def parse_citations(answer: str, num_sources: int) -> set[int]:
    """Return the excerpt numbers the answer actually cited.

    Best-effort: a small local model may cite inconsistently or not at
    all, and may invent numbers with no excerpt behind them, so anything
    outside 1..num_sources is discarded rather than trusted.
    """
    return {
        n
        for n in (int(m) for m in re.findall(r"\[(\d+)\]", answer))
        if 1 <= n <= num_sources
    }
