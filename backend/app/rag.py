"""Hybrid retrieval + cited evidence, with optional cross-encoder reranking.

retrieve() runs two independent searches over chunks — pgvector cosine
similarity and Postgres full-text keyword search — and merges their
rankings with Reciprocal Rank Fusion (RRF).

A cross-encoder reranking stage (Phase 4) runs by default. Phase 6
measured all three modes: with the per-document cap in place they tie on
hit@5 and recall@5, and reranking leads on MRR (0.461 vs hybrid's 0.402),
for ~77 ms. Answer-quality metrics cannot separate the modes at this eval
size — they vary by up to 0.20 between identical runs, since generation is
sampled at Ollama's default temperature. See docs/phase-6-evaluation.md.

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
MAX_CHUNKS_PER_DOCUMENT = 2  # keep one source from filling every result slot
_SENTENCE_END = ".!?\"')]"  # trailing chars that read as a clean stop

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


def _cap_per_document(
    candidates: list[tuple], top_k: int, max_per_document: int = MAX_CHUNKS_PER_DOCUMENT
) -> list[tuple]:
    """Trim to top_k, allowing at most max_per_document chunks per document.

    Without this, a broad question ("What was the Apollo 11 mission?")
    returns five slices of whichever single document mentions the query
    terms most often, which reads as five near-identical excerpts. The cap
    is a floor on source diversity, not a relevance judgement: candidates
    passed over are used to fill any shortfall, so top_k is still met when
    the corpus has too few documents to satisfy the cap.
    """
    kept, overflow, per_document = [], [], {}
    for chunk, document in candidates:
        if per_document.get(document.id, 0) < max_per_document:
            kept.append((chunk, document))
            per_document[document.id] = per_document.get(document.id, 0) + 1
            if len(kept) == top_k:
                return kept
        else:
            overflow.append((chunk, document))
    return (kept + overflow)[:top_k]


def retrieve(
    question: str,
    db: Session,
    top_k: int = TOP_K,
    mode: RetrievalMode = "reranked",
) -> list[RetrievedChunk]:
    """Return the top_k chunks for a question.

    `mode` selects how much of the pipeline runs, so the stages added in
    Phases 2-4 can be measured against each other (see
    scripts/evaluate.py):

      vector   - pgvector cosine similarity only            (Phase 2)
      hybrid   - + keyword search, merged by RRF            (Phase 3)
      reranked - + cross-encoder over the fused candidates  (Phase 4)

    Defaults to "reranked": it ties the others on hit@5 and recall@5 and
    leads on MRR, which is the only metric that both discriminates between
    the modes and is deterministic.
    """
    if mode not in ("vector", "hybrid", "reranked"):
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    # Pull at least top_k from each search, so callers asking for a deeper
    # top_k than the default pool size still get a full result set.
    pool_size = max(CANDIDATE_POOL_SIZE, top_k)
    query_vector = embed_text(question)
    vector_ids = _vector_search(query_vector, db, pool_size)

    # Every mode keeps a pool wider than top_k, so _cap_per_document has
    # alternatives to reach for when one document dominates the ranking.
    if mode == "vector":
        candidate_ids = vector_ids
    else:
        keyword_ids = _keyword_search(question, db, pool_size)
        fused_ids = _reciprocal_rank_fusion([vector_ids, keyword_ids])
        candidate_ids = fused_ids[: max(RERANK_POOL_SIZE, top_k)]

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
    for chunk, document in _cap_per_document(candidates, top_k):
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_title=document.title,
                document_source_url=document.source_url,
                text=chunk.text,
            )
        )
    return results


def trim_to_word_boundaries(text: str) -> str:
    """Drop the partial words at a chunk's edges, for display only.

    Chunks are fixed-size character slices (see ingestion/chunk.py), so
    every chunk but a document's first and last begins and ends mid-word:
    "s, flags of the 50 states...", "...both versions are basically identi".

    This is presentation only. build_prompt() and every retrieval path
    still use the raw chunk text, so answers, embeddings and the Phase 6
    numbers are all unaffected by it.
    """
    trimmed = text.strip()
    if len(trimmed.split()) < 3:
        return trimmed
    # A lowercase first character means the slice landed inside a word or
    # sentence; a real chunk start would be a capital, digit or quote.
    if trimmed[0].islower():
        trimmed = "\u2026" + trimmed.split(None, 1)[1]
    if trimmed[-1] not in _SENTENCE_END:
        trimmed = trimmed.rsplit(None, 1)[0]
        # Dropping the fragment may have left a clean sentence end, in which
        # case a trailing ellipsis on top of it just reads as noise.
        if trimmed[-1] not in _SENTENCE_END:
            trimmed += "\u2026"
    return trimmed


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
