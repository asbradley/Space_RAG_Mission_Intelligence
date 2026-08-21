"""Local cross-encoder reranking — free, no API key.

Unlike app.embeddings' bi-encoder (embeds query and chunk independently,
then compares vectors), a cross-encoder scores a (query, chunk) pair
jointly — slower per pair but more accurate at judging relevance, so it's
used on a small shortlist rather than the whole corpus.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2, shipped by the
sentence-transformers package already required for app.embeddings — no
new dependency.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # Loaded lazily and cached, same reasoning as app.embeddings._model.
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, documents: list[str]) -> list[float]:
    """Return a relevance score per document, in the same order as input."""
    if not documents:
        return []
    pairs = [(query, doc) for doc in documents]
    return _model().predict(pairs).tolist()
