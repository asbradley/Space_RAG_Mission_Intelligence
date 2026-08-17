"""Local text embeddings via sentence-transformers — free, no API key.

Uses all-MiniLM-L6-v2 (384 dims, ~80MB, fast on CPU). If you swap models,
update `models.EMBEDDING_DIM` to match and re-create the `chunks.embedding`
column.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded lazily and cached so importing this module doesn't pay the
    # (one-time, ~seconds) model load cost until an embedding is actually
    # needed.
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input string."""
    if not texts:
        return []
    vectors = _model().encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
