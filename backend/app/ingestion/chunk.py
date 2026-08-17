def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping character-based chunks.

    Character-based rather than token-based to keep Phase 1 dependency-free;
    revisit with a tokenizer-aware splitter in Phase 2 if chunk boundaries
    turn out to hurt retrieval quality.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]
