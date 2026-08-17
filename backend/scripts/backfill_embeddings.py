"""One-off backfill for chunks ingested before embedding was wired into
ingest.py. Safe to re-run — only touches rows where embedding IS NULL.

    python -m scripts.backfill_embeddings
"""

from app.db import SessionLocal
from app.embeddings import embed_texts
from app.models import Chunk

BATCH_SIZE = 64


def main() -> None:
    db = SessionLocal()
    try:
        pending = db.query(Chunk).filter(Chunk.embedding.is_(None)).all()
        print(f"{len(pending)} chunks need embeddings")

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            vectors = embed_texts([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk.embedding = vector
            db.commit()
            print(f"  embedded {min(i + BATCH_SIZE, len(pending))}/{len(pending)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
