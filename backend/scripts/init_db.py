"""Create the pgvector extension and all tables. Run once against a fresh
database:

    python -m scripts.init_db
"""

from sqlalchemy import text

from app.db import Base, engine
from app import models  # noqa: F401  (registers models on Base.metadata)


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # GIN index for Phase 3 keyword/full-text search over chunks.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_text_search "
                "ON chunks USING GIN (text_search)"
            )
        )
    print("Database initialized.")


if __name__ == "__main__":
    main()
