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
    print("Database initialized.")


if __name__ == "__main__":
    main()
