import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Embedding dimension for OpenAI's text-embedding-3-small. Change this (and
# re-create the column) if you pick a different embedding model in Phase 2.
EMBEDDING_DIM = 1536


class Document(Base):
    """A single ingested NASA document (e.g. one NTRS citation)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # NTRS's own citation id, used to avoid re-ingesting the same document.
    source_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    source: Mapped[str] = mapped_column(String, default="ntrs")

    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publish_date: Mapped[datetime.date | None] = mapped_column(nullable=True)

    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A chunk of extracted text from a Document, ready for embedding."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    # Populated in Phase 2 once embedding is wired up; nullable for now so
    # ingestion (Phase 1) doesn't need to call an embedding model yet.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
