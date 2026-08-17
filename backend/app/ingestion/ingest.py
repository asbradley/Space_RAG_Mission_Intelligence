"""CLI entry point for Phase 1: search NTRS, download PDFs, extract text,
chunk it, and store documents + chunks in Postgres.

Usage:
    python -m app.ingestion.ingest "apollo 11 mission report" --limit 10
"""

from __future__ import annotations

import argparse
import datetime

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.embeddings import embed_texts
from app.ingestion import chunk as chunker
from app.ingestion import extract, ntrs_client, storage
from app.models import Chunk, Document


def _parse_date(value: str | None) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None


def ingest_query(query: str, limit: int, db: Session) -> None:
    results = ntrs_client.search(query, page_size=limit)
    print(f"Found {len(results)} results for '{query}'")

    for raw in results:
        fields = ntrs_client.to_document_fields(raw)

        existing = (
            db.query(Document).filter_by(source_id=fields["source_id"]).first()
        )
        if existing:
            print(f"  skip (already ingested): {fields['title']}")
            continue

        if not fields["pdf_url"]:
            print(f"  skip (no PDF available): {fields['title']}")
            continue

        print(f"  ingesting: {fields['title']}")
        pdf_bytes = ntrs_client.download_pdf(fields["pdf_url"])
        raw_path = storage.save(f"{fields['source_id']}.pdf", pdf_bytes)

        text = extract.extract_text(pdf_bytes)
        chunks = chunker.chunk_text(text)
        embeddings = embed_texts(chunks)

        doc = Document(
            source_id=fields["source_id"],
            source=fields["source"],
            title=fields["title"],
            authors=fields["authors"],
            abstract=fields["abstract"],
            publish_date=_parse_date(fields["publish_date"]),
            source_url=fields["source_url"],
            raw_file_path=raw_path,
        )
        db.add(doc)
        db.flush()  # assigns doc.id

        for i, (text_chunk, vector) in enumerate(zip(chunks, embeddings)):
            db.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    text=text_chunk,
                    embedding=vector,
                )
            )

        db.commit()
        print(f"    stored {len(chunks)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NASA documents from NTRS.")
    parser.add_argument("query", help="NTRS search query, e.g. 'apollo 11 mission report'")
    parser.add_argument("--limit", type=int, default=10, help="Max results to fetch")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        ingest_query(args.query, args.limit, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
