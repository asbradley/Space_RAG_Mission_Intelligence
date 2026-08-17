"""Raw-document storage. Local disk for now; swap the implementation for
S3 later without touching ingestion logic — callers only see `save()`.
"""

from pathlib import Path

from app.config import settings


def save(filename: str, content: bytes) -> str:
    """Save raw bytes to the local storage dir and return the path used."""
    settings.raw_storage_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_storage_dir / filename
    path.write_bytes(content)
    return str(path)
