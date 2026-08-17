"""Thin client for the NASA Technical Reports Server (NTRS) search API.

NOTE: NTRS's API response shape isn't guaranteed by this code — it was
written from public documentation/observed responses, not a versioned
schema. Run `python -m app.ingestion.ntrs_client "apollo 11"` once to dump
a raw response and confirm field names before trusting this in a real
ingestion run; adjust the `.get(...)` paths below if NASA changes anything.
"""

from __future__ import annotations

import sys
from typing import Any

import requests

from app.config import settings

TIMEOUT_SECONDS = 30


def search(query: str, page_size: int = 25, page_from: int = 0) -> list[dict[str, Any]]:
    """Search NTRS and return a list of raw result dicts."""
    resp = requests.post(
        settings.ntrs_api_base,
        json={"q": query, "page": {"size": page_size, "from": page_from}},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def to_document_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Map a raw NTRS result into the fields our Document model expects."""
    authors = result.get("authorAffiliations") or result.get("authors") or []
    author_names = ", ".join(
        a.get("meta", {}).get("author", {}).get("name", "")
        if isinstance(a, dict) and "meta" in a
        else str(a)
        for a in authors
    ).strip(", ")

    pdf_url = None
    for download in result.get("downloads", []) or []:
        links = download.get("links", {}) if isinstance(download, dict) else {}
        if links.get("pdf"):
            pdf_url = links["pdf"]
            break

    return {
        "source_id": str(result.get("id")),
        "source": "ntrs",
        "title": result.get("title", "Untitled"),
        "authors": author_names or None,
        "abstract": result.get("abstract"),
        "publish_date": result.get("publicationDate") or result.get("created"),
        "source_url": f"https://ntrs.nasa.gov/citations/{result.get('id')}",
        "pdf_url": pdf_url,
    }


def download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.content


if __name__ == "__main__":
    # Quick manual check: dump the first raw result for a query so you can
    # eyeball the real field names against the mapping above.
    import json

    q = sys.argv[1] if len(sys.argv) > 1 else "apollo 11"
    results = search(q, page_size=1)
    print(json.dumps(results[0] if results else {}, indent=2))
