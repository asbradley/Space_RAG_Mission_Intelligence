from io import BytesIO

from pypdf import PdfReader


def extract_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF's raw bytes, page by page."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
