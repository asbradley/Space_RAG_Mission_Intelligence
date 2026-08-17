"""Local LLM generation via Ollama — free, no API key.

Requires Ollama running locally (`brew services start ollama`) with the
configured model pulled (`ollama pull llama3.2:3b`).
"""

import requests

from app.config import settings

TIMEOUT_SECONDS = 120


def generate(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the full response."""
    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["response"]
