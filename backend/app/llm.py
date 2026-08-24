"""Local LLM generation via Ollama — free, no API key.

Requires Ollama running locally (`brew services start ollama`) with the
configured model pulled (`ollama pull llama3.2:3b`).
"""

import requests

from app.config import settings

TIMEOUT_SECONDS = 120


def generate(prompt: str, temperature: float | None = None) -> str:
    """Send a prompt to the local Ollama server and return the full response.

    `temperature` is left to Ollama's default (0.8) unless a caller pins it.
    scripts/evaluate.py passes 0, because at the default the answer-quality
    metrics move by up to 0.20 between identical runs -- enough to invent
    differences between retrieval modes that are not there.
    """
    payload = {"model": settings.ollama_model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["response"]
