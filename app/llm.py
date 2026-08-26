from __future__ import annotations
import requests
from .domain import Chunk


class LlmUnavailable(Exception):
    pass


def answer(question: str, matches: list[tuple[Chunk, float]], api_key: str | None, model: str, timeout: float) -> str:
    if not api_key:
        raise LlmUnavailable("GROQ_API_KEY is not configured")
    context = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(matches))
    prompt = ("Answer only from the supplied VTU documents. If the answer is absent, say so. "
              "Use concise prose and cite context markers like [1].\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as error:
        raise LlmUnavailable(str(error)) from error
