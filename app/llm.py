from __future__ import annotations

import requests

from .domain import Chunk


class LlmUnavailable(Exception):
    pass


def answer(
    question: str,
    matches: list[tuple[Chunk, float]],
    api_key: str | None,
    model: str,
    timeout: float,
) -> str:

    if not api_key:
        raise LlmUnavailable(
            "GROQ_API_KEY is not configured"
        )

    if not matches:
        return (
            "I could not find relevant material in the "
            "supplied documents."
        )

    context_parts = []

    for i, (chunk, score) in enumerate(matches, start=1):

        source = (
            f"[{i}] "
            f"{chunk.document}, page {chunk.page}"
        )

        if chunk.heading:
            source += f", {chunk.heading}"

        context_parts.append(
            f"{source}\n{chunk.text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are ExamPrepRAG, a study assistant for VTU students.

Answer the student's question using ONLY the supplied document
evidence.

IMPORTANT RULES:

1. Do not invent information that is not supported by the evidence.
2. Use ALL relevant retrieved passages, not just the first passage.
3. If the question asks for topics, list the topics clearly.
4. If the question asks what is important, identify the important
   study/exam topics supported by the documents.
5. If the question asks to explain something, explain it using the
   supplied notes.
6. If the question asks for a comparison, compare the concepts using
   the supplied material.
7. If the question asks about previous papers, use question-paper
   evidence and mention years/pages when available.
8. If the question asks about the syllabus, prioritize syllabus
   evidence over notes.
9. If the question asks for Module N, prioritize evidence belonging
   to Module N.
10. If the evidence is insufficient, explicitly say what is missing.
11. Never pretend that a topic is important merely because it appears
    in one random passage.
12. Cite factual claims using source markers [1], [2], etc.
13. Keep the answer concise and exam-oriented.
14. Use Markdown headings, bullet points, and numbered lists where
    appropriate.
15. Do not repeat the question.

QUESTION:
{question}

SUPPLIED DOCUMENT EVIDENCE:
{context}

Now answer the question.
""".strip()

    try:

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise, grounded "
                            "VTU exam-preparation assistant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            },
        )

        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    except (
        requests.RequestException,
        KeyError,
        IndexError,
    ) as error:

        raise LlmUnavailable(
            str(error)
        ) from error