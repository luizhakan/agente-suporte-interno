"""Formatação compartilhada de respostas fundamentadas para API, web e CLI."""
from __future__ import annotations

import re
from typing import Iterable


_CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")
_SOURCE_PREFIX_PATTERN = re.compile(
    r"De acordo com a seção\s+.*?\s+\([^)]+\.md\):\s*",
    flags=re.IGNORECASE,
)


def _clean_grounded_text(text: str) -> str:
    """Remove ruído técnico sem alterar os fatos extraídos da fonte."""
    cleaned = _SOURCE_PREFIX_PATTERN.sub("", text or "")
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"[ \t]+-\s+", "\n- ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_answer_for_display(answer: str | None, cited_chunk_ids: Iterable[str]) -> str:
    """Converte IDs internos em referências numéricas estáveis e legíveis."""
    formatted = _clean_grounded_text(answer or "")
    citation_numbers = {
        chunk_id: number
        for number, chunk_id in enumerate(cited_chunk_ids, start=1)
    }
    formatted = _CITATION_PATTERN.sub(
        lambda match: (
            f"[{citation_numbers[match.group(1)]}]"
            if match.group(1) in citation_numbers
            else ""
        ),
        formatted,
    )
    formatted = re.sub(r"(\[\d+\])\s+(?=\S)", r"\1\n\n", formatted)
    return re.sub(r"\n{3,}", "\n\n", formatted).strip()


def evidence_excerpt(
    answer: str | None,
    chunk_id: str,
    fallback_content: str,
) -> str:
    """Extrai do texto validado a afirmação associada ao chunk citado."""
    def shorten(text: str) -> str:
        if len(text) <= 500:
            return text
        shortened = text[:497].rsplit(" ", 1)[0].rstrip(" ,;:")
        return f"{shortened}…"

    raw_answer = answer or ""
    cursor = 0
    previous_excerpt = ""
    for match in _CITATION_PATTERN.finditer(raw_answer):
        candidate = _clean_grounded_text(raw_answer[cursor:match.start()])
        if candidate:
            previous_excerpt = candidate
        if match.group(1) == chunk_id and previous_excerpt:
            return shorten(previous_excerpt)
        cursor = match.end()

    fallback = _clean_grounded_text(fallback_content)
    return shorten(fallback)
