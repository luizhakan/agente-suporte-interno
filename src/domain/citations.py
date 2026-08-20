"""Regras de identificação de citações pertencentes ao contexto recuperado."""
from __future__ import annotations

import re
from collections.abc import Iterable


_BRACKETED_TOKEN_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")


def extract_context_citations(
    text: str,
    available_chunk_ids: Iterable[str],
) -> list[str]:
    """Extrai, em ordem, apenas marcações que sejam IDs do contexto informado."""
    available = set(available_chunk_ids)
    citations: list[str] = []
    seen: set[str] = set()

    for match in _BRACKETED_TOKEN_PATTERN.finditer(text or ""):
        chunk_id = match.group(1)
        if chunk_id in available and chunk_id not in seen:
            citations.append(chunk_id)
            seen.add(chunk_id)

    return citations
