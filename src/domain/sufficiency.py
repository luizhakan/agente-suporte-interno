"""Regras puras de domínio para suficiência de evidência."""
from typing import List, Optional, Tuple
from src.domain.models import Chunk


def check_evidence_sufficiency(
    chunks: List[Chunk],
    tau: float,
    delta: float,
    strong_tau: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Avalia se os trechos recuperados atendem à regra de suficiência de evidência:
    - um chunk com correspondência forte (score >= tau + delta); ou
    - score_max >= tau e ao menos 2 chunks com score >= tau - delta.
    
    Retorna:
        Tuple[bool, float]: (is_sufficient, score_max)
    """
    if not chunks:
        return False, 0.0

    scores = [chunk.score for chunk in chunks]
    score_max = max(scores)

    strong_threshold = strong_tau if strong_tau is not None else tau + delta
    if score_max >= strong_threshold:
        return True, score_max

    if score_max < tau:
        return False, score_max

    corroboration_threshold = tau - delta
    qualifying_by_source = {}
    for chunk in chunks:
        if chunk.score >= corroboration_threshold:
            qualifying_by_source[chunk.source] = qualifying_by_source.get(chunk.source, 0) + 1

    is_sufficient = any(count >= 2 for count in qualifying_by_source.values())
    return is_sufficient, score_max
