"""Testes unitários para a regra de suficiência de evidência (Seção 5 da Arquitetura)."""
import pytest
from src.domain.models import Chunk
from src.domain.sufficiency import check_evidence_sufficiency


def test_sufficiency_empty_chunks():
    is_suff, max_score = check_evidence_sufficiency([], tau=0.5, delta=0.1)
    assert not is_suff
    assert max_score == 0.0


def test_sufficiency_score_below_tau():
    chunks = [
        Chunk(chunk_id="c1", source="f.md", section="S1", content="text 1", score=0.40),
        Chunk(chunk_id="c2", source="f.md", section="S1", content="text 2", score=0.35),
    ]
    # score_max (0.40) < tau (0.50)
    is_suff, max_score = check_evidence_sufficiency(chunks, tau=0.5, delta=0.1)
    assert not is_suff
    assert max_score == 0.40


def test_sufficiency_score_meets_tau_but_lacks_second_corroborating_chunk():
    chunks = [
        Chunk(chunk_id="c1", source="f.md", section="S1", content="text 1", score=0.55),
        Chunk(chunk_id="c2", source="f.md", section="S1", content="text 2", score=0.20),
    ]
    # tau=0.50, delta=0.10 -> tau - delta = 0.40. Only 1 chunk >= 0.40.
    is_suff, max_score = check_evidence_sufficiency(chunks, tau=0.5, delta=0.1)
    assert not is_suff
    assert max_score == 0.55


def test_sufficiency_accepts_single_strong_chunk():
    chunks = [
        Chunk(chunk_id="c1", source="f.md", section="S1", content="text 1", score=0.65),
        Chunk(chunk_id="c2", source="f.md", section="S2", content="text 2", score=0.20),
    ]

    is_suff, max_score = check_evidence_sufficiency(chunks, tau=0.5, delta=0.1)

    assert is_suff
    assert max_score == 0.65


def test_sufficiency_satisfied_with_two_corroborating_chunks():
    chunks = [
        Chunk(chunk_id="c1", source="f.md", section="S1", content="text 1", score=0.75),
        Chunk(chunk_id="c2", source="f.md", section="S2", content="text 2", score=0.62),
        Chunk(chunk_id="c3", source="f.md", section="S3", content="text 3", score=0.20),
    ]
    # tau=0.60, delta=0.15 -> threshold = 0.45. Chunks c1 and c2 meet criteria.
    is_suff, max_score = check_evidence_sufficiency(chunks, tau=0.60, delta=0.15)
    assert is_suff
    assert max_score == 0.75


def test_sufficiency_does_not_mix_unrelated_sources_for_corroboration():
    chunks = [
        Chunk(chunk_id="c1", source="ferias.md", section="S1", content="text 1", score=0.55),
        Chunk(chunk_id="c2", source="reembolso.md", section="S2", content="text 2", score=0.50),
    ]

    is_suff, max_score = check_evidence_sufficiency(
        chunks,
        tau=0.5,
        delta=0.1,
        strong_tau=0.7,
    )

    assert not is_suff
    assert max_score == 0.55
