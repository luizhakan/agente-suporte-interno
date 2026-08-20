"""Testes do contrato visual compartilhado entre API, web e CLI."""
from src.domain.models import Chunk
from src.presentation.cli import format_cli_result
from src.presentation.formatting import evidence_excerpt, format_answer_for_display


def test_answer_uses_numeric_references_without_internal_ids_or_repeated_prefixes():
    raw_answer = (
        "De acordo com a seção Modelos (home_office.md): "
        "A empresa adota trabalho híbrido. [home_office_c1]\n\n"
        "De acordo com a seção Auxílio (home_office.md): "
        "- Há ajuda de custo mensal. [home_office_c2]"
    )

    formatted = format_answer_for_display(
        raw_answer,
        ["home_office_c1", "home_office_c2"],
    )

    assert formatted == (
        "A empresa adota trabalho híbrido. [1]\n\n"
        "- Há ajuda de custo mensal. [2]"
    )
    assert "home_office_c" not in formatted
    assert "De acordo com" not in formatted


def test_evidence_excerpt_returns_claim_linked_to_requested_chunk():
    raw_answer = (
        "A empresa adota trabalho híbrido. [home_office_c1]\n\n"
        "Há ajuda de custo de **R$ 150,00**. [home_office_c2]"
    )

    excerpt = evidence_excerpt(
        raw_answer,
        "home_office_c2",
        "Conteúdo de fallback.",
    )

    assert excerpt == "Há ajuda de custo de R$ 150,00."


def test_evidence_excerpt_limits_long_catalog_without_losing_grounding():
    raw_answer = f"{'Assunto documentado. ' * 40}[knowledge_base_catalog_c1]"

    excerpt = evidence_excerpt(
        raw_answer,
        "knowledge_base_catalog_c1",
        "Conteúdo de fallback.",
    )

    assert len(excerpt) <= 500
    assert excerpt.endswith("…")


def test_cli_uses_same_numeric_references_and_separates_technical_details():
    chunk = Chunk(
        chunk_id="home_office_c1",
        source="home_office.md",
        section="Modelos de Trabalho",
        content="A empresa adota trabalho híbrido.",
        score=0.72,
    )
    state = {
        "trace_id": "trace-teste",
        "question": "Posso trabalhar de casa?",
        "effective_query": "Posso trabalhar de casa?",
        "rewrite_count": 0,
        "retrieved_chunks": [chunk],
        "evidence_score": 0.72,
        "answer": "A empresa adota trabalho híbrido. [home_office_c1]",
        "cited_chunk_ids": ["home_office_c1"],
        "timings": {"generate_answer": 10.0},
        "failure": None,
    }

    output = format_cli_result(state["question"], state)

    assert "A empresa adota trabalho híbrido. [1]" in output
    assert "Evidências usadas (1)" in output
    assert "[1] Modelos de Trabalho" in output
    assert "score de recuperação: 0.720" in output
    assert "confiança" not in output.casefold()
