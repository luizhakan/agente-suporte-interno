"""Testes dos adaptadores de infraestrutura e suas falhas de configuração."""
import json
import os

import httpx
import numpy as np
import pytest

from src.config import settings
from src.domain.models import Chunk, ChunkKind
from src.infrastructure.llm_adapter import (
    MockLLMAdapter,
    OllamaLLMAdapter,
    OpenAICompatibleLLMAdapter,
    get_llm_client,
)
from src.infrastructure.embeddings import (
    EmbeddingService,
    LocalDenseEmbedder,
    OllamaEmbeddingAdapter,
)
from src.infrastructure.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_mock_llm_omits_irrelevant_higher_scored_chunk():
    chunks = [
        Chunk(
            chunk_id="ferias_pagamento",
            source="ferias.md",
            section="Pagamento",
            content="A remuneração das férias será paga antes do período.",
            score=0.90,
        ),
        Chunk(
            chunk_id="ferias_fracionamento",
            source="ferias.md",
            section="Fracionamento",
            content="As férias podem ser fracionadas em até 3 períodos.",
            score=0.80,
        ),
    ]

    answer, cited_ids = await MockLLMAdapter().generate_answer(
        "Como funciona o fracionamento de férias?",
        chunks,
    )

    assert cited_ids == ["ferias_fracionamento"]
    assert "[ferias_fracionamento]" in answer
    assert "ferias_pagamento" not in answer


def test_llm_provider_does_not_silently_fall_back_to_mock(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "provedor-inexistente")

    with pytest.raises(RuntimeError, match="LLM_PROVIDER não suportado"):
        get_llm_client()


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm_client()


@pytest.mark.asyncio
async def test_ollama_provider_is_selected_without_cloud_credentials(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "modelo-local")

    adapter = get_llm_client()
    try:
        assert isinstance(adapter, OllamaLLMAdapter)
        assert adapter.provider_name == "ollama"
        assert adapter.model_name == "modelo-local"
    finally:
        await adapter.aclose()


def test_ollama_embedding_uses_distinct_retrieval_prompts_and_normalizes_vectors():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        input_texts = requests[-1]["input"]
        return httpx.Response(
            200,
            json={"embeddings": [[3.0, 4.0] for _ in input_texts]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaEmbeddingAdapter(client=client, model="embedding-teste")

    query_vector = adapter.encode_query("Como evitar invasões?")
    document_vectors = adapter.encode_documents(["Use MFA e senhas fortes."])

    assert requests[0]["input"][0].startswith("task: search result | query:")
    assert requests[1]["input"][0].startswith("title: Política interna | text:")
    assert requests[0]["truncate"] is False
    assert np.linalg.norm(query_vector) == pytest.approx(1.0)
    assert np.linalg.norm(document_vectors[0]) == pytest.approx(1.0)
    assert adapter.fingerprint == "ollama:embedding-teste:retrieval-v1"
    client.close()


def test_embedding_service_reports_remote_io_for_ollama():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        ),
        base_url="http://ollama.test/api/",
    )
    service = EmbeddingService(
        OllamaEmbeddingAdapter(client=client, model="embedding-teste")
    )

    assert service.provider_name == "ollama"
    assert service.requires_io is True
    assert service.get_query_vector("consulta").tolist() == [1.0, 0.0]
    client.close()


def test_semantic_constraint_distinguishes_modifier_from_missing_object():
    modifier_improvement = InMemoryVectorStore._unsupported_constraint_improvement(
        ["trabalhar", "em", "minha", "humilde", "residencia"],
        {"trabalhar": -0.02, "humilde": 0.038, "residencia": -0.053},
        margin=0.03,
    )
    missing_object_improvement = InMemoryVectorStore._unsupported_constraint_improvement(
        ["levar", "meu", "cachorro", "para", "trabalhar"],
        {"levar": -0.08, "cachorro": 0.04, "trabalhar": -0.05},
        margin=0.03,
    )

    assert modifier_improvement == 0.0
    assert missing_object_improvement == pytest.approx(0.04)


def test_catalog_retrieval_text_distinguishes_scope_from_specific_topic():
    store = InMemoryVectorStore()
    store.chunks = [
        Chunk(
            chunk_id="catalog",
            source="base_interna",
            section="Escopo",
            content="Férias; Home office; Segurança.",
            retrieval_content="Como este assistente pode ajudar? Assuntos disponíveis.",
            kind=ChunkKind.CATALOG,
        ),
        Chunk(
            chunk_id="home",
            source="home_office.md",
            section="Trabalho remoto",
            content="É possível trabalhar de casa em regime híbrido.",
        ),
    ]
    store._build_lexical_index()

    scope_scores = store._lexical_scores("Como vocês podem ajudar?")
    topic_scores = store._lexical_scores("Consigo trabalhar em casa?")

    assert scope_scores[0] > 0
    assert scope_scores[1] == 0
    assert topic_scores[0] == 0
    assert topic_scores[1] > 0


@pytest.mark.asyncio
async def test_ollama_generates_with_native_api_and_extracts_valid_citations():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "qwen-teste",
                "message": {"role": "assistant", "content": "SIM"},
                "done": True,
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="c1",
            source="doc.md",
            section="S1",
            content="Texto",
            score=0.70,
        )
    ]

    answer, citations = await adapter.generate_answer("Pergunta", chunks)

    assert "Texto" in answer
    assert "[c1]" in answer
    assert citations == ["c1"]
    assert requests[0].url.path == "/api/chat"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "qwen-teste"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 0
    assert payload["options"]["num_predict"] == 20
    assert payload["think"] is False
    assert "Texto" in payload["messages"][1]["content"]
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_uses_matched_catalog_exclusively_without_model_verification():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("O catálogo validado não deve depender do verificador local")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="knowledge_base_catalog_c1",
            source="base_interna",
            section="Escopo",
            content="Assuntos disponíveis:\n- Férias\n- Segurança\n- Onboarding\n- Reembolso",
            kind=ChunkKind.CATALOG,
            score=0.40,
            catalog_match=True,
        ),
        Chunk(
            chunk_id="home_office_c2",
            source="home_office.md",
            section="Auxílio",
            content="Há ajuda de custo para trabalho remoto.",
            score=0.35,
        ),
    ]

    answer, citations = await adapter.generate_answer(
        "Como vocês podem me ajudar?",
        chunks,
    )

    assert citations == ["knowledge_base_catalog_c1"]
    assert "Férias" in answer
    assert "Reembolso" in answer
    assert "ajuda de custo" not in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_rejects_thematically_related_but_non_answering_chunks():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "qwen-teste",
                    "message": {"role": "assistant", "content": "NAO"},
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="home_office_c1",
            source="home_office.md",
            section="Modelos de Trabalho",
            content="O regime híbrido prevê dias presenciais no escritório.",
            score=0.40,
        )
    ]

    answer, citations = await adapter.generate_answer(
        "Posso levar meu cachorro ao escritório?",
        chunks,
    )

    assert citations == []
    assert "Não encontrei" in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_recovers_literal_high_confidence_match_after_model_refusal():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "qwen-teste",
                    "message": {"role": "assistant", "content": "NAO"},
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="reembolso_c2",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="As despesas reembolsáveis incluem alimentação e transporte.",
            score=0.70,
        )
    ]

    answer, citations = await adapter.generate_answer(
        "Quais despesas podem ser reembolsadas?",
        chunks,
    )

    assert citations == ["reembolso_c2"]
    assert "alimentação e transporte" in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_does_not_accept_policy_for_an_unmentioned_acronym():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "qwen-teste",
                    "message": {"role": "assistant", "content": "SIM"},
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="reembolso_c5",
            source="reembolso.md",
            section="Prazos",
            content="Solicitações de reembolso devem ser enviadas até o dia 20.",
            score=0.38,
        )
    ]

    answer, citations = await adapter.generate_answer(
        "Como pedir reembolso de VR?",
        chunks,
    )

    assert citations == []
    assert "Não encontrei" in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_filters_weak_secondary_chunk_even_if_model_approves_it():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "qwen-teste",
                    "message": {"role": "assistant", "content": "SIM"},
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="home_office_c2",
            source="home_office.md",
            section="Auxílio",
            content="Há reembolso de cadeira ergonômica.",
            score=0.54,
        ),
        Chunk(
            chunk_id="reembolso_c2",
            source="reembolso.md",
            section="Despesas",
            content="Há reembolso de alimentação em viagens.",
            score=0.25,
        ),
    ]

    answer, citations = await adapter.generate_answer(
        "Tem subsídio para comprar cadeira ergonômica?",
        chunks,
    )

    assert citations == ["home_office_c2"]
    assert "alimentação" not in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_rejects_isolated_low_score_semantic_match():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "qwen-teste",
                    "message": {"role": "assistant", "content": "SIM"},
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="home_office_c1",
            source="home_office.md",
            section="Modelos de Trabalho",
            content="O trabalho remoto exige aprovação da Diretoria.",
            score=0.32,
        )
    ]

    answer, citations = await adapter.generate_answer(
        "Qual o modelo do carro da diretoria?",
        chunks,
    )

    assert citations == []
    assert "Não encontrei" in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_uses_document_consensus_for_semantic_paraphrase():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        verifier_input = payload["messages"][1]["content"]
        decision = "SIM" if "três modelos de trabalho" in verifier_input else "NAO"
        return httpx.Response(
            200,
            json={
                "model": "qwen-teste",
                "message": {"role": "assistant", "content": decision},
                "done": True,
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="qwen-teste")
    chunks = [
        Chunk(
            chunk_id="home_office_c1",
            source="home_office.md",
            section="Modelos de Trabalho",
            content=(
                "A empresa adota três modelos de trabalho:\n"
                "- Presencial.\n- Híbrido.\n- Remoto total mediante aprovação."
            ),
            score=0.32,
        ),
        Chunk(
            chunk_id="home_office_c2",
            source="home_office.md",
            section="Auxílio",
            content="Há ajuda de custo para regimes híbrido e remoto.",
            score=0.30,
        ),
        Chunk(
            chunk_id="home_office_c3",
            source="home_office.md",
            section="Jornada",
            content="A jornada contratual deve ser mantida.",
            score=0.29,
        ),
    ]

    answer, citations = await adapter.generate_answer(
        "Consigo trabalhar em meu lar?",
        chunks,
    )

    assert citations == ["home_office_c1"]
    assert "modelos de trabalho" in answer
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_health_requires_configured_model_to_be_downloaded():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"models": [{"name": "outro-modelo", "model": "outro-modelo"}]},
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="modelo-esperado")

    assert await adapter.is_healthy() is False
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_rewrite_rejects_semantically_unrelated_refusal():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "modelo-local",
                    "message": {
                        "role": "assistant",
                        "content": "Não posso ajudar com atividades ilegais.",
                    },
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="modelo-local")

    rewritten = await adapter.rewrite_query("Meu notebook foi furtado")

    assert "furto roubo equipamento corporativo" in rewritten
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_rewrite_never_drops_critical_original_terms():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "modelo-local",
                    "message": {
                        "role": "assistant",
                        "content": "requisitos de segurança do notebook corporativo",
                    },
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="modelo-local")

    rewritten = await adapter.rewrite_query("Meu notebook foi furtado")

    assert "furto roubo equipamento corporativo" in rewritten
    assert "requisitos de segurança" in rewritten
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_rewrite_rejects_answer_appended_to_original_question():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "modelo-local",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "quais despesas podem ser reembolsadas Retorno da pergunta: "
                            "Despesas que podem ser reembolsadas incluem seguros e hospitais"
                        ),
                    },
                    "done": True,
                },
            )
        ),
        base_url="http://ollama.test/api/",
    )
    adapter = OllamaLLMAdapter(client=client, model="modelo-local")

    rewritten = await adapter.rewrite_query("Quais despesas podem ser reembolsadas?")

    assert rewritten == "quais despesas podem ser reembolsadas"
    assert "seguros" not in rewritten
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_health_checks_authentication_and_model_access():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": "modelo-teste", "object": "model", "created": 1, "owned_by": "test"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.test/v1/",
    )
    adapter = OpenAICompatibleLLMAdapter(
        client=client,
        api_key="chave-teste",
        model="modelo-teste",
    )

    assert await adapter.is_healthy() is True
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v1/models/modelo-teste"
    assert requests[0].headers["authorization"] == "Bearer chave-teste"
    await adapter.aclose()
    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_health_is_false_when_provider_rejects_key():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {}})),
        base_url="https://api.openai.test/v1/",
    )
    adapter = OpenAICompatibleLLMAdapter(
        client=client,
        api_key="chave-invalida",
        model="modelo-teste",
    )

    assert await adapter.is_healthy() is False
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_reuses_client_and_retries_generation(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(500, json={"error": {"message": "temporário"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Resposta [c1]"}}]},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.test/v1/",
    )
    adapter = OpenAICompatibleLLMAdapter(
        client=client,
        api_key="chave-teste",
        model="modelo-teste",
    )
    monkeypatch.setattr(adapter, "max_retries", 1)
    monkeypatch.setattr("src.infrastructure.llm_adapter.asyncio.sleep", _no_sleep)
    chunks = [Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto")]

    answer, citations = await adapter.generate_answer("Pergunta", chunks)

    assert attempts == 2
    assert answer == "Resposta [c1]"
    assert citations == ["c1"]
    await client.aclose()


async def _no_sleep(delay):
    return None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY não configurada")
async def test_openai_live_health_when_key_is_available():
    adapter = OpenAICompatibleLLMAdapter()
    try:
        assert await adapter.is_healthy() is True
    finally:
        await adapter.aclose()


def test_vector_store_rejects_mismatched_index(tmp_path):
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto"),
    ]
    (tmp_path / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in chunks]),
        encoding="utf-8",
    )
    np.save(tmp_path / "vectors.npy", np.ones((2, 384), dtype=np.float32))
    repository = InMemoryVectorStore(index_dir=tmp_path)

    assert repository.load() is False
    assert repository._is_loaded is False
    assert repository.vectors is None


def test_vector_store_rejects_index_from_another_embedding_model(tmp_path):
    chunks = [Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto")]
    repository = InMemoryVectorStore(index_dir=tmp_path)
    repository.save(chunks, np.ones((1, 384), dtype=np.float32))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_fingerprint"] = "ollama:outro-modelo:retrieval-v1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert InMemoryVectorStore(index_dir=tmp_path).load() is False


def test_vector_store_rejects_index_after_markdown_changes(tmp_path):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    docs_dir.mkdir()
    document = docs_dir / "politica.md"
    document.write_text("# Política\nConteúdo inicial.", encoding="utf-8")
    chunks = [
        Chunk(
            chunk_id="c1",
            source="politica.md",
            section="Política",
            content="Texto",
        )
    ]
    repository = InMemoryVectorStore(index_dir=index_dir, docs_dir=docs_dir)
    repository.save(chunks, np.ones((1, 384), dtype=np.float32))

    assert InMemoryVectorStore(index_dir=index_dir, docs_dir=docs_dir).load() is True

    document.write_text("# Política\nConteúdo alterado.", encoding="utf-8")

    assert InMemoryVectorStore(index_dir=index_dir, docs_dir=docs_dir).load() is False


@pytest.mark.asyncio
async def test_vector_store_reports_embedding_dimension_mismatch(monkeypatch, tmp_path):
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto"),
    ]
    repository = InMemoryVectorStore(index_dir=tmp_path)
    repository.save(chunks, np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(
        "src.infrastructure.vector_store.embedding_service.get_query_vector",
        lambda query: np.ones(3, dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="Dimensão do embedding"):
        await repository.search("pergunta")


@pytest.mark.asyncio
async def test_vector_store_matches_portuguese_inflections_for_reimbursement(tmp_path):
    chunks = [
        Chunk(
            chunk_id="reembolso_c2",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="São consideradas despesas reembolsáveis alimentação e transporte.",
        ),
        Chunk(
            chunk_id="onboarding_c1",
            source="onboarding.md",
            section="Integração",
            content="O colaborador participa da apresentação da empresa.",
        ),
    ]
    embedder = LocalDenseEmbedder(dim=384)
    vectors = embedder.encode_batch(
        [f"{chunk.section}\n{chunk.content}" for chunk in chunks]
    )
    repository = InMemoryVectorStore(index_dir=tmp_path)
    repository.save(chunks, vectors)

    results = await repository.search("Quais despesas podem ser reembolsadas?", top_k=2)

    assert results[0].chunk_id == "reembolso_c2"
    assert results[0].score >= 0.40


@pytest.mark.asyncio
async def test_vector_store_keeps_split_winning_section_together(tmp_path):
    chunks = [
        Chunk(
            chunk_id="reembolso_c2",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="Despesas reembolsáveis: alimentação, transporte e quilometragem.",
        ),
        Chunk(
            chunk_id="outra_secao",
            source="reembolso.md",
            section="Objetivo",
            content="Esta política explica o reembolso de despesas.",
        ),
        Chunk(
            chunk_id="reembolso_c3",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="Hospedagem, cursos e certificações.",
        ),
    ]
    embedder = LocalDenseEmbedder(dim=384)
    repository = InMemoryVectorStore(index_dir=tmp_path)
    repository.save(
        chunks,
        embedder.encode_batch(
            [f"{chunk.section}\n{chunk.content}" for chunk in chunks]
        ),
    )

    results = await repository.search("Quais despesas podem ser reembolsadas?", top_k=2)

    assert [chunk.chunk_id for chunk in results] == ["reembolso_c2", "reembolso_c3"]


@pytest.mark.asyncio
async def test_vector_store_uses_document_name_for_generic_onboarding_query(tmp_path):
    chunks = [
        Chunk(
            chunk_id="onboarding_c1",
            source="onboarding.md",
            section="Primeiros Dias e Acessos",
            content="Boas-vindas, configuração do notebook e plano de 30-60-90 dias.",
        ),
        Chunk(
            chunk_id="onboarding_c2",
            source="onboarding.md",
            section="Benefícios",
            content="Vale refeição e plano de saúde.",
        ),
        Chunk(
            chunk_id="ferias_c1",
            source="ferias.md",
            section="Férias",
            content="Período aquisitivo e descanso remunerado.",
        ),
    ]
    embedder = LocalDenseEmbedder(dim=384)
    repository = InMemoryVectorStore(index_dir=tmp_path)
    repository.save(
        chunks,
        embedder.encode_batch(
            [f"{chunk.source} | {chunk.section}\n{chunk.content}" for chunk in chunks]
        ),
    )

    results = await repository.search("Como é o processo de onboarding?", top_k=2)

    assert results[0].chunk_id == "onboarding_c1"
    assert results[0].score >= 0.40
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_mock_llm_lists_items_across_adjacent_section_chunks():
    chunks = [
        Chunk(
            chunk_id="reembolso_c2",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="- Alimentação em viagens\n- Transporte urbano\n- Quilometragem",
            score=0.80,
        ),
        Chunk(
            chunk_id="reembolso_c3",
            source="reembolso.md",
            section="Despesas Elegíveis",
            content="- Hospedagem\n- Cursos e certificações",
            score=0.75,
        ),
    ]

    answer, citations = await MockLLMAdapter().generate_answer(
        "Quais despesas podem ser reembolsadas?",
        chunks,
    )

    assert citations == ["reembolso_c2", "reembolso_c3"]
    assert "Alimentação em viagens" in answer
    assert "Cursos e certificações" in answer


@pytest.mark.asyncio
async def test_mock_llm_returns_onboarding_overview_from_first_chunk():
    chunks = [
        Chunk(
            chunk_id="onboarding_c1",
            source="onboarding.md",
            section="Primeiros Dias e Acessos",
            content=(
                "Bem-vindo ao time!\n"
                "- Dia 1: configuração do notebook e do e-mail.\n"
                "- Primeira semana: plano de 30-60-90 dias e apresentação da equipe.\n"
                "- Treinamentos obrigatórios nos primeiros 15 dias."
            ),
            score=0.75,
        ),
        Chunk(
            chunk_id="onboarding_c2",
            source="onboarding.md",
            section="Benefícios",
            content="Vale refeição e plano de saúde.",
            score=0.70,
        ),
    ]

    answer, citations = await MockLLMAdapter().generate_answer(
        "Como é o processo de onboarding?",
        chunks,
    )

    assert citations == ["onboarding_c1"]
    assert "configuração do notebook" in answer
    assert "plano de 30-60-90 dias" in answer
    assert "Treinamentos obrigatórios" in answer


@pytest.mark.asyncio
async def test_mock_llm_ranks_relevant_lines_before_limiting_answer():
    chunks = [
        Chunk(
            chunk_id="home_office_c2",
            source="home_office.md",
            section="Auxílio",
            content=(
                "- Há subsídio mensal de energia.\n"
                "- O kit contém teclado e mouse.\n"
                "- Há reembolso de R$ 500,00 para cadeira ergonômica."
            ),
            score=0.75,
        )
    ]

    answer, citations = await MockLLMAdapter().generate_answer(
        "Tem subsídio para comprar cadeira ergonômica?",
        chunks,
    )

    assert citations == ["home_office_c2"]
    assert "R$ 500,00" in answer


@pytest.mark.asyncio
async def test_mock_llm_matches_portuguese_theft_inflection_across_lines():
    chunks = [
        Chunk(
            chunk_id="seguranca_c4",
            source="seguranca.md",
            section="Incidentes",
            content=(
                "- A perda ou furto de notebook deve ser comunicada imediatamente.\n"
                "- Em caso de furto, registre boletim em até 24 horas."
            ),
            score=0.75,
        )
    ]

    answer, citations = await MockLLMAdapter().generate_answer(
        "O que acontece se meu notebook for furtado?",
        chunks,
    )

    assert citations == ["seguranca_c4"]
    assert "24 horas" in answer


@pytest.mark.asyncio
async def test_mock_llm_keeps_items_after_selected_list_introduction():
    chunks = [
        Chunk(
            chunk_id="home_office_c1",
            source="home_office.md",
            section="Modelos de Trabalho",
            content=(
                "A empresa adota três modelos de trabalho:\n"
                "- Presencial: atividades na empresa.\n"
                "- Híbrido: parte da semana em home office.\n"
                "- Remoto total: elegível mediante aprovação."
            ),
            score=0.75,
        )
    ]

    answer, citations = await MockLLMAdapter().generate_answer(
        "Eu consigo trabalhar de casa?",
        chunks,
    )

    assert citations == ["home_office_c1"]
    assert "Híbrido" in answer
    assert "Remoto total" in answer
