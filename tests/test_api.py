"""Testes de integração da API FastAPI e mapeamento de status HTTP (Seção 4.4)."""
import pytest
from httpx import ASGITransport, AsyncClient
from src.bootstrap import vector_repository
from src.config import settings
from src.domain.models import Chunk
from src.presentation.api import app, lifespan


def auth_headers():
    return {"X-Internal-API-Key": settings.INTERNAL_API_KEY}


@pytest.fixture(autouse=True)
def setup_mock_repository(monkeypatch):
    """Configura dados de teste no repositório vetorial em memória."""
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "test-internal-api-key-with-32-chars")
    mock_chunks = [
        Chunk(
            chunk_id="ferias_c1",
            source="ferias.md",
            section="Elegibilidade",
            content="Todo colaborador CLT tem direito a 30 dias de ferias.",
            score=0.85,
        ),
        Chunk(
            chunk_id="ferias_c2",
            source="ferias.md",
            section="Regras",
            content="As ferias podem ser fracionadas em ate 3 periodos.",
            score=0.75,
        ),
    ]
    import numpy as np
    mock_vectors = np.ones((2, 384), dtype=np.float32)
    vector_repository.chunks = mock_chunks
    vector_repository.vectors = mock_vectors
    vector_repository._is_loaded = True


@pytest.mark.asyncio
async def test_api_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["vector_store"] == "ok"
        assert data["components"]["embedding_client"] == "ok"
        assert data["embeddings"] == {
            "provider": "dense",
            "model": "hashing-ngram-384",
        }
        assert data["llm"] == {
            "provider": "mock",
            "model": "deterministic-mock",
        }


@pytest.mark.asyncio
async def test_web_interface_and_static_assets_are_served():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        stylesheet = await client.get("/static/styles.css")
        script = await client.get("/static/app.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Atende — Suporte interno" in page.text
    assert "/api/v1/query" in script.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert "sessionStorage" in script.text


@pytest.mark.asyncio
async def test_api_query_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "Como funciona o fracionamento de férias?"},
            headers=auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] is not None
        assert "ferias_c1" in data["cited_chunk_ids"] or "ferias_c2" in data["cited_chunk_ids"]
        assert len(data["sources"]) > 0
        assert data["sources"][0]["citation_number"] == 1
        assert data["sources"][0]["excerpt"]
        assert data["sources"][0]["chunk_id"] not in data["answer"]
        assert "[1]" in data["answer"]
        assert "total" in data["timings"]


@pytest.mark.asyncio
async def test_api_query_invalid_input():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "   "},
            headers=auth_headers(),
        )
        # Conforme a arquitetura: INVALID_INPUT deve retornar HTTP 400
        assert response.status_code == 400
        data = response.json()
        assert data["failure"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_api_metrics_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics", headers=auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "p50_latency_ms" in data


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"X-Internal-API-Key": "incorreta"}])
async def test_api_rejects_missing_or_invalid_credentials(headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "Como funcionam as férias?"},
            headers=headers,
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


@pytest.mark.asyncio
async def test_cors_allows_configured_origin_and_rejects_unknown_origin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        allowed = await client.options(
            "/api/v1/query",
            headers={
                "Origin": settings.CORS_ALLOWED_ORIGINS[0],
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Internal-API-Key,Content-Type",
            },
        )
        rejected = await client.options(
            "/api/v1/query",
            headers={
                "Origin": "https://origem-maliciosa.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == settings.CORS_ALLOWED_ORIGINS[0]
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


@pytest.mark.asyncio
async def test_api_startup_rejects_short_internal_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "curta")

    with pytest.raises(RuntimeError, match="pelo menos 32 caracteres"):
        async with lifespan(app):
            pass


@pytest.mark.asyncio
async def test_api_startup_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setattr(settings, "CORS_ALLOWED_ORIGINS", ["*"])

    with pytest.raises(RuntimeError, match="wildcard"):
        async with lifespan(app):
            pass
