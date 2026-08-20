"""Aplicação FastAPI para exposição do Agente de Suporte Interno via HTTP."""
import asyncio
import math
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from src.bootstrap import llm_client, support_graph, vector_repository
from src.application.ingestion import run_ingestion
from src.config import settings
from src.domain.models import (
    AgentState,
    FailureKind,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from src.infrastructure.telemetry import (
    generate_trace_id,
    logger,
    metrics_collector,
)
from src.infrastructure.embeddings import embedding_service
from src.presentation.formatting import evidence_excerpt, format_answer_for_display


STATIC_DIR = Path(__file__).resolve().parent / "static"
_APPROXIMATE_P50_SECONDS = 2.0
_admission_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
_admission_waiting = 0


def _capacity_error() -> HTTPException:
    metrics_collector.record_rejected_429()
    # Estimativa conservadora: profundidade máxima da fila × p50 observado (~2 s),
    # arredondada para o próximo segundo inteiro conforme o Retry-After HTTP.
    retry_after_seconds = math.ceil(
        settings.MAX_QUEUE_DEPTH * _APPROXIMATE_P50_SECONDS
    )
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            "O serviço está no limite de capacidade. "
            "Tente novamente em instantes."
        ),
        headers={"Retry-After": str(retry_after_seconds)},
    )


async def _acquire_admission_slot() -> None:
    """Reserva uma execução ou rejeita quando a fila HTTP está saturada."""
    global _admission_waiting

    # Não há await entre a leitura e a escrita: em um único event loop por
    # processo, a atualização é atômica de forma cooperativa.
    if _admission_waiting >= settings.MAX_QUEUE_DEPTH:
        raise _capacity_error()
    _admission_waiting += 1

    try:
        await asyncio.wait_for(
            _admission_semaphore.acquire(),
            timeout=settings.ADMISSION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise _capacity_error()
    finally:
        _admission_waiting -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida da aplicação: Carrega o índice na memória no startup."""
    if len(settings.INTERNAL_API_KEY) < 32:
        raise RuntimeError(
            "INTERNAL_API_KEY deve ter pelo menos 32 caracteres antes de iniciar a API."
        )
    if "*" in settings.CORS_ALLOWED_ORIGINS:
        raise RuntimeError("CORS_ALLOWED_ORIGINS não pode conter wildcard (*).")
    logger.info("Iniciando API do Agente de Suporte Interno...")
    is_loaded = vector_repository.load()
    if not is_loaded and settings.AUTO_INGEST_ON_STARTUP:
        logger.info(
            "Índice ausente ou incompatível; iniciando reindexação com "
            f"{embedding_service.provider_name}/{embedding_service.model_name}."
        )
        await asyncio.to_thread(run_ingestion)
        is_loaded = vector_repository.load()
    if not is_loaded:
        raise RuntimeError(
            "O índice vetorial não pôde ser carregado. Execute 'make ingest' "
            "com o provedor de embeddings configurado."
        )
    try:
        yield
    finally:
        await llm_client.aclose()
        await asyncio.to_thread(embedding_service.close)
        logger.info("Encerrando API do Agente de Suporte Interno.")


app = FastAPI(
    title="Agente de Suporte Interno - API",
    description="Assistente RAG determinístico para dúvidas de colaboradores sobre políticas internas.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Internal-API-Key"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)


async def require_internal_api_key(
    provided_key: Optional[str] = Security(api_key_header),
) -> None:
    """Exige uma credencial interna usando comparação resistente a timing attacks."""
    configured_key = settings.INTERNAL_API_KEY
    if len(configured_key) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Autenticação interna não configurada.",
        )
    if not provided_key or not secrets.compare_digest(provided_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial interna inválida ou ausente.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@app.get("/", include_in_schema=False)
async def web_interface():
    """Entrega a interface web estática pelo mesmo servidor da API."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Observabilidade"])
async def health_check():
    """Verifica a saúde do serviço, repositório de busca e conexão com LLM."""
    repo_healthy = await vector_repository.is_healthy()
    llm_healthy = await llm_client.is_healthy()
    embedding_healthy = await asyncio.to_thread(embedding_service.is_healthy)
    is_healthy = repo_healthy and llm_healthy and embedding_healthy

    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "llm": {
                "provider": llm_client.provider_name,
                "model": llm_client.model_name,
            },
            "embeddings": {
                "provider": embedding_service.provider_name,
                "model": embedding_service.model_name,
            },
            "components": {
                "vector_store": "ok" if repo_healthy else "error",
                "llm_client": "ok" if llm_healthy else "degraded",
                "embedding_client": "ok" if embedding_healthy else "degraded",
            },
        },
    )


@app.get(
    "/metrics",
    tags=["Observabilidade"],
    dependencies=[Depends(require_internal_api_key)],
)
async def get_metrics():
    """Retorna métricas de latência p50/p95, total de requisições e taxa de erro."""
    return metrics_collector.get_summary()


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Atendimento"],
    dependencies=[Depends(require_internal_api_key)],
    responses={
        200: {"description": "Resposta gerada ou informação não encontrada na base"},
        400: {"description": "Entrada inválida ou vazia"},
        429: {"description": "Capacidade de execução e fila esgotadas"},
        503: {"description": "Serviço ou fonte temporariamente indisponível"},
    },
)
async def process_query(request: QueryRequest):
    """
    Processa a dúvida do colaborador executando o fluxo determinístico LangGraph.
    """
    await _acquire_admission_slot()
    try:
        return await _execute_query(request)
    finally:
        _admission_semaphore.release()


async def _execute_query(request: QueryRequest):
    """Executa uma consulta já admitida pelo controle de capacidade HTTP."""
    start_time = time.perf_counter()
    trace_id = request.trace_id or generate_trace_id()

    # Monta o estado inicial imutável
    initial_state: AgentState = {
        "trace_id": trace_id,
        "question": request.question,
        "effective_query": request.question,
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    # Invoca o grafo
    try:
        final_state: AgentState = await support_graph.ainvoke(initial_state)
    except Exception as e:
        logger.critical(f"Erro fatal na execução do grafo [trace_id={trace_id}]: {e}", exc_info=True)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_collector.record_request(total_latency_ms, is_error=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar requisição.",
        )

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    final_state["timings"]["total"] = round(total_latency_ms, 2)

    failure = final_state.get("failure")
    is_error = failure in (FailureKind.SOURCE_UNAVAILABLE, FailureKind.LLM_UNAVAILABLE)
    metrics_collector.record_request(total_latency_ms, is_error=is_error)

    # Extrai metadados dos chunks citados
    sources: list[SourceCitation] = []
    chunk_map = {c.chunk_id: c for c in final_state.get("retrieved_chunks", [])}
    cited_chunk_ids = final_state.get("cited_chunk_ids", [])
    raw_answer = final_state.get("answer")
    for citation_number, cid in enumerate(cited_chunk_ids, start=1):
        if cid in chunk_map:
            c = chunk_map[cid]
            sources.append(
                SourceCitation(
                    citation_number=citation_number,
                    chunk_id=c.chunk_id,
                    source=c.source,
                    section=c.section,
                    excerpt=evidence_excerpt(raw_answer, cid, c.content),
                )
            )

    response_payload = QueryResponse(
        trace_id=final_state["trace_id"],
        question=final_state["question"],
        effective_query=final_state["effective_query"],
        answer=format_answer_for_display(raw_answer, cited_chunk_ids),
        cited_chunk_ids=cited_chunk_ids,
        sources=sources,
        evidence_score=final_state.get("evidence_score", 0.0),
        timings=final_state.get("timings", {}),
        failure=failure,
    )

    # Mapeamento estrito de erros arquiteturais para códigos HTTP (Seção 4.4)
    if failure == FailureKind.INVALID_INPUT:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=response_payload.model_dump(),
        )
    elif failure == FailureKind.SOURCE_UNAVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_payload.model_dump(),
        )
    elif failure == FailureKind.LLM_UNAVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_payload.model_dump(),
        )

    # Sucesso ou NO_EVIDENCE retornam HTTP 200
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_payload.model_dump(),
    )
