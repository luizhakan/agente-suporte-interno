"""Testes unitários e de integração para o Grafo LangGraph determinístico."""
import asyncio
import pytest
from src.application.graph import build_support_graph
from src.config import settings
from src.domain.models import AgentState, Chunk, FailureKind
from src.ports.llm_client import LLMClient
from src.ports.retrieval_repository import RetrievalRepository


class StubRetrievalRepo(RetrievalRepository):
    def __init__(self, chunks=None, should_fail=False):
        self.chunks = chunks or []
        self.should_fail = should_fail

    async def search(self, query: str, top_k: int = 4):
        if self.should_fail:
            raise ConnectionError("Índice indisponível")
        return self.chunks

    async def is_healthy(self):
        return not self.should_fail


class StubLLMClient(LLMClient):
    def __init__(self, answer="Resposta teste [c1]", cited_ids=None, should_fail=False, delay=0):
        self.answer = answer
        self.cited_ids = ["c1"] if cited_ids is None else cited_ids
        self.should_fail = should_fail
        self.delay = delay
        self.rewrite_calls = 0

    async def generate_answer(self, query: str, chunks: list[Chunk]):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise TimeoutError("LLM timeout")
        return self.answer, self.cited_ids

    async def rewrite_query(self, query: str):
        self.rewrite_calls += 1
        return f"reescrita: {query}"

    async def is_healthy(self):
        return not self.should_fail


@pytest.mark.asyncio
async def test_graph_valid_flow():
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.80),
        Chunk(chunk_id="c2", source="doc.md", section="S2", content="Texto 2", score=0.70),
    ]
    repo = StubRetrievalRepo(chunks=chunks)
    llm = StubLLMClient(answer="Resposta com sucesso [c1]", cited_ids=["c1"])
    graph = build_support_graph(retrieval_repo=repo, llm=llm)

    state: AgentState = {
        "trace_id": "test-1",
        "question": "Como funciona?",
        "effective_query": "Como funciona?",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)
    assert result["failure"] is None
    assert "Resposta com sucesso" in result["answer"]
    assert result["cited_chunk_ids"] == ["c1"]
    assert result["rewrite_count"] == 0


@pytest.mark.asyncio
async def test_graph_rejects_retrieval_that_ignored_query_constraint():
    chunks = [
        Chunk(
            chunk_id="c1",
            source="trabalho.md",
            section="Escritório",
            content="A empresa possui trabalho presencial.",
            score=0.50,
            constraint_violation=True,
        )
    ]
    repo = StubRetrievalRepo(chunks=chunks)
    llm = StubLLMClient()
    graph = build_support_graph(retrieval_repo=repo, llm=llm)
    state: AgentState = {
        "trace_id": "test-semantic-constraint",
        "question": "Posso levar um item não mencionado ao escritório?",
        "effective_query": "Posso levar um item não mencionado ao escritório?",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)

    assert result["failure"] == FailureKind.NO_EVIDENCE
    assert result["rewrite_count"] == 0
    assert llm.rewrite_calls == 0


@pytest.mark.asyncio
async def test_graph_invalid_input_empty():
    repo = StubRetrievalRepo()
    llm = StubLLMClient()
    graph = build_support_graph(retrieval_repo=repo, llm=llm)

    state: AgentState = {
        "trace_id": "test-invalid",
        "question": "   ",
        "effective_query": "   ",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)
    assert result["failure"] == FailureKind.INVALID_INPUT


@pytest.mark.asyncio
async def test_graph_repository_failure():
    repo = StubRetrievalRepo(should_fail=True)
    llm = StubLLMClient()
    graph = build_support_graph(retrieval_repo=repo, llm=llm)

    state: AgentState = {
        "trace_id": "test-source-fail",
        "question": "Pergunta válida",
        "effective_query": "Pergunta válida",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)
    assert result["failure"] == FailureKind.SOURCE_UNAVAILABLE


@pytest.mark.asyncio
async def test_graph_llm_failure():
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.85),
        Chunk(chunk_id="c2", source="doc.md", section="S2", content="Texto 2", score=0.75),
    ]
    repo = StubRetrievalRepo(chunks=chunks)
    llm = StubLLMClient(should_fail=True)
    graph = build_support_graph(retrieval_repo=repo, llm=llm)

    state: AgentState = {
        "trace_id": "test-llm-fail",
        "question": "Pergunta válida",
        "effective_query": "Pergunta válida",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)
    assert result["failure"] == FailureKind.LLM_UNAVAILABLE


@pytest.mark.asyncio
async def test_graph_hallucinated_citation_rejected():
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.85),
        Chunk(chunk_id="c2", source="doc.md", section="S2", content="Texto 2", score=0.75),
    ]
    repo = StubRetrievalRepo(chunks=chunks)
    # LLM cita chunk inexistente 'chunk_alucinado'
    llm = StubLLMClient(answer="Resposta falsa", cited_ids=["chunk_alucinado"])
    graph = build_support_graph(retrieval_repo=repo, llm=llm)

    state: AgentState = {
        "trace_id": "test-citation-validation",
        "question": "Pergunta válida",
        "effective_query": "Pergunta válida",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)
    assert result["failure"] == FailureKind.NO_EVIDENCE
    assert "Não encontrei essa informação" in result["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "cited_ids"),
    [
        ("Resposta sem fonte", []),
        ("Resposta que omite a marcação da fonte", ["c1"]),
        ("Resposta usa [c1], mas declara outra fonte", ["c2"]),
    ],
)
async def test_graph_rejects_missing_explicit_citation(answer, cited_ids):
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.85),
        Chunk(chunk_id="c2", source="doc.md", section="S2", content="Texto 2", score=0.75),
    ]
    graph = build_support_graph(
        retrieval_repo=StubRetrievalRepo(chunks=chunks),
        llm=StubLLMClient(answer=answer, cited_ids=cited_ids),
    )
    state: AgentState = {
        "trace_id": "test-missing-citation",
        "question": "Pergunta válida",
        "effective_query": "Pergunta válida",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)

    assert result["failure"] == FailureKind.NO_EVIDENCE
    assert result["cited_chunk_ids"] == []


@pytest.mark.asyncio
async def test_graph_ignores_non_citation_brackets_in_grounded_answer():
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.85),
    ]
    answer = "O campo [opcional] pode ficar vazio conforme a política. [c1]"
    graph = build_support_graph(
        retrieval_repo=StubRetrievalRepo(chunks=chunks),
        llm=StubLLMClient(answer=answer, cited_ids=["c1"]),
    )
    state: AgentState = {
        "trace_id": "test-editorial-brackets",
        "question": "O campo precisa ser preenchido?",
        "effective_query": "O campo precisa ser preenchido?",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)

    assert result["failure"] is None
    assert result["answer"] == answer
    assert result["cited_chunk_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_graph_enforces_total_llm_timeout(monkeypatch):
    chunks = [
        Chunk(chunk_id="c1", source="doc.md", section="S1", content="Texto 1", score=0.85),
        Chunk(chunk_id="c2", source="doc.md", section="S2", content="Texto 2", score=0.75),
    ]
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 0.001)
    graph = build_support_graph(
        retrieval_repo=StubRetrievalRepo(chunks=chunks),
        llm=StubLLMClient(delay=0.05),
    )
    state: AgentState = {
        "trace_id": "test-timeout",
        "question": "Pergunta válida",
        "effective_query": "Pergunta válida",
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    result = await graph.ainvoke(state)

    assert result["failure"] == FailureKind.LLM_UNAVAILABLE
