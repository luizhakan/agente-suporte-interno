"""Modelos de dados de domínio e estado do agente."""
from enum import Enum
from typing import Optional, TypedDict
from pydantic import BaseModel, Field


class FailureKind(str, Enum):
    """Tipos de falha mapeados arquiteturalmente."""
    INVALID_INPUT = "INVALID_INPUT"
    NO_EVIDENCE = "NO_EVIDENCE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"


class ChunkKind(str, Enum):
    """Papel do trecho dentro da base indexada."""
    CONTENT = "content"
    CATALOG = "catalog"


class Chunk(BaseModel):
    """Representa um fragmento de texto indexado."""
    chunk_id: str
    source: str
    section: str
    content: str
    kind: ChunkKind = ChunkKind.CONTENT
    retrieval_content: Optional[str] = None
    score: float = 0.0
    constraint_violation: bool = False
    catalog_match: bool = False


class QueryRequest(BaseModel):
    """Requisição de pergunta enviada pelo usuário."""
    question: str = Field(..., description="Pergunta do colaborador")
    trace_id: Optional[str] = Field(default=None, description="Identificador único para rastreamento")


class SourceCitation(BaseModel):
    """Metadados da fonte citada."""
    citation_number: int
    chunk_id: str
    source: str
    section: str
    excerpt: str


class QueryResponse(BaseModel):
    """Resposta estruturada gerada pelo agente."""
    trace_id: str
    question: str
    effective_query: str
    answer: Optional[str] = None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    evidence_score: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)
    failure: Optional[FailureKind] = None


class AgentState(TypedDict):
    """Estado imutável da execução do grafo LangGraph (Seção 4.3)."""
    trace_id: str
    question: str
    effective_query: str
    rewrite_count: int
    retrieved_chunks: list[Chunk]
    evidence_score: float
    answer: Optional[str]
    cited_chunk_ids: list[str]
    timings: dict[str, float]
    failure: Optional[FailureKind]
