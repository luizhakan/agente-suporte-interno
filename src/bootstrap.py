"""Composition root: conecta portas de aplicação aos adaptadores de infraestrutura."""
from src.application.graph import build_support_graph
from src.infrastructure.llm_adapter import get_llm_client
from src.infrastructure.vector_store import vector_repository


llm_client = get_llm_client()
support_graph = build_support_graph(
    retrieval_repo=vector_repository,
    llm=llm_client,
)


__all__ = ["llm_client", "support_graph", "vector_repository"]
