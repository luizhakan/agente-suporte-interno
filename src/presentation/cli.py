"""Adaptador CLI para interação via terminal com o Agente de Suporte Interno."""
import asyncio
import os
import sys
import time
import warnings
from textwrap import indent

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

from src.domain.models import AgentState
from src.presentation.formatting import evidence_excerpt, format_answer_for_display


# O CLI é uma interface para usuários finais; warnings de dependências ficam nos
# testes/logs da aplicação, não antes da resposta no terminal.
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


def format_cli_result(question: str, state: AgentState) -> str:
    """Apresenta a mesma hierarquia de resposta e evidências usada pela web."""
    cited_chunk_ids = state.get("cited_chunk_ids", [])
    answer = format_answer_for_display(state.get("answer"), cited_chunk_ids)
    lines = ["", "Resposta", "-" * 72, answer]

    chunk_map = {
        chunk.chunk_id: chunk
        for chunk in state.get("retrieved_chunks", [])
    }
    if cited_chunk_ids:
        lines.extend(["", f"Evidências usadas ({len(cited_chunk_ids)})"])
        for citation_number, chunk_id in enumerate(cited_chunk_ids, start=1):
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            excerpt = evidence_excerpt(state.get("answer"), chunk_id, chunk.content)
            lines.extend(
                [
                    f"[{citation_number}] {chunk.section}",
                    f"    Arquivo: {chunk.source} · trecho: {chunk.chunk_id}",
                    indent(excerpt, "    "),
                ]
            )

    failure = state.get("failure")
    status_label = failure.value if hasattr(failure, "value") else failure
    details = [f"trace: {state['trace_id']}"]
    if status_label:
        details.append(f"status: {status_label}")
    details.append(f"score de recuperação: {state.get('evidence_score', 0.0):.3f}")
    total_latency = state.get("timings", {}).get("total")
    if total_latency is not None:
        details.append(f"tempo: {total_latency:.2f} ms")

    if state.get("effective_query") != question:
        lines.extend(["", f"Consulta interpretada: {state.get('effective_query')}"])
    lines.extend(["", "Detalhes técnicos", " · ".join(details), ""])
    return "\n".join(lines)


async def ask_agent(question: str, support_graph):
    """Executa uma pergunta no grafo e exibe a resposta formatada."""
    from src.infrastructure.telemetry import generate_trace_id

    trace_id = generate_trace_id()
    state: AgentState = {
        "trace_id": trace_id,
        "question": question,
        "effective_query": question,
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "evidence_score": 0.0,
        "answer": None,
        "cited_chunk_ids": [],
        "timings": {},
        "failure": None,
    }

    print("Consultando a base interna…")
    started_at = time.perf_counter()
    final_state = await support_graph.ainvoke(state)
    final_state["timings"]["total"] = round(
        (time.perf_counter() - started_at) * 1000.0,
        2,
    )
    print(format_cli_result(question, final_state))


async def main():
    os.environ["LOG_LEVEL"] = os.getenv("CLI_LOG_LEVEL", "WARNING")

    # Imports tardios mantêm o CLI silencioso antes da saída orientada ao usuário.
    from src.application.ingestion import run_ingestion
    from src.bootstrap import support_graph, vector_repository
    from src.config import settings

    # Carrega o índice antes de iniciar
    if not vector_repository.load() and settings.AUTO_INGEST_ON_STARTUP:
        run_ingestion()
        vector_repository.load()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        await ask_agent(question, support_graph)
    else:
        print("=== Agente de Suporte Interno (CLI Interativo) ===")
        print("Digite sua pergunta ou 'sair' para encerrar.\n")
        while True:
            try:
                q = input("Pergunta > ").strip()
                if not q:
                    continue
                if q.lower() in ("sair", "exit", "quit"):
                    break
                await ask_agent(q, support_graph)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    asyncio.run(main())
