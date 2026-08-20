"""Grafo LangGraph determinístico com orquestração de nós, timeouts e fallbacks."""
import asyncio
import re
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, END

from src.config import settings
from src.domain.models import AgentState, FailureKind
from src.domain.sufficiency import check_evidence_sufficiency
from src.infrastructure.telemetry import measure_time, logger
from src.ports.llm_client import LLMClient
from src.ports.retrieval_repository import RetrievalRepository


def build_support_graph(
    retrieval_repo: RetrievalRepository,
    llm: LLMClient,
) -> Any:
    """
    Constrói e compila o StateGraph determinístico do agente de suporte interno
    conforme especificado na Seção 4.1 do documento de arquitetura.
    """
    workflow = StateGraph(AgentState)

    # 1. Nó validate_input
    async def validate_input(state: AgentState) -> Dict[str, Any]:
        timings = dict(state.get("timings", {}))
        with measure_time(timings, "validate_input"):
            q = state["question"].strip()
            if not q or len(q) > settings.MAX_INPUT_LENGTH:
                return {
                    "failure": FailureKind.INVALID_INPUT,
                    "effective_query": q,
                    "timings": timings,
                }
            return {
                "effective_query": q,
                "timings": timings,
            }

    def route_validate_input(state: AgentState) -> Literal["fallback_invalid", "retrieve_context"]:
        if state.get("failure") == FailureKind.INVALID_INPUT:
            return "fallback_invalid"
        return "retrieve_context"

    # 2. Nó retrieve_context
    async def retrieve_context(state: AgentState) -> Dict[str, Any]:
        timings = dict(state.get("timings", {}))
        with measure_time(timings, "retrieve_context"):
            try:
                chunks = await retrieval_repo.search(
                    query=state["effective_query"],
                    top_k=settings.TOP_K,
                )
                _, score_max = check_evidence_sufficiency(
                    chunks=chunks,
                    tau=settings.TAU,
                    delta=settings.DELTA,
                    strong_tau=settings.STRONG_TAU,
                )
                return {
                    "retrieved_chunks": chunks,
                    "evidence_score": score_max,
                    "timings": timings,
                }
            except Exception as e:
                logger.error(f"Erro ao consultar repositório de busca: {e}", exc_info=True)
                return {
                    "failure": FailureKind.SOURCE_UNAVAILABLE,
                    "retrieved_chunks": [],
                    "evidence_score": 0.0,
                    "timings": timings,
                }

    def route_retrieve_context(state: AgentState) -> Literal["fallback_source", "check_evidence"]:
        if state.get("failure") == FailureKind.SOURCE_UNAVAILABLE:
            return "fallback_source"
        return "check_evidence"

    # 3. Nó check_evidence (Decisão de Roteamento)
    def check_evidence(state: AgentState) -> Literal["generate_answer", "rewrite_query", "fallback_no_answer"]:
        chunks = state.get("retrieved_chunks", [])
        if any(chunk.constraint_violation for chunk in chunks):
            logger.info(
                "Consulta recusada porque a similaridade dependia de ignorar "
                "uma restrição semântica da pergunta."
            )
            return "fallback_no_answer"
        is_sufficient, _ = check_evidence_sufficiency(
            chunks=chunks,
            tau=settings.TAU,
            delta=settings.DELTA,
            strong_tau=settings.STRONG_TAU,
        )

        if is_sufficient:
            return "generate_answer"
        
        # Ponto de autonomia: teto rígido de 1 reescrita (Seção 4.2)
        if state.get("rewrite_count", 0) == 0:
            return "rewrite_query"
        
        return "fallback_no_answer"

    # 4. Nó rewrite_query
    async def rewrite_query(state: AgentState) -> Dict[str, Any]:
        timings = dict(state.get("timings", {}))
        with measure_time(timings, "rewrite_query"):
            try:
                rewritten = await asyncio.wait_for(
                    llm.rewrite_query(state["question"]),
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                logger.info(f"Query reescrita: '{state['question']}' -> '{rewritten}'")
                return {
                    "effective_query": rewritten,
                    "rewrite_count": state.get("rewrite_count", 0) + 1,
                    "timings": timings,
                }
            except Exception as e:
                logger.warning(f"Falha ao reescrever query: {e}. Mantendo query original.")
                return {
                    "rewrite_count": state.get("rewrite_count", 0) + 1,
                    "timings": timings,
                }

    # 5. Nó generate_answer
    async def generate_answer(state: AgentState) -> Dict[str, Any]:
        timings = dict(state.get("timings", {}))
        with measure_time(timings, "generate_answer"):
            try:
                chunks = state.get("retrieved_chunks", [])
                answer, cited_ids = await asyncio.wait_for(
                    llm.generate_answer(
                        query=state["effective_query"],
                        chunks=chunks,
                    ),
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                return {
                    "answer": answer,
                    "cited_chunk_ids": cited_ids,
                    "timings": timings,
                }
            except Exception as e:
                logger.error(f"Erro na geração pelo LLM: {e}", exc_info=True)
                return {
                    "failure": FailureKind.LLM_UNAVAILABLE,
                    "timings": timings,
                }

    def route_generate_answer(state: AgentState) -> Literal["fallback_llm", "validate_citations"]:
        if state.get("failure") == FailureKind.LLM_UNAVAILABLE:
            return "fallback_llm"
        return "validate_citations"

    # 6. Nó validate_citations
    async def validate_citations(state: AgentState) -> Dict[str, Any]:
        timings = dict(state.get("timings", {}))
        with measure_time(timings, "validate_citations"):
            retrieved_ids = {c.chunk_id for c in state.get("retrieved_chunks", [])}
            cited_ids = state.get("cited_chunk_ids", [])
            answer = state.get("answer") or ""
            answer_citations = set(re.findall(r"\[([a-zA-Z0-9_-]+)\]", answer))

            # Uma resposta gerada sem ao menos uma citação explícita não é fundamentada.
            if not cited_ids:
                logger.warning("Resposta sem citações válidas detectada.")
                return {
                    "failure": FailureKind.NO_EVIDENCE,
                    "answer": "Não encontrei essa informação na base de políticas e procedimentos.",
                    "cited_chunk_ids": [],
                    "timings": timings,
                }

            if answer_citations != set(cited_ids):
                logger.warning("As citações declaradas divergem das marcações presentes na resposta.")
                return {
                    "failure": FailureKind.NO_EVIDENCE,
                    "answer": "Não encontrei essa informação na base de políticas e procedimentos.",
                    "cited_chunk_ids": [],
                    "timings": timings,
                }

            # Valida se todas as citações pertencem aos chunks recuperados
            for cid in cited_ids:
                if cid not in retrieved_ids or f"[{cid}]" not in answer:
                    logger.warning(
                        f"Citação inválida detectada: {cid} não está no contexto "
                        "ou não aparece explicitamente na resposta."
                    )
                    return {
                        "failure": FailureKind.NO_EVIDENCE,
                        "answer": "Não encontrei essa informação na base de políticas e procedimentos.",
                        "cited_chunk_ids": [],
                        "timings": timings,
                    }

            return {"timings": timings}

    def route_validate_citations(state: AgentState) -> Literal["fallback_no_answer", END]:
        if state.get("failure") == FailureKind.NO_EVIDENCE:
            return "fallback_no_answer"
        return END

    # Nós de Fallback (Seção 4.4)
    async def fallback_invalid(state: AgentState) -> Dict[str, Any]:
        return {
            "failure": FailureKind.INVALID_INPUT,
            "answer": "Por favor, formule uma pergunta válida sobre as políticas e procedimentos da empresa.",
        }

    async def fallback_no_answer(state: AgentState) -> Dict[str, Any]:
        return {
            "failure": FailureKind.NO_EVIDENCE,
            "answer": "Não encontrei essa informação na base de políticas e procedimentos.",
            "cited_chunk_ids": [],
        }

    async def fallback_source(state: AgentState) -> Dict[str, Any]:
        return {
            "failure": FailureKind.SOURCE_UNAVAILABLE,
            "answer": "A base de conhecimento está temporariamente indisponível. Por favor, tente novamente em instantes.",
            "cited_chunk_ids": [],
        }

    async def fallback_llm(state: AgentState) -> Dict[str, Any]:
        return {
            "failure": FailureKind.LLM_UNAVAILABLE,
            "answer": "Não foi possível gerar a resposta agora devido a uma instabilidade temporária.",
            "cited_chunk_ids": [],
        }

    # Adicionar os nós ao grafo
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_citations", validate_citations)
    workflow.add_node("fallback_invalid", fallback_invalid)
    workflow.add_node("fallback_no_answer", fallback_no_answer)
    workflow.add_node("fallback_source", fallback_source)
    workflow.add_node("fallback_llm", fallback_llm)

    # Definir ponto de entrada e arestas
    workflow.set_entry_point("validate_input")

    workflow.add_conditional_edges(
        "validate_input",
        route_validate_input,
        {
            "fallback_invalid": "fallback_invalid",
            "retrieve_context": "retrieve_context",
        },
    )

    workflow.add_conditional_edges(
        "retrieve_context",
        route_retrieve_context,
        {
            "fallback_source": "fallback_source",
            "check_evidence": "check_evidence",
        },
    )

    # check_evidence é puramente condicional e direciona para generate_answer, rewrite_query ou fallback_no_answer
    workflow.add_node("check_evidence", lambda state: state)
    workflow.add_conditional_edges(
        "check_evidence",
        check_evidence,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback_no_answer": "fallback_no_answer",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_context")

    workflow.add_conditional_edges(
        "generate_answer",
        route_generate_answer,
        {
            "fallback_llm": "fallback_llm",
            "validate_citations": "validate_citations",
        },
    )

    workflow.add_conditional_edges(
        "validate_citations",
        route_validate_citations,
        {
            "fallback_no_answer": "fallback_no_answer",
            END: END,
        },
    )

    workflow.add_edge("fallback_invalid", END)
    workflow.add_edge("fallback_no_answer", END)
    workflow.add_edge("fallback_source", END)
    workflow.add_edge("fallback_llm", END)

    return workflow.compile()
