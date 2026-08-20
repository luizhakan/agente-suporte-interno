"""Script de avaliação automatizada de latência (Seção 6) e concorrência/escalabilidade (Seção 7)."""
import argparse
import asyncio
import csv
import time
from pathlib import Path
from typing import Dict, Any, Sequence
import numpy as np

from src.application.ingestion import run_ingestion
from src.bootstrap import llm_client, support_graph, vector_repository
from src.config import settings
from src.domain.models import AgentState, FailureKind
from src.infrastructure.embeddings import embedding_service
from src.infrastructure.telemetry import generate_trace_id


# 26 perguntas de avaliação divididas em 5 grupos
EVALUATION_DATASET = [
    # Grupo 1: Resposta Direta
    {"id": "q01", "group": "direta", "question": "Qual o limite de valor para reembolso de almoço e jantar em viagens?", "expected_fail": None, "expected_chunk": "reembolso_c2", "expected_text": "R$ 120,00"},
    {"id": "q02", "group": "direta", "question": "Em quantos períodos posso fracionar minhas férias?", "expected_fail": None, "expected_chunk": "ferias_c2", "expected_text": "até 3 períodos"},
    {"id": "q03", "group": "direta", "question": "Qual o valor da ajuda de custo mensal de home office?", "expected_fail": None, "expected_chunk": "home_office_c2", "expected_text": "R$ 150,00"},
    {"id": "q04", "group": "direta", "question": "Quantos caracteres no mínimo deve ter a senha corporativa?", "expected_fail": None, "expected_chunk": "seguranca_informacao_c1", "expected_text": "12 caracteres"},
    {"id": "q05", "group": "direta", "question": "Qual o prazo para envio de atestado médico ao RH?", "expected_fail": None, "expected_chunk": "onboarding_c4", "expected_text": "48 horas"},

    # Grupo 2: Paráfrase e Linguagem Coloquial
    {"id": "q06", "group": "parafrase", "question": "Posso vender 10 dias de descanso remunerado?", "expected_fail": None, "expected_chunk": "ferias_c3", "expected_text": "venda de 10 dias"},
    {"id": "q07", "group": "parafrase", "question": "Quanto a firma devolve se eu rodar com meu próprio carro a trabalho?", "expected_fail": None, "expected_chunk": "reembolso_c2", "expected_text": "R$ 1,20"},
    {"id": "q08", "group": "parafrase", "question": "Tem subsídio para comprar cadeira ergonômica?", "expected_fail": None, "expected_chunk": "home_office_c2", "expected_text": "R$ 500,00"},
    {"id": "q09", "group": "parafrase", "question": "O que acontece se meu notebook for furtado?", "expected_fail": None, "expected_chunk": "seguranca_informacao_c4", "expected_text": "24 horas"},
    {"id": "q10", "group": "parafrase", "question": "Quando cai o vale refeição no cartão flexível?", "expected_fail": None, "expected_chunk": "onboarding_c2", "expected_text": "dia 1º"},

    # Grupo 3: Fora da Base (Sem Evidência)
    {"id": "q11", "group": "fora_da_base", "question": "Qual a receita do bolo de cenoura com cobertura de chocolate?", "expected_fail": FailureKind.NO_EVIDENCE},
    {"id": "q12", "group": "fora_da_base", "question": "A empresa oferece empréstimo consignado com desconto em folha?", "expected_fail": FailureKind.NO_EVIDENCE},
    {"id": "q13", "group": "fora_da_base", "question": "Qual o modelo do carro da diretoria?", "expected_fail": FailureKind.NO_EVIDENCE},
    {"id": "q14", "group": "fora_da_base", "question": "Posso levar meu cachorro para trabalhar no escritório todo dia?", "expected_fail": FailureKind.NO_EVIDENCE},
    {"id": "q15", "group": "fora_da_base", "question": "Como configurar o cluster Kubernetes de produção?", "expected_fail": FailureKind.NO_EVIDENCE},

    # Grupo 4: Casos de Borda e Validação
    {"id": "q16", "group": "borda", "question": "   ", "expected_fail": FailureKind.INVALID_INPUT},
    {"id": "q17", "group": "borda", "question": "a" * 1200, "expected_fail": FailureKind.INVALID_INPUT},
    {"id": "q18", "group": "borda", "question": "Como pedir reembolso de VR?", "expected_fail": FailureKind.NO_EVIDENCE},
    {"id": "q19", "group": "borda", "question": "MFA é obrigatório para todos?", "expected_fail": None, "expected_chunk": "seguranca_informacao_c1", "expected_text": "obrigatoriamente"},
    {"id": "q20", "group": "borda", "question": "Quantos dias antes posso adiantar o décimo terceiro nas férias?", "expected_fail": None, "expected_chunk": "ferias_c4", "expected_text": "mês de janeiro"},
    {"id": "q21", "group": "parafrase", "question": "Consigo trabalhar em meu lar?", "expected_fail": None, "expected_chunk": "home_office_c1", "expected_text": "Híbrido"},
    {"id": "q22", "group": "parafrase", "question": "Consigo trabalhar em minha humilde residencia?", "expected_fail": None, "expected_chunk": "home_office_c1", "expected_text": "Híbrido"},
    {"id": "q23", "group": "fora_da_base", "question": "Posso pedir reembolso de um videogame?", "expected_fail": FailureKind.NO_EVIDENCE},

    # Grupo 5: Descoberta dinâmica do escopo da base
    {"id": "q24", "group": "escopo", "question": "O que vocês fazem?", "expected_fail": None, "expected_chunk": "knowledge_base_catalog_c1", "expected_text": "Política de Férias"},
    {"id": "q25", "group": "escopo", "question": "Sobre quais assuntos posso perguntar?", "expected_fail": None, "expected_chunk": "knowledge_base_catalog_c1", "expected_text": "Política de Segurança da Informação"},
    {"id": "q26", "group": "fora_da_base", "question": "O que a empresa fabrica?", "expected_fail": FailureKind.NO_EVIDENCE},
]


BATTERIES = {
    "mock": {
        "llm_provider": "mock",
        "embedding_provider": "dense",
        "concurrency_levels": (1, 5, 10, 20, 40),
    },
    "ollama": {
        "llm_provider": "ollama",
        "embedding_provider": "ollama",
        "concurrency_levels": (1, 5, 10),
    },
}


def provider_label() -> str:
    """Identifica inequivocamente os adaptadores medidos em cada linha."""
    return (
        f"{llm_client.provider_name}:{llm_client.model_name}"
        f"+{embedding_service.provider_name}:{embedding_service.model_name}"
    )


def validate_battery_configuration(battery: str) -> None:
    """Impede que uma execução seja salva sob o nome de outro provedor."""
    expected = BATTERIES[battery]
    actual = (llm_client.provider_name, embedding_service.provider_name)
    required = (expected["llm_provider"], expected["embedding_provider"])
    if actual != required:
        raise RuntimeError(
            f"Bateria {battery!r} exige provedores {required}, mas recebeu {actual}."
        )


async def run_single_query(question: str) -> Dict[str, Any]:
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
    start = time.perf_counter()
    res = await support_graph.ainvoke(state)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "failure": res.get("failure"),
        "evidence_score": res.get("evidence_score", 0.0),
        "rewrite_count": res.get("rewrite_count", 0),
        "citations_count": len(res.get("cited_chunk_ids", [])),
        "cited_chunk_ids": res.get("cited_chunk_ids", []),
        "answer": res.get("answer") or "",
    }


async def run_latency_benchmark(output_path: Path, provider: str):
    """Executa a bateria funcional e de latência em três rodadas."""
    print("Iniciando Bateria de Latência (26 perguntas × 3 rodadas)...")
    results = []

    for round_idx in range(1, 4):
        for item in EVALUATION_DATASET:
            out = await run_single_query(item["question"])
            actual_failure = out["failure"]
            expected_failure = item["expected_fail"]
            expected_chunk = item.get("expected_chunk")
            expected_text = item.get("expected_text")
            passed = (
                actual_failure == expected_failure
                and (not expected_chunk or expected_chunk in out["cited_chunk_ids"])
                and (not expected_text or expected_text.casefold() in out["answer"].casefold())
            )
            results.append({
                "provider": provider,
                "round": round_idx,
                "question_id": item["id"],
                "group": item["group"],
                "latency_ms": round(out["elapsed_ms"], 2),
                "failure": actual_failure.value if actual_failure else "NONE",
                "expected_failure": expected_failure.value if expected_failure else "NONE",
                "passed": passed,
                "evidence_score": round(out["evidence_score"], 3),
                "rewrite_count": out["rewrite_count"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)

    latencies = [r["latency_ms"] for r in results]
    passed = sum(1 for result in results if result["passed"])
    print(f"Latência concluída! Salvo em: {output_path}")
    print(f"p50: {np.percentile(latencies, 50):.2f}ms | p95: {np.percentile(latencies, 95):.2f}ms | p99: {np.percentile(latencies, 99):.2f}ms\n")
    print(f"Resultado funcional: {passed}/{len(results)} casos com resposta e fontes esperadas.\n")


async def run_scale_benchmark(
    output_path: Path,
    provider: str,
    concurrency_levels: Sequence[int],
    total_requests: int = 100,
):
    """Executa a bateria de escala nos níveis definidos para cada provedor."""
    print("Iniciando Bateria de Escala e Concorrência...")
    scale_rows = []

    for concurrency in concurrency_levels:
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        errors = 0

        async def worker(q: str):
            nonlocal errors
            async with semaphore:
                try:
                    out = await run_single_query(q)
                    latencies.append(out["elapsed_ms"])
                except Exception:
                    errors += 1

        test_questions = [item["question"] for item in EVALUATION_DATASET]
        tasks = [worker(test_questions[i % len(test_questions)]) for i in range(total_requests)]

        start_time = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time_s = time.perf_counter() - start_time
        throughput_rps = total_requests / total_time_s

        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        err_rate = (errors / total_requests) * 100.0

        scale_rows.append({
            "provider": provider,
            "concurrency": concurrency,
            "total_requests": total_requests,
            "total_time_s": round(total_time_s, 3),
            "throughput_rps": round(throughput_rps, 1),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "error_rate_pct": err_rate,
        })
        print(f"Concorrência: {concurrency:2d} | Throughput: {throughput_rps:6.1f} req/s | p95: {p95:5.2f}ms | Erros: {err_rate}%")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(scale_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(scale_rows)

    print(f"Escala concluída! Salvo em: {output_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera evidências funcionais, de latência e de escala por provedor.",
    )
    parser.add_argument("--battery", choices=sorted(BATTERIES), required=True)
    parser.add_argument(
        "--scale-requests",
        type=int,
        default=100,
        help="Total de requisições por nível de concorrência (padrão: 100).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.scale_requests <= 0:
        raise ValueError("--scale-requests deve ser positivo.")
    validate_battery_configuration(args.battery)
    if not vector_repository.load():
        run_ingestion()
        if not vector_repository.load():
            raise RuntimeError("Não foi possível carregar o índice para a avaliação.")

    provider = provider_label()
    battery = BATTERIES[args.battery]
    evidence_dir = settings.EVIDENCE_DIR
    await run_latency_benchmark(
        evidence_dir / f"latency_{args.battery}.csv",
        provider,
    )
    await run_scale_benchmark(
        evidence_dir / f"scale_{args.battery}.csv",
        provider,
        battery["concurrency_levels"],
        total_requests=args.scale_requests,
    )


if __name__ == "__main__":
    asyncio.run(main())
