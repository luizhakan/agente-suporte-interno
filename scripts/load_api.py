"""Carga HTTP assíncrona para medir o controle de admissão da API."""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import time
from pathlib import Path
from typing import Sequence

import httpx

from scripts.evaluate import EVALUATION_DATASET
from src.domain.models import FailureKind


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Calcula percentil com interpolação linear sem dependência adicional."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


async def identify_provider(client: httpx.AsyncClient) -> str:
    response = await client.get("/health")
    response.raise_for_status()
    health = response.json()
    return (
        f"{health['llm']['provider']}:{health['llm']['model']}"
        f"+{health['embeddings']['provider']}:{health['embeddings']['model']}"
    )


async def run_level(
    client: httpx.AsyncClient,
    api_key: str,
    provider: str,
    concurrency: int,
    total_requests: int,
) -> dict[str, object]:
    client_limit = asyncio.Semaphore(concurrency)
    admitted_latencies: list[float] = []
    rejected_429 = 0
    other_errors = 0
    # Entradas inválidas são excluídas porque mediriam validação HTTP, não a fila
    # de inferência. As demais perguntas são as mesmas da bateria funcional.
    questions = [
        item["question"]
        for item in EVALUATION_DATASET
        if item["expected_fail"] != FailureKind.INVALID_INPUT
    ]

    async def request_once(index: int) -> None:
        nonlocal rejected_429, other_errors
        async with client_limit:
            started_at = time.perf_counter()
            try:
                response = await client.post(
                    "/api/v1/query",
                    headers={"X-Internal-API-Key": api_key},
                    json={"question": questions[index % len(questions)]},
                )
            except httpx.HTTPError:
                other_errors += 1
                return

            latency_ms = (time.perf_counter() - started_at) * 1000.0
            if response.status_code == 429:
                rejected_429 += 1
            elif response.status_code == 200:
                admitted_latencies.append(latency_ms)
            else:
                other_errors += 1

    started_at = time.perf_counter()
    await asyncio.gather(*(request_once(i) for i in range(total_requests)))
    total_time_s = time.perf_counter() - started_at
    admitted = len(admitted_latencies)

    return {
        "provider": provider,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "admitted": admitted,
        "rejected_429_pct": round(rejected_429 / total_requests * 100.0, 2),
        "throughput_rps": round(admitted / total_time_s, 2),
        "p50_ms": round(percentile(admitted_latencies, 0.50), 2),
        "p95_ms": round(percentile(admitted_latencies, 0.95), 2),
        "error_rate_pct": round(other_errors / total_requests * 100.0, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede escala e rejeições HTTP 429 da API em execução.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/scale_ollama_api.csv"),
    )
    parser.add_argument("--requests-per-level", type=int, default=50)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10])
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    api_key = os.getenv("INTERNAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("INTERNAL_API_KEY deve estar definido no ambiente.")
    if args.requests_per_level <= 0 or any(level <= 0 for level in args.concurrency):
        raise ValueError("Requisições e níveis de concorrência devem ser positivos.")

    timeout = httpx.Timeout(90.0)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeout,
    ) as client:
        provider = await identify_provider(client)
        rows = []
        for concurrency in args.concurrency:
            row = await run_level(
                client,
                api_key,
                provider,
                concurrency,
                args.requests_per_level,
            )
            rows.append(row)
            print(
                f"C={concurrency}: admitidas={row['admitted']}/"
                f"{row['total_requests']}, 429={row['rejected_429_pct']}%, "
                f"throughput={row['throughput_rps']} req/s, "
                f"p95={row['p95_ms']} ms"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evidência salva em {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
