"""Telemetria: Logging estruturado, geração de trace_id e medição de latência."""
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Dict, List, Any
import numpy as np

# Configuração de Logger Estruturado
logger = logging.getLogger("agente_suporte_interno")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(
    getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
)


def generate_trace_id() -> str:
    """Gera um identificador único de rastreamento (trace_id)."""
    return str(uuid.uuid4())


@contextmanager
def measure_time(timings: Dict[str, float], node_name: str):
    """Context manager para registrar o tempo de execução de um nó em milissegundos."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        timings[node_name] = round(elapsed_ms, 2)


class MetricsCollector:
    """Coletor simples em memória de métricas operacionais para observabilidade."""

    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.rejected_429_total = 0
        self.latencies_ms: List[float] = []

    def record_request(self, total_latency_ms: float, is_error: bool = False):
        self.total_requests += 1
        if is_error:
            self.total_errors += 1
        self.latencies_ms.append(total_latency_ms)
        # Manter uma janela recente de até 10.000 requisições
        if len(self.latencies_ms) > 10000:
            self.latencies_ms = self.latencies_ms[-10000:]

    def record_rejected_429(self):
        self.rejected_429_total += 1

    def get_summary(self) -> Dict[str, Any]:
        if not self.latencies_ms:
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "rejected_429_total": self.rejected_429_total,
                "error_rate_pct": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }
        
        p50 = float(np.percentile(self.latencies_ms, 50))
        p95 = float(np.percentile(self.latencies_ms, 95))
        p99 = float(np.percentile(self.latencies_ms, 99))
        error_rate = (self.total_errors / self.total_requests) * 100.0 if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "rejected_429_total": self.rejected_429_total,
            "error_rate_pct": round(error_rate, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
        }


metrics_collector = MetricsCollector()
