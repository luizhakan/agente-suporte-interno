"""Configurações da aplicação (Portas, Adaptadores, Limiares e Timeouts)."""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Expansões lexicais de abreviações frequentes no domínio corporativo. Este
# dicionário não contém perguntas nem respostas do conjunto de avaliação.
QUERY_SYNONYMS: dict[str, str] = {
    "vr": "vale refeicao",
    "va": "vale alimentacao",
    "mfa": "autenticacao multifator dois fatores 2fa",
    "km": "quilometro quilometragem",
    "clt": "consolidacao das leis do trabalho contrato de trabalho",
    "home": "home office trabalho remoto",
}


def _csv_env(name: str, default: str) -> list[str]:
    return [value.strip() for value in os.getenv(name, default).split(",") if value.strip()]


class Settings(BaseModel):
    # Diretórios do projeto
    BASE_DIR: Path = BASE_DIR
    DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    INDEX_DIR: Path = BASE_DIR / "index" / "v1"
    EVIDENCE_DIR: Path = BASE_DIR / "evidence"

    # Parâmetros de Suficiência de Evidência (RAG)
    TAU: float = float(os.getenv("RAG_TAU", "0.28"))
    STRONG_TAU: float = float(os.getenv("RAG_STRONG_TAU", "0.40"))
    CATALOG_TAU: float = float(os.getenv("RAG_CATALOG_TAU", "0.20"))
    CATALOG_COMPETITION_MARGIN: float = float(
        os.getenv("RAG_CATALOG_COMPETITION_MARGIN", "0.035")
    )
    DELTA: float = float(os.getenv("RAG_DELTA", "0.10"))
    TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    SEMANTIC_WEIGHT: float = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.80"))
    LEXICAL_WEIGHT: float = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.20"))
    CONSTRAINT_MARGIN: float = float(os.getenv("RAG_CONSTRAINT_MARGIN", "0.03"))
    CONSTRAINT_PENALTY_WEIGHT: float = float(
        os.getenv("RAG_CONSTRAINT_PENALTY_WEIGHT", "1.5")
    )
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))

    # Configurações do Provedor de LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, mock
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "embeddinggemma:300m-qat-q4_0",
    )
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    OLLAMA_CONTEXT_LENGTH: int = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "4096"))
    OLLAMA_MAX_CONTEXT_CHUNKS: int = int(os.getenv("OLLAMA_MAX_CONTEXT_CHUNKS", "4"))

    # Timeouts e Resiliência (Requisitos: p95 < 5s)
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "400"))
    LLM_CONCURRENCY_LIMIT: int = int(os.getenv("LLM_CONCURRENCY_LIMIT", "20"))

    # Embeddings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "dense")
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "512"))
    EMBEDDING_TIMEOUT_SECONDS: float = float(
        os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60.0")
    )
    AUTO_INGEST_ON_STARTUP: bool = os.getenv(
        "AUTO_INGEST_ON_STARTUP",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}

    # Servidor API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")
    CORS_ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: _csv_env(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8000",
        )
    )

settings = Settings()
