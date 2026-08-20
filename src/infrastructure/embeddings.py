"""Adaptadores de embeddings locais/cloud e serviço selecionado por configuração."""
import functools
import hashlib
import re
import unicodedata
from typing import List, Optional

import httpx
import numpy as np

from src.config import settings
from src.infrastructure.telemetry import logger
from src.ports.embedding_client import EmbeddingClient


def normalize_portuguese_text(text: str) -> str:
    """Normaliza texto em português removendo acentos e convertendo para minúsculas."""
    text = text.lower().strip()
    text = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_matrix(raw_vectors: List[List[float]]) -> np.ndarray:
    vectors = np.asarray(raw_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise RuntimeError("O provedor retornou uma matriz de embeddings inválida.")
    if not np.isfinite(vectors).all():
        raise RuntimeError("O provedor retornou embeddings com valores não finitos.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class LocalDenseEmbedder(EmbeddingClient):
    """Hashing determinístico usado como fallback offline e nos testes."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    @property
    def provider_name(self) -> str:
        return "dense"

    @property
    def model_name(self) -> str:
        return f"hashing-ngram-{self.dim}"

    def encode(self, text: str) -> np.ndarray:
        normalized = normalize_portuguese_text(text)
        words = normalized.split()
        vector = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vector

        tokens = list(words)
        tokens.extend(
            f"{words[index]}_{words[index + 1]}"
            for index in range(len(words) - 1)
        )
        for word in words:
            if len(word) >= 4:
                tokens.extend(word[index:index + 3] for index in range(len(word) - 2))

        for token in tokens:
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = digest % self.dim
            sign = 1.0 if ((digest >> 8) % 2 == 0) else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode(text)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self.encode_batch(texts)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Alias mantido para consumidores existentes."""
        return np.asarray([self.encode(text) for text in texts], dtype=np.float32)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.encode_batch(texts)


class OllamaEmbeddingAdapter(EmbeddingClient):
    """Embeddings semânticos multilíngues servidos pela API nativa do Ollama."""

    QUERY_PREFIX = "task: search result | query: "
    DOCUMENT_PREFIX = "title: Política interna | text: "

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        configured_url = settings.OLLAMA_BASE_URL if base_url is None else base_url
        self.model = settings.OLLAMA_EMBEDDING_MODEL if model is None else model
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=f"{configured_url.rstrip('/')}/",
            timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def _encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.post(
            "embed",
            json={
                "model": self.model,
                "input": texts,
                "truncate": False,
                "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            },
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama retornou quantidade inesperada de embeddings.")
        return _normalized_matrix(embeddings)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([f"{self.QUERY_PREFIX}{text}"])[0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode([f"{self.QUERY_PREFIX}{text}" for text in texts])

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode([f"{self.DOCUMENT_PREFIX}{text}" for text in texts])

    def is_healthy(self) -> bool:
        try:
            response = self.client.get("tags")
            response.raise_for_status()
            return any(
                self.model in {item.get("name"), item.get("model")}
                for item in response.json().get("models", [])
            )
        except Exception as exc:
            logger.warning(f"Health check do embedding Ollama falhou: {exc}")
            return False

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class OpenAIEmbeddingAdapter(EmbeddingClient):
    """Adaptador de embeddings da OpenAI via API HTTP."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url="https://api.openai.com/v1/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def _encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.post(
            "embeddings",
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return _normalized_matrix([item["embedding"] for item in data])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def get_embedding_client() -> EmbeddingClient:
    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    if provider == "dense":
        logger.info("Embeddings configurados com hashing local determinístico.")
        return LocalDenseEmbedder(dim=384)
    if provider == "ollama":
        logger.info(
            "Embeddings semânticos configurados via Ollama com o modelo "
            f"{settings.OLLAMA_EMBEDDING_MODEL}."
        )
        return OllamaEmbeddingAdapter()
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY é obrigatória quando EMBEDDING_PROVIDER=openai."
            )
        logger.info(
            "Embeddings semânticos configurados via OpenAI com o modelo "
            f"{settings.OPENAI_EMBEDDING_MODEL}."
        )
        return OpenAIEmbeddingAdapter(settings.OPENAI_API_KEY)
    raise RuntimeError(
        f"EMBEDDING_PROVIDER não suportado: {settings.EMBEDDING_PROVIDER!r}. "
        "Use 'ollama', 'openai' ou 'dense'."
    )


class EmbeddingService:
    """Fachada com cache de consultas e adaptador intercambiável."""

    def __init__(self, client: Optional[EmbeddingClient] = None):
        self.client = client or get_embedding_client()

    @property
    def provider_name(self) -> str:
        return self.client.provider_name

    @property
    def model_name(self) -> str:
        return self.client.model_name

    @property
    def fingerprint(self) -> str:
        return self.client.fingerprint

    @property
    def requires_io(self) -> bool:
        return self.provider_name != "dense"

    @functools.lru_cache(maxsize=settings.EMBEDDING_CACHE_SIZE)
    def encode_query(self, query: str) -> tuple:
        return tuple(self.client.encode_query(query).tolist())

    def get_query_vector(self, query: str) -> np.ndarray:
        return np.asarray(self.encode_query(query), dtype=np.float32)

    def get_query_vectors(self, queries: List[str]) -> np.ndarray:
        return self.client.encode_queries(queries)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.client.encode_documents(texts)

    def is_healthy(self) -> bool:
        return self.client.is_healthy()

    def close(self) -> None:
        self.client.close()


embedding_service = EmbeddingService()
