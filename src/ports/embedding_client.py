"""Porta para geração intercambiável de embeddings de busca."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingClient(ABC):
    """Contrato independente de provedor para indexação e consultas semânticas."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    def fingerprint(self) -> str:
        return f"{self.provider_name}:{self.model_name}:retrieval-v1"

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        pass

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Codifica consultas em lote; adaptadores podem otimizar a chamada."""
        return np.asarray([self.encode_query(text) for text in texts], dtype=np.float32)

    @abstractmethod
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        pass

    def is_healthy(self) -> bool:
        return True

    def close(self) -> None:
        return None
