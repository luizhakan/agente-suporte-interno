"""Porta para recuperação de contexto vetorial / documental."""
from abc import ABC, abstractmethod
from typing import List
from src.domain.models import Chunk


class RetrievalRepository(ABC):
    """Interface abstrata para recuperação de documentos."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 4) -> List[Chunk]:
        """Recupera os chunks mais relevantes para a query dada com seus scores."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Verifica a integridade e disponibilidade do repositório/índice."""
        pass
