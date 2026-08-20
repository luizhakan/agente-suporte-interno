"""Porta para interação com o modelo de linguagem (LLM)."""
from abc import ABC, abstractmethod
from typing import List, Tuple
from src.domain.models import Chunk


class LLMClient(ABC):
    """Interface abstrata para o cliente de LLM."""

    @property
    def provider_name(self) -> str:
        """Nome estável do provedor para observabilidade, sem expor credenciais."""
        return self.__class__.__name__

    @property
    def model_name(self) -> str:
        """Nome do modelo ativo para observabilidade."""
        return "unknown"

    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        chunks: List[Chunk],
    ) -> Tuple[str, List[str]]:
        """
        Gera uma resposta fundamentada estritamente nos chunks fornecidos.
        
        Retorna:
            Tuple[str, List[str]]: (texto_da_resposta, lista_de_chunk_ids_citados)
        """
        pass

    @abstractmethod
    async def rewrite_query(self, query: str) -> str:
        """
        Reescreve e expande a pergunta do usuário para melhorar a recuperação.
        """
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Verifica a disponibilidade do provedor de LLM."""
        pass

    async def aclose(self) -> None:
        """Libera recursos mantidos pelo adaptador, quando houver."""
        return None
