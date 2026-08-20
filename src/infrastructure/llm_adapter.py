"""Adaptador do cliente LLM com semáforo de concorrência, timeouts, retries e extração de citações."""
import asyncio
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote
import httpx

from src.config import QUERY_SYNONYMS, settings
from src.domain.citations import extract_context_citations
from src.domain.models import Chunk, ChunkKind
from src.infrastructure.embeddings import normalize_portuguese_text
from src.infrastructure.telemetry import logger
from src.ports.llm_client import LLMClient


PROMPT_SYSTEM_ANSWER = """Você é um assistente virtual de suporte interno a colaboradores.
Sua responsabilidade é responder dúvidas sobre políticas e procedimentos da empresa com base ESTRITAMENTE nos trechos fornecidos.

Regras inegociáveis:
1. Responda SEMPRE em língua portuguesa, de forma concisa, cordial e direta.
2. Nunca invente fatos ou adicione regras que não estejam nos trechos fornecidos.
3. Ao usar uma informação, você DEVE citar explicitamente o chunk_id entre colchetes, por exemplo: [chunk_id].
4. Se os trechos não contiverem a informação necessária para responder à pergunta, diga: "Não encontrei essa informação na base de políticas e procedimentos."
"""

PROMPT_SYSTEM_REWRITE = """Você é um otimizador de buscas para base de conhecimento interna.
Reescreva a pergunta do usuário para torná-la mais clara e assertiva para busca vetorial.
Expanda siglas comuns, remova saudações e foque nos termos-chave e políticas pertinentes.
Retorne APENAS a pergunta reescrita, sem explicações adicionais.
"""

PROMPT_SYSTEM_VERIFY_LOCAL = """Classifique se o trecho é evidência aplicável para
responder à intenção da pergunta. Responda apenas SIM ou NAO. Aceite sinônimos e
paráfrases. Uma política com opções e condições é SIM para uma pergunta de
possibilidade, mesmo sem confirmar se o usuário cumpre a condição: ela permite uma
resposta condicional. Não basta compartilhar tema ou palavra. Se houver item
específico, o trecho deve incluí-lo ou um equivalente semântico. Um procedimento
genérico não prova que um item específico seja elegível.

Exemplos:
Pergunta: Posso executar minhas atividades fora da sede?
Trecho: O regime híbrido tem dias presenciais e dias em trabalho remoto; remoto integral depende de aprovação.
Classificação: SIM

Pergunta: Qual é o modelo do automóvel?
Trecho: Há três modelos de contratação.
Classificação: NAO

Pergunta: Como reembolsar ingresso de cinema?
Trecho: Pedidos de reembolso são enviados pelo portal.
Classificação: NAO

Pergunta: Como evitar invasores digitais?
Trecho: Contas devem usar autenticação de dois fatores.
Classificação: SIM
"""

PROMPT_SYSTEM_REWRITE_LOCAL = """Reformule a pergunta como uma consulta curta para
pesquisar políticas internas. Expresse a mesma intenção com termos formais e
sinônimos usuais. Preserve entidades, objetos específicos, siglas, negações e
números. Não responda, não explique e não introduza outro assunto. Retorne somente
uma linha.
"""


class MockLLMAdapter(LLMClient):
    """
    Adaptador Mock de LLM determinístico e ultrarrápido.
    Utilizado para testes automatizados, CI e demonstrações sem necessidade de chave de API externa.
    """

    _STOP_WORDS = {
        "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do",
        "dos", "e", "em", "eu", "me", "meu", "minha", "na", "nas", "no",
        "nos", "o", "os", "ou", "para", "pode", "podem", "por", "posso",
        "qual", "quais", "que", "regras", "se", "ser", "sobre", "um", "uma",
        "fluxo", "funcionamento", "informacao", "informacoes", "politica", "processo",
        "voce", "voces",
    }

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "deterministic-mock"

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in normalize_portuguese_text(text).split()
            if len(token) > 2 and token not in cls._STOP_WORDS
        }

    @staticmethod
    def _same_term(left: str, right: str) -> bool:
        if left == right:
            return True
        return len(left) >= 5 and len(right) >= 5 and left[:4] == right[:4]

    @classmethod
    def _lexical_relevance(cls, query: str, text: str) -> float:
        query_tokens = cls._tokens(query)
        text_tokens = cls._tokens(text)
        if not query_tokens:
            return 0.0
        matches = sum(
            any(cls._same_term(query_token, text_token) for text_token in text_tokens)
            for query_token in query_tokens
        )
        return matches / len(query_tokens)

    async def generate_answer(self, query: str, chunks: List[Chunk]) -> Tuple[str, List[str]]:
        if not chunks:
            return "Não encontrei essa informação na base de políticas e procedimentos.", []

        # O ranking vetorial pode trazer chunks do mesmo documento que não respondem
        # à pergunta. No mock, mantenha somente os trechos lexicalmente mais próximos.
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: (
                self._lexical_relevance(query, f"{chunk.section}\n{chunk.content}"),
                chunk.score,
            ),
            reverse=True,
        )
        best_relevance = self._lexical_relevance(
            query,
            f"{ranked_chunks[0].section}\n{ranked_chunks[0].content}",
        )
        selected_chunks = [ranked_chunks[0]]
        normalized_query = normalize_portuguese_text(query)
        query_tokens = self._tokens(query)
        source_tokens = self._tokens(
            Path(ranked_chunks[0].source).stem.replace("_", " ")
        )
        is_document_overview = bool(query_tokens) and query_tokens.issubset(source_tokens)
        is_list_query = bool(
            set(normalized_query.split()).intersection({"quais", "listar", "liste", "tipos"})
        ) or is_document_overview or ranked_chunks[0].kind == ChunkKind.CATALOG

        # Perguntas de listagem podem atravessar chunks adjacentes da mesma seção.
        # Incluí-los evita omitir itens apenas por causa do limite de chunking.
        if is_list_query:
            first = ranked_chunks[0]
            for chunk in ranked_chunks[1:]:
                relevance = self._lexical_relevance(
                    query,
                    f"{chunk.section}\n{chunk.content}",
                )
                if (
                    chunk.source == first.source
                    and chunk.section == first.section
                    and relevance >= best_relevance * 0.8
                ):
                    selected_chunks.append(chunk)
                if len(selected_chunks) == 2:
                    break

        cited_ids: List[str] = []
        answer_parts: List[str] = []

        for chunk in selected_chunks:
            cited_ids.append(chunk.chunk_id)
            # Seleciona as linhas que de fato respondem à consulta, em vez de
            # truncar mecanicamente o começo do chunk.
            lines = [l.strip() for l in chunk.content.split("\n") if l.strip() and not l.strip().startswith("#")]
            line_scores = [self._lexical_relevance(query, line) for line in lines]
            best_line_score = max(line_scores, default=0.0)
            if is_list_query:
                relevant_lines = lines
            elif best_line_score > 0:
                candidate_indices = [
                    index
                    for index, score in enumerate(line_scores)
                    if score >= best_line_score * 0.45
                ]
                best_indices = sorted(
                    sorted(
                        candidate_indices,
                        key=lambda index: line_scores[index],
                        reverse=True,
                    )[:2]
                )
                relevant_lines = [lines[index] for index in best_indices]
            else:
                relevant_lines = lines[:1]

            # Se a linha escolhida introduz uma lista, os itens imediatamente
            # seguintes fazem parte da mesma evidência. Omiti-los produziria
            # respostas como "há três modelos:" sem dizer quais são.
            selected_line_set = set(relevant_lines)
            for index, line in enumerate(lines):
                if line in selected_line_set and line.rstrip().endswith(":"):
                    for following_line in lines[index + 1:]:
                        if not following_line.startswith("-"):
                            break
                        selected_line_set.add(following_line)
            relevant_lines = [line for line in lines if line in selected_line_set]
            separator = "\n" if chunk.kind == ChunkKind.CATALOG else " "
            summary_snippet = separator.join(relevant_lines) if relevant_lines else chunk.content[:200]
            answer_parts.append(f"{summary_snippet} [{chunk.chunk_id}]")

        answer = "\n\n".join(answer_parts)
        return answer, cited_ids

    async def rewrite_query(self, query: str) -> str:
        """Normaliza a consulta e expande apenas abreviações gerais do domínio."""
        q = normalize_portuguese_text(query)
        return " ".join(QUERY_SYNONYMS.get(token, token) for token in q.split()).strip()

    async def is_healthy(self) -> bool:
        return True


class OpenAICompatibleLLMAdapter(LLMClient):
    """Adaptador para OpenAI com semáforo de concorrência e retries."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY_LIMIT)
        self.api_key = settings.OPENAI_API_KEY if api_key is None else api_key
        self.model = settings.OPENAI_MODEL if model is None else model
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        self.client.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def _extract_citations(self, text: str, available_chunk_ids: set) -> List[str]:
        """Extrai chunk_ids citados no formato [chunk_id] que existam nos chunks disponíveis."""
        return extract_context_citations(text, available_chunk_ids)

    async def _call_api_with_retry(self, messages: List[dict]) -> str:
        async with self.semaphore:
            last_err = None
            for attempt in range(self.max_retries + 1):
                try:
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": settings.LLM_MAX_TOKENS,
                        "temperature": 0.0,
                    }
                    response = await self.client.post("chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    last_err = e
                    logger.warning(
                        f"Tentativa {attempt + 1} de chamada ao LLM falhou: {e}. "
                        f"Tentando novamente se houver retries restantes."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.3 * (2 ** attempt))
            raise last_err

    async def generate_answer(self, query: str, chunks: List[Chunk]) -> Tuple[str, List[str]]:
        context_blocks = [
            f"--- Início do Chunk: [{c.chunk_id}] (Fonte: {c.source}, Seção: {c.section}) ---\n{c.content}\n--- Fim do Chunk [{c.chunk_id}] ---"
            for c in chunks
        ]
        context_str = "\n\n".join(context_blocks)
        user_message = f"Pergunta: {query}\n\nContexto disponível:\n{context_str}\n\nResposta fundamentada com citações dos chunks no formato [chunk_id]:"

        messages = [
            {"role": "system", "content": PROMPT_SYSTEM_ANSWER},
            {"role": "user", "content": user_message},
        ]

        text = await self._call_api_with_retry(messages)
        available_ids = {c.chunk_id for c in chunks}
        cited = self._extract_citations(text, available_ids)

        return text, cited

    async def rewrite_query(self, query: str) -> str:
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM_REWRITE},
            {"role": "user", "content": f"Pergunta original: {query}"},
        ]
        return await self._call_api_with_retry(messages)

    async def is_healthy(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = await self.client.get(f"models/{quote(self.model, safe='')}")
            response.raise_for_status()
            data = response.json()
            return data.get("id") == self.model and data.get("object") == "model"
        except Exception as exc:
            logger.warning(f"Health check do provedor OpenAI falhou: {exc}")
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class OllamaLLMAdapter(LLMClient):
    """Adaptador de saída para um modelo local servido pela API nativa do Ollama."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY_LIMIT)
        self.model = settings.OLLAMA_MODEL if model is None else model
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self._owns_client = client is None
        configured_url = settings.OLLAMA_BASE_URL if base_url is None else base_url
        self.client = client or httpx.AsyncClient(
            base_url=f"{configured_url.rstrip('/')}/",
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    @staticmethod
    def _extract_citations(text: str, available_chunk_ids: set[str]) -> List[str]:
        return extract_context_citations(text, available_chunk_ids)

    async def _call_api_with_retry(
        self,
        messages: List[dict],
        *,
        think: Optional[bool] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        async with self.semaphore:
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                try:
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                        "options": {
                            "temperature": 0.0,
                            "seed": 0,
                            "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
                            "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
                        },
                    }
                    if think is not None:
                        payload["think"] = think
                    response = await self.client.post("chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if not data.get("done") or not content:
                        raise RuntimeError("Ollama retornou uma geração incompleta ou vazia.")
                    return content
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        f"Tentativa {attempt + 1} de chamada ao Ollama falhou: {exc}. "
                        "Tentando novamente se houver retries restantes."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.3 * (2 ** attempt))
            if last_error is None:
                raise RuntimeError("Falha desconhecida ao chamar o Ollama.")
            raise last_error

    async def generate_answer(self, query: str, chunks: List[Chunk]) -> Tuple[str, List[str]]:
        selected_chunks = chunks[: settings.OLLAMA_MAX_CONTEXT_CHUNKS]
        if not selected_chunks:
            return "Não encontrei essa informação na base de políticas e procedimentos.", []

        matched_catalogs = [chunk for chunk in selected_chunks if chunk.catalog_match]
        if matched_catalogs:
            # Uma consulta sobre o escopo já foi resolvida pelo catálogo. Misturar
            # chunks temáticos pode introduzir homônimos, como "ajudar" versus
            # "ajuda de custo", e degradar uma resposta estrutural completa.
            selected_chunks = matched_catalogs[:1]

        async def verify_chunk(chunk: Chunk) -> bool:
            # O catálogo é gerado pela própria ingestão e só recebe esta marca
            # depois da classificação semântica contrastiva com os demais
            # documentos. Não dependa da oscilação do modelo pequeno para validar
            # novamente essa evidência estrutural.
            if chunk.catalog_match:
                return True
            decision = await self._call_api_with_retry(
                [
                    {"role": "system", "content": PROMPT_SYSTEM_VERIFY_LOCAL},
                    {
                        "role": "user",
                        "content": (
                            f"PERGUNTA: {query}\n\n"
                            f"TRECHO:\n{chunk.content}\n\nDECISAO:"
                        ),
                    },
                ],
                think=False,
                max_tokens=20,
            )
            return normalize_portuguese_text(decision).strip(" .!") == "sim"

        decisions = await asyncio.gather(
            *(verify_chunk(chunk) for chunk in selected_chunks)
        )
        best_score = max(chunk.score for chunk in selected_chunks)
        required_acronyms = {
            match.casefold()
            for match in re.findall(r"(?<!\w)[A-ZÀ-Ý0-9]{2,5}(?!\w)", query)
        }
        approved_chunks = [
            chunk
            for chunk, approved in zip(selected_chunks, decisions)
            if approved
            and chunk.score >= best_score * 0.65
            and required_acronyms.issubset(
                {
                    match.casefold()
                    for match in re.findall(
                        r"(?<!\w)[A-Za-zÀ-ÿ0-9]{2,5}(?!\w)",
                        chunk.content,
                    )
                }
            )
        ]
        # Similaridades baixas podem aproximar conceitos vagamente relacionados.
        # Nessa faixa, combine uma aprovação semântica do verificador com a
        # convergência de ao menos dois resultados fortes no mesmo documento.
        # Isso preserva paráfrases sem depender de dicionários específicos, mas
        # rejeita uma coincidência isolada como "modelo do carro" versus
        # "modelos de trabalho".
        if best_score < 0.35:
            candidates_by_source: dict[str, int] = {}
            for chunk in selected_chunks:
                if chunk.score < best_score * 0.65:
                    continue
                candidates_by_source[chunk.source] = (
                    candidates_by_source.get(chunk.source, 0) + 1
                )
            approved_chunks = [
                chunk
                for chunk in approved_chunks
                if candidates_by_source.get(chunk.source, 0) >= 2
            ]
        model_selected_ids = [chunk.chunk_id for chunk in approved_chunks]
        if not model_selected_ids:
            # Modelos muito pequenos podem ocasionalmente recusar até uma
            # correspondência literal inequívoca. Nesse caso, recupere apenas
            # trechos com cobertura lexical muito alta. A exigência de 75%
            # evita transformar coincidências genéricas de tema em evidência.
            lexical_matches = [
                chunk
                for chunk in selected_chunks
                if (
                    MockLLMAdapter._lexical_relevance(
                        query,
                        f"{chunk.section}\n{chunk.content}",
                    )
                    >= 0.75
                    or (
                        best_score >= 0.40
                        and MockLLMAdapter._lexical_relevance(
                            query,
                            f"{chunk.section}\n{chunk.content}",
                        )
                        >= 0.50
                    )
                )
            ]
            model_selected_ids = [chunk.chunk_id for chunk in lexical_matches[:2]]
            if not model_selected_ids:
                logger.info("O verificador local não aprovou nenhum trecho recuperado.")
                return "Não encontrei essa informação na base de políticas e procedimentos.", []
            logger.info(
                "Aplicando fallback lexical de alta confiança após recusa do "
                "verificador local."
            )

        # Preserve a ordem do ranking de recuperação entre os trechos aprovados.
        # A redação continua extrativa: nenhum fato pode sair do modelo pequeno.
        chunk_by_id = {chunk.chunk_id: chunk for chunk in selected_chunks}
        verified_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id in model_selected_ids
            if chunk_id in chunk_by_id
        ][:2]
        answer_parts: List[str] = []
        cited_ids: List[str] = []
        extractor = MockLLMAdapter()
        for chunk in verified_chunks:
            partial_answer, partial_citations = await extractor.generate_answer(
                query,
                [chunk],
            )
            answer_parts.append(partial_answer)
            cited_ids.extend(partial_citations)
        return " ".join(answer_parts), cited_ids

    async def rewrite_query(self, query: str) -> str:
        rewritten = await self._call_api_with_retry(
            [
                {"role": "system", "content": PROMPT_SYSTEM_REWRITE_LOCAL},
                {"role": "user", "content": f"Pergunta original: {query}"},
            ],
            think=False,
        )
        deterministic = await MockLLMAdapter().rewrite_query(query)
        normalized_rewrite = normalize_portuguese_text(rewritten)
        original_token_count = len(normalize_portuguese_text(query).split())
        rewrite_token_count = len(normalized_rewrite.split())
        looks_like_answer = (
            "\n" in rewritten
            or ":" in rewritten
            or any(
                marker in normalized_rewrite
                for marker in (
                    "retorno da pergunta",
                    "a resposta e",
                    "incluem",
                    "nao posso ajudar",
                )
            )
            or rewrite_token_count > max(12, original_token_count * 2)
        )
        if (
            looks_like_answer
            or MockLLMAdapter._lexical_relevance(query, rewritten) < 0.25
        ):
            logger.warning(
                "Reescrita local descartada por formato inválido ou divergência; "
                "aplicando normalização determinística."
            )
            return deterministic

        # Preserve sempre os termos críticos da pergunta. O LLM pode enriquecer a
        # busca, mas não pode remover palavras como "furto", "prazo" ou "reembolso".
        return f"{deterministic} {rewritten}".strip()

    async def is_healthy(self) -> bool:
        try:
            response = await self.client.get("tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(
                self.model in {item.get("name"), item.get("model")}
                for item in models
            )
        except Exception as exc:
            logger.warning(f"Health check do provedor Ollama falhou: {exc}")
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def get_llm_client() -> LLMClient:
    """Factory para instanciar o cliente LLM conforme as configurações."""
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "mock":
        logger.info("Instanciando Mock LLM Client.")
        return MockLLMAdapter()
    if provider == "ollama":
        logger.info(
            f"Instanciando LLM local Ollama com o modelo {settings.OLLAMA_MODEL}."
        )
        return OllamaLLMAdapter()
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY é obrigatória quando LLM_PROVIDER=openai.")
        logger.info("Instanciando LLM Client OpenAI.")
        return OpenAICompatibleLLMAdapter()
    raise RuntimeError(
        f"LLM_PROVIDER não suportado: {settings.LLM_PROVIDER!r}. "
        "Use 'ollama', 'openai' ou 'mock'."
    )
