"""Adaptador do repositório vetorial de recuperação (RetrievalRepository)."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

from src.config import settings
from src.domain.models import Chunk, ChunkKind
from src.infrastructure.embeddings import embedding_service, normalize_portuguese_text
from src.infrastructure.telemetry import logger
from src.ports.retrieval_repository import RetrievalRepository


class InMemoryVectorStore(RetrievalRepository):
    """
    Implementação do RetrievalRepository com armazenamento vetorial em memória.
    Atende aos requisitos de latência (< 5ms de busca) e serialização em artefatos versionados.
    """

    _STOP_WORDS = {
        "a", "ao", "aos", "as", "ate", "antes", "com", "como", "consigo", "da",
        "das", "de", "deve", "do", "dos", "e", "em", "eu", "funciona", "me",
        "meu", "minha", "na", "nas", "no", "nos", "o", "os", "ou", "para",
        "pedir", "pode", "podem", "por", "posso", "qual", "quais", "quando",
        "quantos", "que", "regras", "se", "ser", "sobre", "tem", "ter", "todos",
        "um", "uma", "fluxo", "funcionamento", "informacao", "informacoes",
        "politica", "processo", "voce", "voces",
    }

    # Termos que tornam explícito que a pergunta é sobre a organização, e não
    # sobre a capacidade deste assistente. O catálogo nunca pode virar uma
    # descrição inventada do negócio.
    _ORGANIZATION_SUBJECT_TERMS = {
        "companhia", "empresa", "fabrica", "fabricam", "fabricar",
        "negocio", "organizacao", "produto", "produtos", "ramo",
    }

    _STEM_SUFFIXES = (
        "amentos", "imentos", "adoras", "adores", "acoes", "aveis", "iveis",
        "amento", "imento", "mente", "adas", "ados", "idas", "idos", "es", "s",
    )

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        docs_dir: Optional[Path] = None,
    ):
        self.index_dir = index_dir or settings.INDEX_DIR
        self.docs_dir = docs_dir or settings.DOCS_DIR
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._is_loaded = False
        self._lexical_documents: List[Set[str]] = []
        self._source_documents: List[Set[str]] = []
        self._document_frequency: Dict[str, int] = {}

    def _corpus_fingerprint(self) -> str:
        """Identifica deterministicamente nomes e conteúdos Markdown indexados."""
        digest = hashlib.sha256()
        for document_path in sorted(self.docs_dir.glob("*.md")):
            digest.update(document_path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document_path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def load(self) -> bool:
        """Carrega os artefatos de índice versionados da pasta index/v{N}/."""
        chunks_path = self.index_dir / "chunks.json"
        vectors_path = self.index_dir / "vectors.npy"
        metadata_path = self.index_dir / "metadata.json"

        if not chunks_path.exists() or not vectors_path.exists() or not metadata_path.exists():
            logger.warning(
                f"Índice não encontrado em {self.index_dir}. "
                "Execute 'make ingest' para gerar os artefatos."
            )
            self._is_loaded = False
            return False

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            chunks = [Chunk(**c) for c in raw_chunks]
            vectors = np.load(vectors_path, allow_pickle=False)
            self._validate_index(chunks, vectors)
            if metadata.get("embedding_fingerprint") != embedding_service.fingerprint:
                raise ValueError(
                    "Índice gerado por outro provedor/modelo de embeddings: "
                    f"índice={metadata.get('embedding_fingerprint')!r}, "
                    f"ativo={embedding_service.fingerprint!r}."
                )
            if metadata.get("corpus_fingerprint") != self._corpus_fingerprint():
                raise ValueError(
                    "A base Markdown mudou desde a geração do índice."
                )
            if metadata.get("vector_dimension") != vectors.shape[1]:
                raise ValueError("Metadados do índice divergem da dimensão dos vetores.")
            self.chunks = chunks
            self.vectors = vectors
            self._build_lexical_index()
            self._is_loaded = True
            logger.info(
                f"Índice carregado com sucesso de {self.index_dir}: "
                f"{len(self.chunks)} chunks, matriz {self.vectors.shape}."
            )
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar índice vetorial: {e}", exc_info=True)
            self.chunks = []
            self.vectors = None
            self._lexical_documents = []
            self._source_documents = []
            self._document_frequency = {}
            self._is_loaded = False
            return False

    def save(self, chunks: List[Chunk], vectors: np.ndarray):
        """Salva os chunks e a matriz de vetores no diretório do índice."""
        self._validate_index(chunks, vectors)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.index_dir / "chunks.json"
        vectors_path = self.index_dir / "vectors.npy"
        metadata_path = self.index_dir / "metadata.json"

        raw_chunks = [c.model_dump() for c in chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(raw_chunks, f, ensure_ascii=False, indent=2)

        np.save(vectors_path, vectors)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_fingerprint": embedding_service.fingerprint,
                    "corpus_fingerprint": self._corpus_fingerprint(),
                    "vector_dimension": int(vectors.shape[1]),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.chunks = chunks
        self.vectors = vectors
        self._build_lexical_index()
        self._is_loaded = True
        logger.info(f"Índice salvo com sucesso em {self.index_dir} ({len(chunks)} chunks).")

    @staticmethod
    def _validate_index(chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Impede que um índice corrompido ou incompatível seja marcado como saudável."""
        if vectors.ndim != 2:
            raise ValueError("A matriz de vetores do índice deve ter duas dimensões.")
        if len(chunks) == 0 or vectors.shape[0] != len(chunks):
            raise ValueError(
                "O número de vetores do índice deve ser igual ao número de chunks e maior que zero."
            )
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("A matriz de vetores contém dimensão vazia ou valores inválidos.")

    @classmethod
    def _stem(cls, token: str) -> str:
        """Reduz flexões comuns sem depender de bibliotecas linguísticas pesadas."""
        for suffix in cls._STEM_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]
        return token

    @classmethod
    def _tokens(cls, text: str) -> Set[str]:
        return {
            cls._stem(token)
            for token in normalize_portuguese_text(text).split()
            if len(token) >= 2 and token not in cls._STOP_WORDS
        }

    def _build_lexical_index(self) -> None:
        self._source_documents = [
            self._tokens(Path(chunk.source).stem.replace("_", " "))
            for chunk in self.chunks
        ]
        self._lexical_documents = [
            self._tokens(
                f"{chunk.section}\n{chunk.retrieval_content or chunk.content}"
            ) | source_tokens
            for chunk, source_tokens in zip(self.chunks, self._source_documents)
        ]
        frequencies: Dict[str, int] = {}
        for document_tokens in self._lexical_documents:
            for token in document_tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
        self._document_frequency = frequencies

    def _lexical_scores(self, query: str) -> np.ndarray:
        """Calcula a cobertura dos termos da consulta ponderada por raridade no corpus."""
        if len(self._lexical_documents) != len(self.chunks):
            self._build_lexical_index()
        query_tokens = self._tokens(query)
        if not query_tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)

        document_count = len(self._lexical_documents)
        weights = {
            token: math.log((document_count + 1) / (self._document_frequency.get(token, 0) + 1)) + 1
            for token in query_tokens
        }
        total_weight = sum(weights.values())
        return np.array(
            [
                (
                    sum(weight for token, weight in weights.items() if token in document_tokens)
                    / total_weight
                ) ** 2
                for document_tokens in self._lexical_documents
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _unsupported_constraint_improvement(
        tokens: List[str],
        improvements: Dict[str, float],
        margin: float,
    ) -> float:
        """Retorna o maior ganho causado por ignorar uma restrição essencial.

        Um termo adjacente a um núcleo fortemente suportado é tratado como
        modificador. Assim, "humilde residência" preserva "residência", enquanto
        "cachorro para trabalhar" não consegue apagar "cachorro" da intenção.
        """
        violating_improvements: List[float] = []
        for index, token in enumerate(tokens):
            improvement = improvements.get(token, 0.0)
            if improvement < margin:
                continue
            adjacent_tokens = [
                tokens[adjacent_index]
                for adjacent_index in (index - 1, index + 1)
                if 0 <= adjacent_index < len(tokens)
            ]
            is_modifier = any(
                improvements.get(adjacent_token, 0.0) <= -margin
                for adjacent_token in adjacent_tokens
            )
            if not is_modifier:
                violating_improvements.append(improvement)
        return max(violating_improvements, default=0.0)

    async def search(self, query: str, top_k: int = 4) -> List[Chunk]:
        """Realiza busca vetorial por similaridade de cosseno."""
        if not self._is_loaded or self.vectors is None or len(self.chunks) == 0:
            raise RuntimeError("Repositório vetorial não está carregado ou está vazio.")

        if embedding_service.requires_io:
            query_vec = await asyncio.to_thread(
                embedding_service.get_query_vector,
                query,
            )
        else:
            query_vec = embedding_service.get_query_vector(query)
        if query_vec.ndim != 1 or query_vec.shape[0] != self.vectors.shape[1]:
            raise RuntimeError(
                "Dimensão do embedding da consulta incompatível com o índice: "
                f"consulta={query_vec.shape}, índice={self.vectors.shape}. "
                "Reexecute a ingestão com o mesmo provedor de embeddings da API."
            )
        # O provedor semântico entende paráfrases; a cobertura lexical preserva
        # nomes, siglas, números e termos raros. O fallback hashing mantém os pesos
        # históricos usados pelos testes determinísticos.
        dense_scores = np.clip(np.dot(self.vectors, query_vec), 0.0, 1.0)
        lexical_scores = self._lexical_scores(query)
        if embedding_service.provider_name == "dense":
            semantic_weight, lexical_weight = 0.35, 0.65
        else:
            semantic_weight = settings.SEMANTIC_WEIGHT
            lexical_weight = settings.LEXICAL_WEIGHT
        weight_total = semantic_weight + lexical_weight
        if weight_total <= 0:
            raise RuntimeError("Os pesos de recuperação devem somar um valor positivo.")
        scores = (
            (semantic_weight * dense_scores) + (lexical_weight * lexical_scores)
        ) / weight_total
        # Ausência lexical na consulta inteira não é evidência negativa. Normalize
        # nesse caso para o score semântico puro. Quando existe qualquer sinal
        # lexical, mantenha todos os chunks na mesma escala híbrida para um trecho
        # semanticamente genérico não ultrapassar uma correspondência literal.
        if float(np.max(lexical_scores, initial=0.0)) == 0.0:
            scores = dense_scores.copy()

        # Detecta restrições semânticas ausentes por ablação de termos. Se remover
        # uma palavra relevante aumenta materialmente a similaridade com o melhor
        # trecho, a busca estava ignorando essa restrição (por exemplo, encontra
        # política de escritório somente depois de desconsiderar "cachorro").
        # O cálculo ocorre em uma única chamada em lote. Ele também se aplica a
        # consultas híbridas: uma palavra literal ampla como "reembolso" não pode
        # mascarar um objeto específico ausente da base.
        semantic_constraint_penalty = 0.0
        best_dense_index = int(np.argmax(dense_scores))
        if embedding_service.provider_name != "dense":
            normalized_tokens = normalize_portuguese_text(query).split()
            removable_tokens = list(dict.fromkeys(
                token
                for token in normalized_tokens
                if len(token) >= 3 and token not in self._STOP_WORDS
            ))
            if 1 < len(removable_tokens) <= 12:
                ablated_queries = [
                    " ".join(
                        token
                        for token in normalized_tokens
                        if token != removed_token
                    )
                    for removed_token in removable_tokens
                ]
                if embedding_service.requires_io:
                    ablated_vectors = await asyncio.to_thread(
                        embedding_service.get_query_vectors,
                        ablated_queries,
                    )
                else:
                    ablated_vectors = embedding_service.get_query_vectors(ablated_queries)
                if (
                    ablated_vectors.ndim != 2
                    or ablated_vectors.shape[1] != self.vectors.shape[1]
                ):
                    raise RuntimeError(
                        "Dimensão dos embeddings de ablação incompatível com o índice."
                    )
                ablated_scores = np.clip(
                    np.dot(ablated_vectors, self.vectors[best_dense_index]),
                    0.0,
                    1.0,
                )
                improvements = {
                    removed_token: (
                        float(ablated_score)
                        - float(dense_scores[best_dense_index])
                    )
                    for removed_token, ablated_score in zip(
                        removable_tokens,
                        ablated_scores,
                    )
                }
                max_improvement = self._unsupported_constraint_improvement(
                    normalized_tokens,
                    improvements,
                    settings.CONSTRAINT_MARGIN,
                )
                if max_improvement >= settings.CONSTRAINT_MARGIN:
                    semantic_constraint_penalty = (
                        max_improvement * settings.CONSTRAINT_PENALTY_WEIGHT
                    )
                    logger.info(
                        "Aplicando penalidade de restrição semântica: %.3f.",
                        semantic_constraint_penalty,
                    )
        if semantic_constraint_penalty:
            scores = np.clip(scores - semantic_constraint_penalty, 0.0, 1.0)

        # O catálogo é uma evidência singular e completa sobre o escopo da base.
        # Promova-o somente quando ele próprio for semanticamente o melhor destino
        # da consulta e o sujeito não for explicitamente a organização. Isso
        # atende perguntas abertas sobre a capacidade do assistente sem confundir
        # a base de políticas com uma descrição institucional da empresa.
        normalized_query_tokens = set(normalize_portuguese_text(query).split())
        catalog_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.kind == ChunkKind.CATALOG
        ]
        if catalog_indices and not semantic_constraint_penalty:
            catalog_index = max(catalog_indices, key=lambda index: dense_scores[index])
            catalog_dense_score = float(dense_scores[catalog_index])
            catalog_lexical_score = float(lexical_scores[catalog_index])
            max_lexical_score = float(np.max(lexical_scores, initial=0.0))
            organization_subject = bool(
                normalized_query_tokens.intersection(self._ORGANIZATION_SUBJECT_TERMS)
            )
            if (
                not organization_subject
                and (catalog_lexical_score > 0.0 or max_lexical_score == 0.0)
                and catalog_dense_score >= settings.CATALOG_TAU
                and catalog_dense_score
                >= float(np.max(dense_scores)) - settings.CATALOG_COMPETITION_MARGIN
            ):
                scores[catalog_index] = max(
                    float(scores[catalog_index]),
                    settings.STRONG_TAU,
                )
                logger.info(
                    "Catálogo da base selecionado para consulta sobre o escopo "
                    "do assistente (similaridade %.3f).",
                    catalog_dense_score,
                )

        # Obter índices dos top_k maiores scores. Quando a seção vencedora foi
        # dividida em mais de um chunk, preserve a continuidade antes de misturar
        # seções diferentes; isso é essencial para perguntas de listagem.
        top_k = min(top_k, len(self.chunks))
        ranked_indices = list(np.argsort(scores)[::-1])

        # Se a consulta contém apenas o nome do documento (por exemplo,
        # "processo de onboarding"), ela pede uma visão geral. Nesse caso,
        # comece pelo primeiro chunk do documento em vez de deixar pequenas
        # diferenças do hashing escolherem uma seção arbitrária.
        query_tokens = self._tokens(query)
        broad_source_names = {
            chunk.source
            for chunk, source_tokens in zip(self.chunks, self._source_documents)
            if query_tokens and query_tokens.issubset(source_tokens)
        }
        if len(broad_source_names) == 1:
            broad_source = next(iter(broad_source_names))
            overview_index = next(
                idx for idx, chunk in enumerate(self.chunks) if chunk.source == broad_source
            )
            source_indices = [
                idx for idx, chunk in enumerate(self.chunks) if chunk.source == broad_source
            ]
            scores[overview_index] = min(
                1.0,
                max(float(scores[idx]) for idx in source_indices) + 0.01,
            )
            ranked_indices.remove(overview_index)
            ranked_indices.insert(0, overview_index)

        best_index = ranked_indices[0]
        best_chunk = self.chunks[best_index]
        sibling_indices = [
            idx
            for idx in ranked_indices[1:]
            if self.chunks[idx].source == best_chunk.source
            and self.chunks[idx].section == best_chunk.section
        ]
        other_indices = [
            idx for idx in ranked_indices[1:] if idx not in sibling_indices
        ]
        top_indices = [best_index, *sibling_indices, *other_indices][:top_k]

        results: List[Chunk] = []
        for idx in top_indices:
            chunk = self.chunks[idx].model_copy()
            chunk.score = float(scores[idx])
            chunk.constraint_violation = semantic_constraint_penalty > 0.0
            chunk.catalog_match = (
                chunk.kind == ChunkKind.CATALOG
                and float(scores[idx]) >= settings.STRONG_TAU
                and not normalized_query_tokens.intersection(
                    self._ORGANIZATION_SUBJECT_TERMS
                )
            )
            results.append(chunk)

        return results

    async def is_healthy(self) -> bool:
        """Verifica se o repositório está carregado e pronto para consultas."""
        return self._is_loaded and self.vectors is not None and len(self.chunks) > 0


# Instância global singleton do repositório
vector_repository = InMemoryVectorStore()
