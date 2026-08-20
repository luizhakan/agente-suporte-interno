"""Serviço de ingestão e indexação offline de documentos Markdown."""
import re
from pathlib import Path
from typing import List, Sequence

from src.config import settings
from src.domain.models import Chunk, ChunkKind
from src.infrastructure.embeddings import embedding_service
from src.infrastructure.telemetry import logger
from src.infrastructure.vector_store import InMemoryVectorStore


def split_markdown_into_chunks(
    file_path: Path,
    target_chunk_size: int = 700,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Divide um documento Markdown em chunks respeitando seções e cabeçalhos.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    source_name = file_path.name
    doc_stem = file_path.stem

    # Divide por seções marcadas por cabeçalhos (## ou #)
    lines = content.split("\n")
    sections: List[tuple[str, str]] = []
    current_section_title = "Introdução"
    current_section_lines: List[str] = []

    for line in lines:
        if line.startswith("#"):
            if current_section_lines:
                sec_text = "\n".join(current_section_lines).strip()
                if sec_text:
                    sections.append((current_section_title, sec_text))
                current_section_lines = []
            current_section_title = line.lstrip("#").strip()
        else:
            current_section_lines.append(line)

    if current_section_lines:
        sec_text = "\n".join(current_section_lines).strip()
        if sec_text:
            sections.append((current_section_title, sec_text))

    chunks: List[Chunk] = []
    chunk_counter = 1

    for sec_title, sec_text in sections:
        # Se o texto da seção couber no tamanho limite, cria um chunk único
        if len(sec_text) <= target_chunk_size:
            chunk_id = f"{doc_stem}_c{chunk_counter}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=source_name,
                    section=sec_title,
                    content=f"## {sec_title}\n{sec_text}",
                )
            )
            chunk_counter += 1
        else:
            # Divide com overlap preservando palavras
            start = 0
            while start < len(sec_text):
                end = min(start + target_chunk_size, len(sec_text))
                chunk_slice = sec_text[start:end]

                # Tenta quebrar em quebra de linha ou espaço
                if end < len(sec_text):
                    last_space = chunk_slice.rfind("\n")
                    if last_space == -1:
                        last_space = chunk_slice.rfind(" ")
                    if last_space > target_chunk_size // 2:
                        end = start + last_space
                        chunk_slice = sec_text[start:end]

                chunk_id = f"{doc_stem}_c{chunk_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=source_name,
                        section=sec_title,
                        content=f"## {sec_title}\n{chunk_slice.strip()}",
                    )
                )
                chunk_counter += 1
                start = end - overlap if end < len(sec_text) else end

    return chunks


def _markdown_title(file_path: Path) -> str:
    """Obtém o título principal do documento, com fallback legível pelo nome."""
    for line in file_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return file_path.stem.replace("_", " ").strip().title()


def build_knowledge_catalog(
    documents: Sequence[tuple[Path, Sequence[Chunk]]],
) -> Chunk:
    """Cria evidência navegacional a partir do conteúdo realmente indexado.

    O catálogo torna consultável o escopo da base sem manter uma lista manual de
    assuntos. Novos Markdown e novas seções passam a aparecer automaticamente na
    próxima ingestão.
    """
    lines = [
        "## Escopo da base interna de conhecimento",
        (
            "Se você se refere a este assistente, ele consulta os documentos da "
            "base interna e responde dúvidas com evidências sobre estes assuntos:"
        ),
    ]
    for file_path, chunks in documents:
        sections = list(dict.fromkeys(chunk.section for chunk in chunks))
        section_summary = "; ".join(sections)
        lines.append(f"- {_markdown_title(file_path)} — {section_summary}")

    return Chunk(
        chunk_id="knowledge_base_catalog_c1",
        source="base_interna",
        section="Escopo da base interna de conhecimento",
        content="\n".join(lines),
        kind=ChunkKind.CATALOG,
        retrieval_content=(
            "O que este assistente faz? Como este assistente pode ajudar? "
            "Sobre quais assuntos e temas posso perguntar? Qual é a finalidade "
            "e o escopo desta base interna de conhecimento? O assistente consulta "
            "documentos e responde dúvidas com evidências. Assuntos disponíveis: "
            + "; ".join(_markdown_title(path) for path, _ in documents)
        ),
    )


def run_ingestion(
    docs_dir: Path = settings.DOCS_DIR,
    index_dir: Path = settings.INDEX_DIR,
) -> int:
    """Executa o pipeline de ingestão e gera os artefatos versionados em index/v{N}/."""
    logger.info(f"Iniciando pipeline de ingestão a partir de: {docs_dir}")

    if not docs_dir.exists():
        raise FileNotFoundError(f"Diretório de documentos não encontrado: {docs_dir}")

    md_files = sorted(list(docs_dir.glob("*.md")))
    if not md_files:
        logger.warning(f"Nenhum arquivo .md encontrado em {docs_dir}")
        return 0

    all_chunks: List[Chunk] = []
    indexed_documents: List[tuple[Path, Sequence[Chunk]]] = []
    for doc_path in md_files:
        doc_chunks = split_markdown_into_chunks(doc_path)
        all_chunks.extend(doc_chunks)
        indexed_documents.append((doc_path, doc_chunks))
        logger.info(f"Documento '{doc_path.name}' processado em {len(doc_chunks)} chunks.")

    all_chunks.append(build_knowledge_catalog(indexed_documents))

    logger.info(f"Total de chunks extraídos: {len(all_chunks)}. Calculando embeddings...")
    texts_to_embed = [
        f"{chunk.source} | {chunk.section}\n{chunk.retrieval_content or chunk.content}"
        for chunk in all_chunks
    ]
    vectors = embedding_service.encode_documents(texts_to_embed)

    vector_store = InMemoryVectorStore(index_dir=index_dir, docs_dir=docs_dir)
    vector_store.save(all_chunks, vectors)

    logger.info(f"Ingestão concluída com sucesso! Artefatos gravados em: {index_dir}")
    return len(all_chunks)


if __name__ == "__main__":
    count = run_ingestion()
    print(f"Sucesso! {count} chunks indexados.")
