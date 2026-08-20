"""Testes unitários para o pipeline de ingestão e chunking de documentos."""
import json
import tempfile
from pathlib import Path
from src.application.ingestion import (
    build_knowledge_catalog,
    run_ingestion,
    split_markdown_into_chunks,
)
from src.domain.models import ChunkKind


def test_split_markdown_chunks():
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as tmp:
        tmp.write("# Política de Teste\n\n## Seção 1\nTexto da seção 1 com detalhes.\n\n## Seção 2\nTexto da seção 2.")
        tmp_path = Path(tmp.name)

    try:
        chunks = split_markdown_into_chunks(tmp_path, target_chunk_size=100)
        assert len(chunks) >= 2
        assert chunks[0].section == "Seção 1"
        assert chunks[1].section == "Seção 2"
        assert "Texto da seção 1" in chunks[0].content
    finally:
        tmp_path.unlink()


def test_run_ingestion_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_docs_dir, tempfile.TemporaryDirectory() as tmp_idx_dir:
        docs_p = Path(tmp_docs_dir)
        idx_p = Path(tmp_idx_dir)

        test_file = docs_p / "manual.md"
        test_file.write_text("# Manual\n## Cap 1\nConteúdo informativo do manual.", encoding="utf-8")

        count = run_ingestion(docs_dir=docs_p, index_dir=idx_p)
        assert count == 2
        assert (idx_p / "chunks.json").exists()
        assert (idx_p / "vectors.npy").exists()
        assert (idx_p / "metadata.json").exists()
        metadata = json.loads((idx_p / "metadata.json").read_text(encoding="utf-8"))
        assert len(metadata["corpus_fingerprint"]) == 64
        chunks = json.loads((idx_p / "chunks.json").read_text(encoding="utf-8"))
        catalog = next(chunk for chunk in chunks if chunk["kind"] == "catalog")
        assert catalog["source"] == "base_interna"
        assert "Manual — Cap 1" in catalog["content"]


def test_catalog_is_derived_from_document_titles_and_sections(tmp_path):
    first = tmp_path / "politica_a.md"
    second = tmp_path / "politica_b.md"
    first.write_text("# Benefícios\n## Vale\nTexto.", encoding="utf-8")
    second.write_text("# Segurança\n## Acesso\nTexto.\n## Incidentes\nTexto.", encoding="utf-8")

    documents = [
        (first, split_markdown_into_chunks(first)),
        (second, split_markdown_into_chunks(second)),
    ]
    catalog = build_knowledge_catalog(documents)

    assert catalog.kind == ChunkKind.CATALOG
    assert "Benefícios — Vale" in catalog.content
    assert "Segurança — Acesso; Incidentes" in catalog.content
    assert "Como este assistente pode ajudar?" in catalog.retrieval_content
    assert "Benefícios; Segurança" in catalog.retrieval_content
