"""Tarea 2 — Procesamiento y chunking inteligente de documentos."""

from typing import List
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Divide documentos en chunks con estrategia adaptativa por formato."""
    chunks: List[Document] = []

    markdown_docs = [d for d in documents if d.metadata.get("file_format") == "markdown"]
    other_docs = [d for d in documents if d.metadata.get("file_format") != "markdown"]

    # Markdown: split por headers primero
    if markdown_docs:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        for doc in markdown_docs:
            md_chunks = md_splitter.split_text(doc.page_content)
            for chunk in md_chunks:
                chunk.metadata.update(doc.metadata)
            chunks.extend(md_chunks)
        logger.info(f"Markdown: {len(markdown_docs)} docs → {len(chunks)} chunks")

    # Resto de formatos: RecursiveCharacterTextSplitter
    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,
        )
        other_chunks = splitter.split_documents(other_docs)
        chunks.extend(other_chunks)
        logger.info(f"Otros formatos: {len(other_docs)} docs → {len(other_chunks)} chunks")

    # Limpiar chunks vacíos
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    logger.success(f"Total chunks generados: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    from document_loader import load_directory
    import os
    docs = load_directory(os.getenv("DOCS_DIR", "./docs/sample_docs"))
    chunks = chunk_documents(docs)
    print(f"\n✅ {len(chunks)} chunks generados")
    print(f"   Tamaño promedio: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
