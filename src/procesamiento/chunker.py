"""
Etapa 2 — Procesamiento y chunking.
Divide documentos en fragmentos óptimos para RAG.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def crear_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """Crea un splitter con parámetros configurables."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def procesar_documentos(
    documentos: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Divide y limpia una lista de documentos."""
    splitter = crear_chunker(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documentos)

    # Limpieza básica de contenido
    chunks_limpios = []
    for chunk in chunks:
        texto = chunk.page_content.strip()
        if len(texto) > 50:  # Descarta fragmentos muy cortos
            chunk.page_content = texto
            chunks_limpios.append(chunk)

    print(f"✂️  Chunks generados: {len(chunks_limpios)} (de {len(documentos)} docs)")
    return chunks_limpios
