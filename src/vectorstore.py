"""Tarea 3 — Indexación vectorial con ChromaDB."""

import os
from typing import List
from loguru import logger

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def get_embeddings():
    """Retorna modelo de embeddings según la API key disponible."""
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Usando OpenAI Embeddings")
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
    elif os.getenv("GOOGLE_API_KEY"):
        logger.info("Usando Google Generative AI Embeddings")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
    else:
        raise ValueError("Se requiere OPENAI_API_KEY o GOOGLE_API_KEY en .env")


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Crea o actualiza el vectorstore con los chunks indexados."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection = os.getenv("CHROMA_COLLECTION_NAME", "corporate_docs")

    embeddings = get_embeddings()

    logger.info(f"Indexando {len(chunks)} chunks en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )

    logger.success(f"Vectorstore creado: {persist_dir} | Colección: {collection}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Carga un vectorstore existente desde disco."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection = os.getenv("CHROMA_COLLECTION_NAME", "corporate_docs")
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection,
    )
    logger.info(f"Vectorstore cargado desde: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    from document_loader import load_directory
    from chunking import chunk_documents
    import os

    docs = load_directory(os.getenv("DOCS_DIR", "./docs/sample_docs"))
    chunks = chunk_documents(docs)
    vs = build_vectorstore(chunks)
    print(f"\n✅ Vectorstore construido con {len(chunks)} chunks")
