"""
vectorstore.py — Gestión del vectorstore ChromaDB con embeddings configurables.
Soporta embeddings locales (sentence-transformers) y de OpenAI.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import settings

logger = logging.getLogger(__name__)


def _get_embeddings() -> Embeddings:
    """Retorna el modelo de embeddings configurado en Settings."""
    if settings.openai_api_key and "text-embedding" in settings.embedding_model:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Usando embeddings de OpenAI: %s", settings.embedding_model)
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Usando embeddings locales: %s", settings.embedding_model)
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Retorna la instancia singleton del vectorstore ChromaDB.
    Persiste en disco en la ruta configurada.
    """
    embeddings = _get_embeddings()
    vs = Chroma(
        collection_name="alura_rag_docs",
        embedding_function=embeddings,
        persist_directory=str(settings.vectorstore_dir),
    )
    logger.info(
        "VectorStore listo: %s | colección: alura_rag_docs",
        settings.vectorstore_dir,
    )
    return vs


def reset_vectorstore() -> None:
    """Elimina y reinicia el vectorstore (útil para tests y desarrollo)."""
    import shutil

    get_vectorstore.cache_clear()
    vs_path = settings.vectorstore_dir
    if vs_path.exists():
        shutil.rmtree(vs_path)
        logger.warning("VectorStore eliminado: %s", vs_path)
    settings.ensure_dirs()
