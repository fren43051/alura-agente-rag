"""
Etapa 4 — Recuperación RAG.
Configurar y ejecutar el retriever sobre el vectorstore.
"""

import os
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


def crear_retriever(
    vectorstore: Chroma,
    k: int = 5,
    tipo: str = "similarity",
) -> BaseRetriever:
    """
    Crea un retriever configurable.
    tipo: 'similarity' | 'mmr' (Maximum Marginal Relevance)
    """
    if tipo == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
        )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def buscar_documentos(retriever: BaseRetriever, pregunta: str) -> list:
    """Recupera los fragmentos más relevantes para una pregunta."""
    docs = retriever.invoke(pregunta)
    print(f"🔍 Recuperados {len(docs)} fragmentos para: '{pregunta[:60]}...'")
    return docs
