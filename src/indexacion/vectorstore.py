"""
Etapa 3 — Indexación vectorial.
Genera embeddings y almacena en ChromaDB.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chromadb")
COLECCION = os.getenv("CHROMA_COLLECTION", "alura_agente_docs")


def crear_embeddings() -> OpenAIEmbeddings:
    """Inicializa el modelo de embeddings."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def indexar_documentos(chunks: List[Document]) -> Chroma:
    """Genera embeddings e indexa los chunks en ChromaDB."""
    embeddings = crear_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLECCION,
        persist_directory=CHROMA_DIR,
    )
    print(f"🗂️  Indexados {len(chunks)} chunks en ChromaDB → {CHROMA_DIR}")
    return vectorstore


def cargar_vectorstore() -> Chroma:
    """Carga un vectorstore existente desde disco."""
    embeddings = crear_embeddings()
    return Chroma(
        collection_name=COLECCION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
