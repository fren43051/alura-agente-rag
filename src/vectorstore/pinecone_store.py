"""Gestión del vector store con Pinecone para el pipeline RAG.

Índice  : quickstart
Endpoint: langchain-rag-zkzmbte.svc.aped-4627-b74a.pinecone.io
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

# ── Constantes Pinecone ───────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv('PINECONE_API_KEY', '')
PINECONE_INDEX    = os.getenv('PINECONE_INDEX', 'quickstart')
PINECONE_HOST     = 'langchain-rag-zkzmbte.svc.aped-4627-b74a.pinecone.io'
PINECONE_CLOUD    = os.getenv('PINECONE_CLOUD', 'aws')
PINECONE_REGION   = os.getenv('PINECONE_REGION', 'us-east-1')
BATCH_SIZE        = int(os.getenv('PINECONE_BATCH_SIZE', '100'))


class PineconeStore:
    """
    Wrapper sobre LangChain PineconeVectorStore.

    Uso:
        store = PineconeStore()
        store.upsert(chunks)           # indexar documentos
        results = store.search(query)  # búsqueda semántica
    """

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self._pc    = Pinecone(api_key=PINECONE_API_KEY)
        self._index = self._get_or_create_index()
        self._store = PineconeVectorStore(
            index=self._index,
            embedding=self._embeddings,
            text_key='page_content',
        )
        print(f'[Pinecone] Conectado al índice "{PINECONE_INDEX}" ✓')

    # ── Índice ────────────────────────────────────────────────────────────────

    def _get_or_create_index(self):
        """Obtiene el índice existente o crea uno nuevo."""
        existing = [i.name for i in self._pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            print(f'[Pinecone] Creando índice "{PINECONE_INDEX}"...')
            self._pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric='cosine',
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        return self._pc.Index(PINECONE_INDEX, host=PINECONE_HOST)

    # ── Escritura ─────────────────────────────────────────────────────────────

    def upsert(self, documents: List[Document]) -> int:
        """
        Indexa una lista de Document chunks en Pinecone.
        Procesa en lotes para evitar timeouts.
        Retorna el total de vectores insertados.
        """
        total = 0
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            self._store.add_documents(batch)
            total += len(batch)
            print(f'  [Pinecone] Lote {i // BATCH_SIZE + 1}: {total}/{len(documents)} vectores')
        print(f'[Pinecone] Indexación completa: {total} vectores ✓')
        return total

    def upsert_from_directory(self, dir_path: str) -> int:
        """Carga, divide e indexa todos los docs de un directorio."""
        from src.ingestion.loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker

        print(f'[Pipeline] Cargando documentos desde: {dir_path}')
        loader  = DocumentLoader()
        chunker = DocumentChunker()

        docs   = loader.load_directory(dir_path)
        chunks = chunker.split(docs)
        print(f'[Pipeline] {len(docs)} docs → {len(chunks)} chunks')
        return self.upsert(chunks)

    # ── Lectura ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Búsqueda semántica por similitud coseno."""
        return self._store.similarity_search(query, k=k, filter=filter)

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
    ) -> List[tuple[Document, float]]:
        """Búsqueda con puntuación de relevancia."""
        results = self._store.similarity_search_with_score(query, k=k)
        return [(doc, score) for doc, score in results if score >= score_threshold]

    def as_retriever(self, k: int = 5, score_threshold: float = 0.7):
        """Retorna un retriever compatible con LangChain chains."""
        return self._store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                'k': k,
                'score_threshold': score_threshold,
            },
        )

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Retorna estadísticas del índice Pinecone."""
        return self._index.describe_index_stats()
