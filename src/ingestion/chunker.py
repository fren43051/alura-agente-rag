"""Chunking inteligente de documentos para el pipeline RAG."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """
    Divide documentos en chunks optimizados para embeddings.
    Usa RecursiveCharacterTextSplitter con separadores
    jerárquicos para respetar la estructura del texto.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', ''],
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Divide una lista de documentos en chunks."""
        chunks = self.splitter.split_documents(documents)
        # Agregar índice de chunk a los metadatos
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_total'] = len(chunks)
        return chunks

    def split_file(self, file_path: str) -> List[Document]:
        """Carga y divide un archivo en un solo paso."""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_file(file_path)
        return self.split(docs)
