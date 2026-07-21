"""Cargadores de documentos multi-formato para el pipeline RAG."""
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from src.config import SUPPORTED_FORMATS


class DocumentLoader:
    """
    Carga documentos corporativos en múltiples formatos y retorna
    objetos Document de LangChain con metadatos enriquecidos.

    Formatos soportados:
        PDF, DOCX, XLSX, PPTX, MD, TXT, CSV, JSON, HTML
    """

    def load_file(self, file_path: str) -> List[Document]:
        """Carga un archivo según su extensión."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f'Archivo no encontrado: {file_path}')

        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f'Formato no soportado: {ext}')

        loaders = {
            '.pdf':  self._load_pdf,
            '.docx': self._load_docx,
            '.doc':  self._load_docx,
            '.xlsx': self._load_excel,
            '.xls':  self._load_excel,
            '.pptx': self._load_pptx,
            '.ppt':  self._load_pptx,
            '.md':   self._load_text,
            '.txt':  self._load_text,
            '.csv':  self._load_csv,
            '.json': self._load_json,
            '.html': self._load_html,
        }

        docs = loaders[ext](str(path))
        return self._enrich_metadata(docs, path)

    def load_directory(self, dir_path: str) -> List[Document]:
        """Carga todos los documentos soportados de un directorio."""
        path = Path(dir_path)
        all_docs = []
        for ext in SUPPORTED_FORMATS:
            for file in path.rglob(f'*{ext}'):
                try:
                    docs = self.load_file(str(file))
                    all_docs.extend(docs)
                    print(f'  [OK] {file.name} — {len(docs)} fragmentos')
                except Exception as e:
                    print(f'  [ERROR] {file.name}: {e}')
        return all_docs

    # --- Loaders internos ---

    def _load_pdf(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader(path).load()

    def _load_docx(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(path).load()

    def _load_excel(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(path, mode='elements').load()

    def _load_pptx(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        return UnstructuredPowerPointLoader(path).load()

    def _load_text(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(path, encoding='utf-8').load()

    def _load_csv(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(path, encoding='utf-8').load()

    def _load_json(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import JSONLoader
        return JSONLoader(path, jq_schema='.', text_content=False).load()

    def _load_html(self, path: str) -> List[Document]:
        from langchain_community.document_loaders import BSHTMLLoader
        return BSHTMLLoader(path).load()

    def _enrich_metadata(self, docs: List[Document], path: Path) -> List[Document]:
        """Agrega metadatos enriquecidos a cada documento."""
        stat = path.stat()
        for doc in docs:
            doc.metadata.update({
                'source_file': path.name,
                'source_path': str(path),
                'file_type':   path.suffix.lower().lstrip('.'),
                'file_size_kb': round(stat.st_size / 1024, 2),
                'indexed_at':  datetime.utcnow().isoformat(),
            })
        return docs
