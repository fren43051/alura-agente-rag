"""
Etapa 1 — Colecta y organización de documentos.
Carga documentos corporativos en múltiples formatos:
PDF, Word, Excel, PowerPoint, Markdown, CSV, JSON, HTML.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
)

# Mapeo de extensión → loader
LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".csv":  CSVLoader,
    ".json": JSONLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm":  UnstructuredHTMLLoader,
}


def cargar_documento(ruta: str) -> List[Document]:
    """Carga un documento individual según su extensión."""
    ext = Path(ruta).suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"Formato no soportado: {ext}")
    loader = LOADER_MAP[ext](ruta)
    docs = loader.load()
    # Agrega metadata de origen
    for doc in docs:
        doc.metadata["fuente"] = ruta
        doc.metadata["formato"] = ext.lstrip(".")
    return docs


def cargar_directorio(directorio: str) -> List[Document]:
    """Carga recursivamente todos los documentos de un directorio."""
    todos = []
    for ruta in Path(directorio).rglob("*"):
        if ruta.suffix.lower() in LOADER_MAP:
            try:
                docs = cargar_documento(str(ruta))
                todos.extend(docs)
                print(f"✅ Cargado: {ruta.name} ({len(docs)} fragmentos)")
            except Exception as e:
                print(f"❌ Error en {ruta.name}: {e}")
    print(f"\n📁 Total documentos cargados: {len(todos)}")
    return todos
