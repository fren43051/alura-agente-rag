"""Tarea 1 — Colecta y carga de documentos multi-formato."""

import os
from pathlib import Path
from typing import List
from loguru import logger

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    BSHTMLLoader,
    TextLoader,
    JSONLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "word",
    ".xlsx": "excel",
    ".xls": "excel",
    ".pptx": "powerpoint",
    ".csv": "csv",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".txt": "text",
}


def load_document(file_path: str) -> List[Document]:
    """Carga un documento individual según su extensión."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no soportado: {ext}")

    fmt = SUPPORTED_EXTENSIONS[ext]
    logger.info(f"Cargando {fmt.upper()}: {path.name}")

    loaders = {
        "pdf":        lambda: PyMuPDFLoader(file_path).load(),
        "word":       lambda: Docx2txtLoader(file_path).load(),
        "excel":      lambda: UnstructuredExcelLoader(file_path).load(),
        "powerpoint": lambda: UnstructuredPowerPointLoader(file_path).load(),
        "csv":        lambda: CSVLoader(file_path).load(),
        "json":       lambda: JSONLoader(file_path, jq_schema=".", text_content=False).load(),
        "html":       lambda: BSHTMLLoader(file_path).load(),
        "markdown":   lambda: TextLoader(file_path, encoding="utf-8").load(),
        "text":       lambda: TextLoader(file_path, encoding="utf-8").load(),
    }

    docs = loaders[fmt]()

    # Enriquecer metadata
    for doc in docs:
        doc.metadata.update({
            "source_file": path.name,
            "file_format": fmt,
            "file_path": str(path.absolute()),
        })

    logger.success(f"Cargados {len(docs)} fragmentos de {path.name}")
    return docs


def load_directory(docs_dir: str) -> List[Document]:
    """Carga todos los documentos soportados de un directorio."""
    docs_path = Path(docs_dir)
    all_docs: List[Document] = []
    skipped = []

    for file in docs_path.rglob("*"):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(str(file))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Error cargando {file.name}: {e}")
                skipped.append(file.name)

    logger.info(f"Total documentos cargados: {len(all_docs)} | Omitidos: {len(skipped)}")
    return all_docs


if __name__ == "__main__":
    import os
    docs_dir = os.getenv("DOCS_DIR", "./docs/sample_docs")
    documents = load_directory(docs_dir)
    print(f"\n✅ {len(documents)} documentos cargados")
