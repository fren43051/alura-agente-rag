"""Tests para el módulo de chunking."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from chunking import chunk_documents


def test_chunk_basic():
    docs = [
        Document(
            page_content="Este es un texto largo. " * 100,
            metadata={"source_file": "test.pdf", "file_format": "pdf"}
        )
    ]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 300  # con overlap


def test_chunk_filters_empty():
    docs = [
        Document(
            page_content="   ",
            metadata={"file_format": "pdf"}
        ),
        Document(
            page_content="Contenido válido con suficiente texto para pasar el filtro mínimo.",
            metadata={"file_format": "pdf"}
        ),
    ]
    chunks = chunk_documents(docs)
    assert all(len(c.page_content.strip()) > 50 for c in chunks)
