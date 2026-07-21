"""Tests para el cargador de documentos."""

import pytest
from pathlib import Path
import tempfile
import os

# Agregar src al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_loader import load_document, SUPPORTED_EXTENSIONS


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".json" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


def test_load_markdown():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\n\nEste es un documento de prueba.")
        tmp_path = f.name

    try:
        docs = load_document(tmp_path)
        assert len(docs) > 0
        assert "Test" in docs[0].page_content
    finally:
        os.unlink(tmp_path)


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Formato no soportado"):
        load_document("archivo.xyz")
