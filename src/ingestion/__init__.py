"""Módulo de ingesta de documentos."""
from .loader import DocumentLoader
from .chunker import DocumentChunker

__all__ = ['DocumentLoader', 'DocumentChunker']
