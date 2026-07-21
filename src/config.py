"""
Configuración centralizada del proyecto.
Carga variables de entorno y define constantes globales.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Directorios
DOCS_DIR   = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chromadb")

# Modelos
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Chunking
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
RETRIEVER_K    = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_TIPO = os.getenv("RETRIEVER_TIPO", "mmr")

print(f"⚙️  Config cargada | LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
