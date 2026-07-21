"""Configuración central del proyecto AluraAgente RAG."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / 'docs'
DATA_DIR = BASE_DIR / 'data'
VECTOR_DB_DIR = BASE_DIR / 'vectordb'

# Modelo de lenguaje
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Embeddings
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1536'))

# Vector DB
VECTOR_STORE = os.getenv('VECTOR_STORE', 'chroma')  # chroma | pinecone | faiss
CHROMA_PERSIST_DIR = str(VECTOR_DB_DIR / 'chroma')

# Chunking
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))

# Recuperación RAG
RETRIEVER_K = int(os.getenv('RETRIEVER_K', '5'))  # top-K documentos
RETRIEVER_SCORE_THRESHOLD = float(os.getenv('RETRIEVER_SCORE_THRESHOLD', '0.7'))

# OCI
OCI_REGION = os.getenv('OCI_REGION', 'us-ashburn-1')
OCI_BUCKET = os.getenv('OCI_BUCKET', '')
OCI_NAMESPACE = os.getenv('OCI_NAMESPACE', '')

# Formatos de documentos soportados
SUPPORTED_FORMATS = [
    '.pdf', '.docx', '.doc',
    '.xlsx', '.xls',
    '.pptx', '.ppt',
    '.md', '.txt',
    '.csv', '.json', '.html'
]

if __name__ == '__main__':
    print(f'[Config] Proyecto: AluraAgente RAG v{__import__("src").__version__}')
    print(f'[Config] LLM: {LLM_PROVIDER}/{LLM_MODEL}')
    print(f'[Config] Vector Store: {VECTOR_STORE}')
    print(f'[Config] Formatos soportados: {len(SUPPORTED_FORMATS)}')
