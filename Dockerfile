# ============================================
# AluraAgente RAG — Dockerfile
# Compatible con Oracle Cloud Infrastructure
# ============================================

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY .env.example .env

# Directorio para documentos y ChromaDB
RUN mkdir -p /app/docs/sample_docs /app/chroma_db

# Puerto de la aplicación
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/ || exit 1

# Comando de inicio
CMD ["python", "src/app.py"]
