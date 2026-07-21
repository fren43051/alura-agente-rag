# 🤖 AluraAgente RAG — Agente Corporativo de Documentos

> **Challenge AluraAgente — ONE IA FOR TECH**  
> Agente de IA corporativo con arquitectura RAG para responder preguntas sobre documentos internos empresariales.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?logo=chainlink)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)](https://chromadb.com)
[![OCI](https://img.shields.io/badge/Deploy-Oracle_Cloud-red?logo=oracle)](https://oracle.com/cloud)

---

## 📋 Descripción

Este proyecto implementa un **Agente RAG (Retrieval-Augmented Generation)** que permite a los colaboradores de una empresa consultar documentos internos en lenguaje natural. El agente procesa múltiples formatos de archivo, los indexa en una base vectorial y responde preguntas con contexto preciso citando las fuentes.

### 🎯 Capacidades

| Formato | Librería | Estado |
|---------|----------|--------|
| PDF | PyMuPDF | 🔄 En desarrollo |
| Word (.docx) | python-docx | 🔄 En desarrollo |
| Excel (.xlsx) | openpyxl | 🔄 En desarrollo |
| PowerPoint (.pptx) | python-pptx | 🔄 En desarrollo |
| CSV | pandas | 🔄 En desarrollo |
| JSON | built-in | 🔄 En desarrollo |
| HTML | BeautifulSoup4 | 🔄 En desarrollo |
| Markdown | built-in | 🔄 En desarrollo |

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   ALURA AGENTE RAG                      │
│                                                         │
│  📂 Documentos → 🔧 Procesamiento → 🔢 Embeddings      │
│        │               │                  │             │
│        ▼               ▼                  ▼             │
│  [PDF/Word/Excel]  [Chunking]      [ChromaDB]           │
│  [CSV/JSON/HTML]   [Limpieza]      [Vectorstore]        │
│                                          │              │
│                                          ▼              │
│              Pregunta del usuario → [Retriever]         │
│                                          │              │
│                                          ▼              │
│                              [LangGraph Agent]          │
│                              [LangChain RAG Chain]      │
│                                          │              │
│                                          ▼              │
│                              Respuesta + Fuentes 📄     │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
alura-agente-rag/
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # Carga multi-formato
│   ├── chunking.py             # Procesamiento y chunking
│   ├── vectorstore.py          # Indexación ChromaDB
│   ├── rag_chain.py            # Cadena RAG LangChain
│   ├── agent.py                # Agente LangGraph
│   └── app.py                  # Interfaz Gradio
│
├── docs/
│   └── sample_docs/            # Documentos de prueba
│
├── tests/
│   ├── test_loader.py
│   ├── test_chunking.py
│   └── test_rag.py
│
├── notebooks/
│   └── exploracion.ipynb
│
└── deploy/
    └── oci-setup.sh            # Script deploy OCI
```

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/fren43051/alura-agente-rag.git
cd alura-agente-rag
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus API keys
```

### 5. Ejecutar la aplicación

```bash
python src/app.py
```

---

## 🔑 Variables de Entorno

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...         # Alternativa: Gemini
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs/sample_docs
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

---

## 🗺️ Roadmap — Tareas del Challenge

- [ ] **Tarea 1** — Colecta y organización de documentos
- [ ] **Tarea 2** — Procesamiento y extracción de contenido
- [ ] **Tarea 3** — Indexación vectorial con ChromaDB
- [ ] **Tarea 4** — Capa de recuperación RAG
- [ ] **Tarea 5** — Producción y validación de respuestas
- [ ] **Tarea 6** — Interfaz de usuario (Gradio)
- [ ] **Tarea 7** — Deploy en Oracle Cloud (OCI)
- [ ] **Tarea 8** — Registro de ejecución + README final

---

## 👨‍💻 Autor

**Nelson Reyes** — [@fren43051](https://github.com/fren43051)  
Edutek Academy · Challenge ONE IA FOR TECH · Alura Latam

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE)
