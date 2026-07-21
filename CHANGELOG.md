# Changelog — AluraAgente RAG

Todos los cambios notables de este proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [0.1.0] — 2026-07-21

### Agregado
- 🏗️ Estructura inicial del pipeline RAG por etapas
- `src/colecta/cargador.py` — Carga de documentos en 9 formatos (PDF, DOCX, XLSX, PPTX, MD, CSV, JSON, HTML)
- `src/procesamiento/chunker.py` — Chunking con `RecursiveCharacterTextSplitter`
- `src/indexacion/vectorstore.py` — Indexación con ChromaDB + OpenAI Embeddings
- `src/recuperacion/retriever.py` — Retriever con soporte para similarity y MMR
- `src/agente/grafo.py` — Grafo LangGraph con nodos recuperar → generar
- `src/config.py` — Configuración centralizada desde variables de entorno
- `main.py` — Punto de entrada con comandos `indexar` y consulta libre
- `CHANGELOG.md` — Registro de cambios del proyecto

### Pendiente
- [ ] Etapa 6: Interfaz de usuario (Streamlit / Gradio)
- [ ] Etapa 7: Deploy en OCI
- [ ] Etapa 8: README final y registro de ejecución
