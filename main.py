"""
AluraAgente RAG — Punto de entrada principal.
Ejecuta el pipeline completo: Colecta → Procesamiento → Indexación → Agente.
"""

import sys
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, RETRIEVER_TIPO
from src.colecta.cargador import cargar_directorio
from src.procesamiento.chunker import procesar_documentos
from src.indexacion.vectorstore import indexar_documentos, cargar_vectorstore
from src.recuperacion.retriever import crear_retriever
from src.agente.grafo import construir_grafo


def pipeline_indexacion():
    """Ejecuta el pipeline completo de indexación."""
    print("\n🚀 ALURA AGENTE RAG — Pipeline de Indexación")
    print("=" * 50)

    # Etapa 1: Colecta
    print("\n📂 Etapa 1: Colectando documentos...")
    documentos = cargar_directorio(DOCS_DIR)

    # Etapa 2: Procesamiento
    print("\n✂️  Etapa 2: Procesando y dividiendo...")
    chunks = procesar_documentos(documentos, CHUNK_SIZE, CHUNK_OVERLAP)

    # Etapa 3: Indexación
    print("\n🗂️  Etapa 3: Indexando en ChromaDB...")
    indexar_documentos(chunks)

    print("\n✅ Indexación completada con éxito!")


def pipeline_consulta(pregunta: str):
    """Responde una pregunta usando el agente RAG."""
    print(f"\n🤖 ALURA AGENTE RAG — Consulta")
    print("=" * 50)
    print(f"❓ Pregunta: {pregunta}\n")

    # Carga vectorstore existente
    vectorstore = cargar_vectorstore()
    retriever   = crear_retriever(vectorstore, RETRIEVER_K, RETRIEVER_TIPO)
    agente      = construir_grafo(retriever)

    # Ejecuta el agente
    resultado = agente.invoke({"pregunta": pregunta, "messages": [], "documentos": [], "respuesta": ""})

    print(f"\n💬 Respuesta:\n{resultado['respuesta']}")
    return resultado["respuesta"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "indexar":
            pipeline_indexacion()
        else:
            pipeline_consulta(" ".join(sys.argv[1:]))
    else:
        print("Uso:")
        print("  python main.py indexar               # Indexa todos los documentos")
        print("  python main.py '¿Cuál es la política de vacaciones?'  # Hace una consulta")
