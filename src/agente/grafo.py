"""
Etapa 5 — Agente LangGraph.
Define el grafo de estados del agente RAG corporativo.
"""

import os
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.recuperacion.retriever import buscar_documentos


# --- Estado del agente ---
class EstadoAgente(TypedDict):
    messages: Annotated[list, add_messages]
    pregunta: str
    documentos: List[Document]
    respuesta: str


# --- Nodos del grafo ---
def nodo_recuperar(estado: EstadoAgente, retriever) -> EstadoAgente:
    """Recupera documentos relevantes para la pregunta."""
    docs = buscar_documentos(retriever, estado["pregunta"])
    return {**estado, "documentos": docs}


def nodo_generar(estado: EstadoAgente) -> EstadoAgente:
    """Genera respuesta usando el LLM con contexto RAG."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    contexto = "\n\n".join(
        [f"[Fuente: {d.metadata.get('fuente', 'N/A')}]\n{d.page_content}"
         for d in estado["documentos"]]
    )

    prompt = f"""Eres un asistente corporativo. Responde la pregunta basándote
    EXCLUSIVAMENTE en los documentos internos proporcionados.
    Si no encuentras la información, dilo claramente.

    DOCUMENTOS:
    {contexto}

    PREGUNTA: {estado['pregunta']}

    RESPUESTA:"""

    respuesta = llm.invoke([HumanMessage(content=prompt)])
    return {**estado, "respuesta": respuesta.content}


def construir_grafo(retriever) -> StateGraph:
    """Construye y compila el grafo del agente."""
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("recuperar", lambda s: nodo_recuperar(s, retriever))
    grafo.add_node("generar", nodo_generar)

    grafo.set_entry_point("recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    return grafo.compile()
