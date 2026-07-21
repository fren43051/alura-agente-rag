"""Tarea 5 — Agente LangGraph con herramientas RAG."""

import os
from typing import Annotated, TypedDict
from loguru import logger
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rag_chain import build_rag_chain, load_vectorstore

load_dotenv()


# --- Estado del grafo ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    documents_searched: bool


# --- Herramientas del agente ---
@tool
def search_documents(query: str) -> str:
    """Busca información relevante en los documentos corporativos indexados.
    
    Usa esta herramienta cuando el usuario pregunte sobre políticas, 
    procedimientos, información de la empresa o cualquier dato interno.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "No se encontraron documentos relevantes para esta consulta."

    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "desconocido")
        results.append(f"[{source}]\n{doc.page_content[:500]}...")

    return "\n\n".join(results)


@tool
def list_available_documents() -> str:
    """Lista los documentos corporativos disponibles en el sistema."""
    docs_dir = os.getenv("DOCS_DIR", "./docs/sample_docs")
    from pathlib import Path
    from document_loader import SUPPORTED_EXTENSIONS

    files = [
        f.name for f in Path(docs_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        return "No hay documentos indexados en el sistema."

    return f"Documentos disponibles ({len(files)}):\n" + "\n".join(f"• {f}" for f in files)


# --- Nodos del grafo ---
def agent_node(state: AgentState):
    """Nodo principal del agente."""
    from rag_chain import get_llm
    from langchain_core.messages import SystemMessage

    llm = get_llm()
    tools = [search_documents, list_available_documents]
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(
        content="""Eres el asistente corporativo AluraAgente. 
        Ayudas a los colaboradores respondiendo preguntas sobre documentos internos.
        Siempre usa las herramientas disponibles para buscar información antes de responder.
        Responde siempre en español y cita las fuentes de los documentos."""
    )

    all_messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Decide si continuar con herramientas o terminar."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# --- Construcción del grafo ---
def build_agent():
    tools = [search_documents, list_available_documents]
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    agent = graph.compile()
    logger.success("Agente LangGraph compilado")
    return agent


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content="¿Qué documentos tengo disponibles?")],
        "documents_searched": False,
    })
    print(result["messages"][-1].content)
