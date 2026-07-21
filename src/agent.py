"""
agent.py — Agente RAG corporativo con LangGraph + Claude 3.5 Haiku
Arquitectura: StateGraph con memoria persistente, retry automático y logging estructurado.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .config import Settings
from .tools import get_all_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
settings = Settings()

SYSTEM_PROMPT = """
Eres un asistente corporativo especializado en análisis de documentos internos.
Tu función es responder preguntas de colaboradores utilizando únicamente la
información disponible en los documentos indexados de la empresa.

Reglas de comportamiento:
1. SIEMPRE cita la fuente (nombre del documento + sección) al responder.
2. Si la información no está en los documentos, dilo explícitamente.
3. Para preguntas cuantitativas, genera automáticamente el gráfico más adecuado.
4. Responde en el mismo idioma que el usuario.
5. Sé conciso pero completo. Usa markdown para estructurar respuestas largas.
6. En caso de ambigüedad, solicita aclaración antes de proceder.

Formato de citas:
📎 **Fuente:** `{nombre_documento}` — sección "{sección}"
"""

# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    referenced_docs: list[str]      # documentos citados en la respuesta
    chart_paths: list[str]          # rutas de gráficos generados
    retry_count: int                # intentos de retry en nodo actual
    session_metadata: dict[str, Any]  # tokens, timing, herramientas usadas


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------

def _build_llm() -> ChatAnthropic:
    """Construye el LLM con las herramientas vinculadas."""
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=4096,
        timeout=60,
    )
    tools = get_all_tools()
    return llm.bind_tools(tools)


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Nodo principal: invoca el LLM con el historial completo."""
    start = time.perf_counter()
    llm_with_tools = _build_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    try:
        response: AIMessage = llm_with_tools.invoke(messages, config=config)
    except Exception as exc:
        logger.error("LLM invocation failed: %s", exc)
        retry = state.get("retry_count", 0)
        if retry < settings.max_retries:
            logger.warning("Retrying (%d/%d)...", retry + 1, settings.max_retries)
            time.sleep(2 ** retry)  # backoff exponencial
            return {
                "messages": [],
                "retry_count": retry + 1,
                "session_metadata": state.get("session_metadata", {}),
            }
        raise

    elapsed = time.perf_counter() - start
    meta = state.get("session_metadata", {})
    meta["last_latency_ms"] = round(elapsed * 1000, 2)
    meta["tools_called"] = meta.get("tools_called", 0)

    # Extraer documentos referenciados del contenido de la respuesta
    docs_cited: list[str] = []
    if isinstance(response.content, str) and "Fuente:" in response.content:
        import re
        matches = re.findall(r"`([^`]+\.(?:pdf|xlsx|docx|csv|md|json|html))`",
                             response.content, re.IGNORECASE)
        docs_cited = list(set(matches))

    return {
        "messages": [response],
        "referenced_docs": docs_cited,
        "retry_count": 0,
        "session_metadata": meta,
    }


def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    """Nodo de ejecución de herramientas con seguimiento de métricas."""
    tool_node = ToolNode(get_all_tools())
    result = tool_node.invoke(state, config=config)

    meta = state.get("session_metadata", {})
    meta["tools_called"] = meta.get("tools_called", 0) + 1

    # Detectar rutas de gráficos generados en los mensajes de herramienta
    chart_paths = list(state.get("chart_paths", []))
    for msg in result.get("messages", []):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("CHART:"):
            path = content.replace("CHART:", "").strip()
            if path not in chart_paths:
                chart_paths.append(path)

    return {
        **result,
        "chart_paths": chart_paths,
        "session_metadata": meta,
    }


def should_continue(state: AgentState) -> str:
    """Enrutador: decide si continuar con herramientas o terminar."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construye y compila el StateGraph con checkpointing en memoria."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Instancia global del grafo compilado
_compiled_graph = build_graph()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    content: str
    referenced_docs: list[str] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    session_metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def run_agent(user_input: str, session_id: str | None = None) -> AgentResponse:
    """
    Punto de entrada principal del agente.

    Args:
        user_input: Pregunta o instrucción del usuario.
        session_id: ID de sesión para mantener historial. Si es None, crea uno nuevo.

    Returns:
        AgentResponse con el contenido, fuentes citadas y rutas de gráficos.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info("Nueva sesión creada: %s", session_id)

    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "referenced_docs": [],
        "chart_paths": [],
        "retry_count": 0,
        "session_metadata": {"session_id": session_id},
    }

    try:
        result = _compiled_graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Agent graph execution failed: %s", exc)
        return AgentResponse(
            content=f"❌ Error al procesar tu solicitud: {exc}\nPor favor intenta reformular tu pregunta.",
            session_id=session_id,
        )

    last_msg = result["messages"][-1]
    content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    return AgentResponse(
        content=content,
        referenced_docs=result.get("referenced_docs", []),
        chart_paths=result.get("chart_paths", []),
        session_metadata=result.get("session_metadata", {}),
        session_id=session_id,
    )


def clear_session(session_id: str) -> None:
    """Elimina el historial de una sesión del checkpointer."""
    # MemorySaver no expone delete directo; recreamos el grafo para limpiar.
    global _compiled_graph
    logger.info("Sesión %s limpiada (reiniciando checkpointer)", session_id)
    _compiled_graph = build_graph()
