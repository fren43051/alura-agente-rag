# src/agent.py — LangGraph con memoria persistente + Claude 3.5 Haiku

import os
import uuid
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from src.tools import get_all_tools
import operator

load_dotenv()

# ─── Prompt del sistema ──────────────────────────────────────
SYSTEM_PROMPT = """Eres un Asistente Corporativo de Documentos inteligente para la empresa.
Tienes acceso a documentos internos indexados y 7 herramientas especializadas.

## Tus capacidades:
1. **search_documents**: Busca información en documentos corporativos con RAG
2. **analyze_data**: Analiza estadísticas de archivos CSV/Excel
3. **generate_chart**: Crea gráficos de tipo bar, line, scatter, pie, hist, box
4. **summarize_document**: Resume documentos completos indexados
5. **smart_chart_detector**: Detecta automáticamente el mejor tipo de gráfico
6. **cross_document_analyst**: Cruza información entre múltiples documentos
7. **error_retry_handler**: Maneja errores con retry automático (máx. 3 intentos)

## Reglas de comportamiento:
- Siempre cita el documento fuente cuando respondas (📄 Fuente: nombre_archivo)
- Si no encuentras información, dilo claramente y sugiere alternativas
- Para datos numéricos, ofrece proactivamente generar un gráfico
- Responde siempre en el idioma del usuario
- Sé conciso pero completo. Usa markdown para estructurar respuestas largas
- Si un análisis falla, usa error_retry_handler antes de rendirte

## Formato de respuesta con fuentes:
Cuando uses search_documents, incluye al final:
📎 **Referencia:** [nombre_del_documento]
"""

# ─── Estado del grafo ────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    referenced_docs: list
    session_id: str
    tools_used: Annotated[list, operator.add]

# ─── Inicialización del LLM ──────────────────────────────────
def create_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY no encontrada. "
            "Copia .env.example a .env y agrega tu API key."
        )
    return ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=api_key
    )

tools = get_all_tools()
memory = MemorySaver()

# ─── Nodos del grafo ─────────────────────────────────────────
def agent_node(state: AgentState):
    """Nodo principal: llama al LLM con herramientas disponibles."""
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)
    messages = state["messages"]
    # Insertar system prompt si es el primer mensaje
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    # Rastrear documentos citados
    referenced = state.get("referenced_docs", [])
    if "Fuente:" in str(response.content):
        import re
        found = re.findall(r'📄\s*\*\*Fuente:\*\*\s*([^\n]+)', str(response.content))
        found += re.findall(r'📎\s*\*\*Referencia:\*\*\s*([^\n]+)', str(response.content))
        referenced = list(set(referenced + found))
    return {
        "messages": [response],
        "referenced_docs": referenced,
        "tools_used": [tc["name"] for tc in (response.tool_calls or [])]
    }

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    """Decide si continuar usando herramientas o terminar."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ─── Construcción del grafo ───────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=memory)

app_agent = build_agent()

# ─── Función principal de ejecución ──────────────────────────
def run_agent(user_input: str, session_id: str = "default") -> tuple[str, list, list]:
    """
    Ejecuta el agente con memoria persistente por sesión.
    Returns:
        (respuesta_texto, documentos_citados, herramientas_usadas)
    """
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "referenced_docs": [],
        "session_id": session_id,
        "tools_used": []
    }
    try:
        result = app_agent.invoke(initial_state, config=config)
        last_msg = result["messages"][-1]
        response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        referenced_docs = result.get("referenced_docs", [])
        tools_used = result.get("tools_used", [])
        return response_text, referenced_docs, tools_used
    except Exception as e:
        error_msg = f"⚠️ Error del agente: {str(e)}\nPor favor intenta reformular tu pregunta."
        return error_msg, [], []

def reset_session(session_id: str) -> str:
    """Reinicia la memoria de una sesión específica."""
    # LangGraph MemorySaver no expone delete directo; creamos nuevo thread_id
    new_id = str(uuid.uuid4())
    return new_id
