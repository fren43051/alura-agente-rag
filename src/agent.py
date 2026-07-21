# src/agent.py — LangGraph con memoria persistente + Claude 3.5 Haiku
import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from src.tools import get_all_tools

load_dotenv()

# ─── Estado del grafo ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    referenced_docs: list
    session_id: str


# ─── LLM: Claude 3.5 Haiku ───────────────────────────────────────
llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0,
    max_tokens=4096,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

SYSTEM_PROMPT = """Eres un asistente corporativo inteligente especializado en análisis de documentos empresariales.

Capacidades:
- Buscar información en documentos internos (PDFs, Excel, Word, CSV, etc.)
- Analizar datos y generar visualizaciones automáticas
- Cruzar información entre múltiples documentos
- Mantener historial de conversación en la sesión

Instrucciones:
1. SIEMPRE usa las herramientas disponibles antes de responder
2. Cita el nombre del documento fuente en CADA respuesta (ej: "Según ventas.xlsx...")
3. Si una herramienta falla, usa error_retry_handler automáticamente
4. Para preguntas que involucren múltiples documentos, usa cross_document_analyst
5. Para visualizaciones, prefiere smart_chart_detector sobre generate_chart básico
6. Responde siempre en español con formato claro y profesional
7. Usa emojis con moderación para mejorar legibilidad"""

# ─── Herramientas ────────────────────────────────────────────────
tools = get_all_tools()
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()


# ─── Nodos del grafo ─────────────────────────────────────────────
def call_model(state: AgentState):
    """Nodo principal: llama al LLM con historial completo."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    # Extraer documentos citados del response
    referenced = state.get("referenced_docs", [])
    if hasattr(response, "content") and isinstance(response.content, str):
        import re
        doc_pattern = re.compile(r'\b\w+\.(pdf|xlsx|xls|docx|csv|md|json|html)\b', re.IGNORECASE)
        found_docs = doc_pattern.findall(response.content)
        # Re-buscar los nombres completos
        full_names = doc_pattern.findall(response.content)
        new_refs = [m.group() for m in doc_pattern.finditer(response.content)]
        for ref in new_refs:
            if ref not in referenced:
                referenced.append(ref)
    return {"messages": [response], "referenced_docs": referenced}


def should_continue(state: AgentState):
    """Decide si continuar con herramientas o terminar."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─── Construcción del grafo ───────────────────────────────────────
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

# Compilar con memoria persistente
app_agent = workflow.compile(checkpointer=memory)


# ─── Función principal de ejecución ──────────────────────────────
def run_agent(user_input: str, session_id: str = "default") -> tuple[str, list]:
    """
    Ejecuta el agente con memoria de sesión.
    Retorna (respuesta_texto, lista_documentos_citados)
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = app_agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "referenced_docs": [],
                "session_id": session_id
            },
            config=config
        )
        last_message = result["messages"][-1]
        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)
        referenced_docs = result.get("referenced_docs", [])
        return response_text, referenced_docs
    except Exception as e:
        error_msg = (
            f"❌ Error en el agente: {str(e)}\n\n"
            f"Por favor intenta reformular tu pregunta o verifica que haya documentos indexados."
        )
        return error_msg, []


def get_session_history(session_id: str) -> list:
    """Retorna el historial de mensajes de una sesión."""
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = app_agent.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []
    except Exception:
        return []
