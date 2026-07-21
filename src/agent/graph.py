"""Grafo LangGraph para el agente RAG de Mercado Central 24h.

Flujo:
    START → retrieve → generate → validate → END
                              └─ (sin contexto) → fallback → END
"""
from __future__ import annotations

from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.vectorstore.pinecone_store import PineconeStore
from src.retriever.rag_chain import _format_docs, SYSTEM_PROMPT


# ── Estado del grafo ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:     str
    context:      List[Document]
    answer:       str
    sources:      List[str]
    messages:     Annotated[list, operator.add]
    has_context:  bool


# ── Nodos del grafo ─────────────────────────────────────────────────────────

_store = None  # singleton

def _get_store() -> PineconeStore:
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


def node_retrieve(state: AgentState) -> AgentState:
    """Recupera documentos relevantes de Pinecone."""
    store   = _get_store()
    results = store.search_with_score(state['question'], k=5, score_threshold=0.65)
    docs    = [doc for doc, _ in results]
    sources = list({d.metadata.get('source_file', '') for d in docs})
    return {
        **state,
        'context':     docs,
        'sources':     sources,
        'has_context': len(docs) > 0,
    }


def node_generate(state: AgentState) -> AgentState:
    """Genera una respuesta con el LLM usando el contexto recuperado."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

    context_text = _format_docs(state['context']) if state['context'] else ''
    messages = [
        HumanMessage(content=(
            f'{SYSTEM_PROMPT.format(context=context_text)}\n\n'
            f'Pregunta: {state["question"]}'
        ))
    ]
    response = llm.invoke(messages)
    return {
        **state,
        'answer':   response.content,
        'messages': [AIMessage(content=response.content)],
    }


def node_fallback(state: AgentState) -> AgentState:
    """Respuesta de fallback cuando no hay contexto relevante."""
    fallback_msg = (
        'No encontré información sobre ese tema en los documentos disponibles '
        'de Mercado Central 24h. Por favor, consulta con tu supervisor o '
        'el área correspondiente.'
    )
    return {
        **state,
        'answer':   fallback_msg,
        'messages': [AIMessage(content=fallback_msg)],
    }


def node_validate(state: AgentState) -> AgentState:
    """Valida que la respuesta sea coherente con el contexto."""
    # En producción: aquí se puede añadir guardrails o factuality check
    return state


# ── Enrutamiento condicional ─────────────────────────────────────────────────

def route_after_retrieve(state: AgentState) -> str:
    """Decide si generar respuesta o ejecutar fallback."""
    return 'generate' if state['has_context'] else 'fallback'


# ── Constructor del grafo ─────────────────────────────────────────────────────

def build_agent_graph():
    """
    Construye y compila el grafo LangGraph del agente RAG.

    Flujo:
        START → retrieve → [generate | fallback] → validate → END
    """
    builder = StateGraph(AgentState)

    # Nodos
    builder.add_node('retrieve', node_retrieve)
    builder.add_node('generate', node_generate)
    builder.add_node('fallback', node_fallback)
    builder.add_node('validate', node_validate)

    # Aristas
    builder.add_edge(START, 'retrieve')
    builder.add_conditional_edges(
        'retrieve',
        route_after_retrieve,
        {'generate': 'generate', 'fallback': 'fallback'},
    )
    builder.add_edge('generate', 'validate')
    builder.add_edge('fallback', 'validate')
    builder.add_edge('validate', END)

    return builder.compile()


if __name__ == '__main__':
    graph = build_agent_graph()
    result = graph.invoke({
        'question': '¿Cuál es la política de devoluciones de Mercado Central?',
        'context': [], 'answer': '', 'sources': [], 'messages': [], 'has_context': False,
    })
    print('Respuesta:', result['answer'])
    print('Fuentes:',   result['sources'])
