"""Cadena RAG con historial de conversación para Mercado Central 24h."""
from __future__ import annotations

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from src.config import LLM_MODEL, OPENAI_API_KEY, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD
from src.vectorstore.pinecone_store import PineconeStore

# ── Prompt del sistema ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres el asistente virtual corporativo de **Mercado Central 24h México**.
Respondes preguntas de colaboradores y proveedores basandote únicamente en los
documentos internos de la empresa.

Reglas:
- Responde siempre en español.
- Si la respuesta está en el contexto, cita el documento fuente.
- Si NO encuentras la respuesta en el contexto, di: "No encontré información sobre
  ese tema en los documentos disponibles. Por favor, consulta con tu supervisor."
- Nunca inventes datos, políticas o procedimientos.
- Sé conciso y profesional.

Contexto de documentos:
{context}
"""


def _format_docs(docs: List[Document]) -> str:
    """Formatea los documentos recuperados para el prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src  = doc.metadata.get('source_file', 'Documento desconocido')
        page = doc.metadata.get('page', '')
        ref  = f'{src} (pág. {page})' if page != '' else src
        parts.append(f'[Doc {i} — {ref}]\n{doc.page_content}')
    return '\n\n'.join(parts)


class RAGRetriever:
    """
    Cadena RAG conversacional con Pinecone + OpenAI.

    Uso:
        rag = RAGRetriever()
        respuesta = rag.ask('\u00bfCuál es la política de devoluciones?')
    """

    def __init__(self, session_id: str = 'default'):
        self._store     = PineconeStore()
        self._retriever = self._store.as_retriever(
            k=RETRIEVER_K,
            score_threshold=RETRIEVER_SCORE_THRESHOLD,
        )
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
        self._memory = ConversationBufferWindowMemory(
            k=6,
            memory_key='chat_history',
            return_messages=True,
        )
        self._chain  = self._build_chain()
        self.session_id = session_id

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system', SYSTEM_PROMPT),
            MessagesPlaceholder('chat_history'),
            ('human', '{question}'),
        ])
        return (
            {
                'context':      self._retriever | _format_docs,
                'question':     RunnablePassthrough(),
                'chat_history': lambda _: self._memory.chat_memory.messages,
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        """
        Responde una pregunta usando el pipeline RAG.
        Retorna la respuesta y los documentos fuente.
        """
        # Recuperar documentos relevantes
        source_docs = self._retriever.invoke(question)

        # Generar respuesta
        answer = self._chain.invoke(question)

        # Guardar en memoria
        self._memory.save_context(
            {'input': question},
            {'output': answer},
        )

        return {
            'answer':   answer,
            'sources':  [d.metadata.get('source_file', '') for d in source_docs],
            'num_docs': len(source_docs),
        }

    def reset_memory(self):
        """Limpia el historial de conversación."""
        self._memory.clear()
