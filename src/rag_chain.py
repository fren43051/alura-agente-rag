"""Tarea 4 — Cadena RAG con LangChain."""

import os
from loguru import logger
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from vectorstore import load_vectorstore

load_dotenv()


RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente corporativo experto. Responde la pregunta del usuario 
exclusivamente basándote en el contexto proporcionado de los documentos internos.

Si la información no está en el contexto, responde: 
"No encontré información sobre ese tema en los documentos disponibles."

Cita siempre el nombre del archivo fuente al final de tu respuesta.

## Contexto de los documentos:
{context}

## Pregunta del colaborador:
{question}

## Respuesta:
""")


def get_llm():
    """Retorna el LLM configurado según la API key disponible."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    elif os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            temperature=0,
        )
    else:
        raise ValueError("Se requiere OPENAI_API_KEY o GOOGLE_API_KEY en .env")


def format_docs(docs):
    """Formatea los documentos recuperados con metadatos."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "desconocido")
        formatted.append(f"[Doc {i} - {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """Construye la cadena RAG completa."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={"k": 5, "fetch_k": 20},
    )
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.success("Cadena RAG construida correctamente")
    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    test_q = "¿Cuáles son las políticas de vacaciones de la empresa?"
    print(f"\nPregunta: {test_q}")
    print("Respuesta:", chain.invoke(test_q))
