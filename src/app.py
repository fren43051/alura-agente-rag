"""Tarea 6 — Interfaz de usuario con Gradio."""

import os
import gradio as gr
from loguru import logger
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from agent import build_agent

load_dotenv()


# Inicializar agente
logger.info("Iniciando AluraAgente RAG...")
agent = build_agent()
logger.success("Agente listo")


def chat(message: str, history: list):
    """Procesa un mensaje del usuario y retorna la respuesta del agente."""
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=message)],
            "documents_searched": False,
        })
        response = result["messages"][-1].content
        return response
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        return f"❌ Error al procesar tu consulta: {str(e)}"


# --- Interfaz Gradio ---
with gr.Blocks(
    title="AluraAgente RAG",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
    ),
    css="""
    .contain { max-width: 900px; margin: auto; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown("""
    # 🤖 AluraAgente RAG
    ### Asistente Corporativo de Documentos — Challenge ONE IA FOR TECH
    
    Haz preguntas sobre los documentos internos de la empresa.
    El agente buscará la información relevante y te responderá citando las fuentes.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                examples=[
                    "¿Qué documentos están disponibles?",
                    "¿Cuáles son las políticas de vacaciones?",
                    "Resume el contenido del reporte más reciente",
                    "¿Cuáles son los procedimientos de onboarding?",
                ],
                title="",
                description="",
            )

if __name__ == "__main__":
    demo.launch(
        server_port=int(os.getenv("APP_PORT", 7860)),
        share=False,
        show_error=True,
    )
