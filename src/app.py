# src/app.py — Gradio UI profesional con sidebar de métricas y panel de documentos
import os
import sys
import time
import uuid
import shutil
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

# Asegurar imports desde raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import run_agent, get_session_history
from src.tools import index_documents, list_indexed_docs, DOCUMENTS_DIR

load_dotenv()

# ─── Estado de sesión (en memoria — sin localStorage) ─────────────
class SessionState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.tokens_used = 0
        self.tools_called = 0
        self.messages_count = 0

session = SessionState()


# ─── Funciones del chat ───────────────────────────────────────────
def chat_fn(message: str, history: list):
    """Procesa mensaje del usuario y retorna respuesta del agente."""
    if not message.strip():
        return "", history
    session.messages_count += 1
    session.tokens_used += len(message.split())
    session.tools_called += 1
    response, docs_cited = run_agent(message, session.session_id)
    session.tokens_used += len(response.split())
    # Agregar referencias al final si hay documentos citados
    if docs_cited:
        unique_docs = list(set(docs_cited))
        refs = "\n\n---\n📎 **Documentos consultados:** " + " | ".join(
            [f"`{d}`" for d in unique_docs]
        )
        response += refs
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def upload_docs_fn(files):
    """Maneja la subida e indexación de nuevos documentos."""
    if not files:
        return get_doc_list_text(), "⚠️ No se seleccionaron archivos."
    paths = []
    for f in files:
        dest = Path(DOCUMENTS_DIR) / Path(f.name).name
        shutil.copy2(f.name, dest)
        paths.append(str(dest))
    result = index_documents(paths)
    return get_doc_list_text(), result


def get_doc_list_text() -> str:
    """Genera el texto de la lista de documentos indexados."""
    docs = list_indexed_docs()
    if not docs:
        return "*No hay documentos indexados aún.*\n\nSube archivos usando el panel de abajo."
    lines = [f"☑ {doc}" for doc in docs]
    return "\n".join(lines)


def get_metrics_text() -> str:
    """Genera el texto de métricas en vivo de la sesión."""
    elapsed = int(time.time() - session.start_time)
    mins, secs = divmod(elapsed, 60)
    docs_count = len(list_indexed_docs())
    return (
        f"🔢 **Tokens estimados:** {session.tokens_used:,}\n"
        f"📄 **Documentos:** {docs_count}\n"
        f"💬 **Mensajes:** {session.messages_count}\n"
        f"🔧 **Llamadas al agente:** {session.tools_called}\n"
        f"⏱ **Sesión:** {mins}m {secs}s\n"
        f"🆔 **ID sesión:** `{session.session_id[:8]}...`"
    )


def new_chat_fn():
    """Reinicia el chat manteniendo los documentos indexados."""
    session.session_id = str(uuid.uuid4())
    session.start_time = time.time()
    session.tokens_used = 0
    session.tools_called = 0
    session.messages_count = 0
    return [], get_metrics_text()


# ─── Interfaz Gradio ──────────────────────────────────────────────
def create_ui():
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        title="AluraAgente RAG",
        css="""
        .sidebar { border-right: 1px solid #e2e8f0; min-height: 90vh; }
        .chat-container { min-height: 520px; }
        .metric-box { background: #f8fafc; border-radius: 8px; padding: 12px; }
        footer { display: none !important; }
        """
    ) as demo:

        # ── Header ───────────────────────────────────────────────
        gr.Markdown(
            "# 🤖 AluraAgente RAG\n"
            "**Asistente Corporativo de Documentos** — "
            "Powered by Claude 3.5 Haiku + LangGraph"
        )

        with gr.Row():
            # ══ SIDEBAR IZQUIERDO ══════════════════════════════════
            with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):

                gr.Markdown("### 📁 Documentos Indexados")
                doc_list_display = gr.Markdown(
                    value=get_doc_list_text,
                    every=10,
                    label=""
                )

                gr.Markdown("---\n### 📤 Subir Archivos")
                upload_btn = gr.File(
                    file_count="multiple",
                    file_types=[
                        ".pdf", ".xlsx", ".xls", ".docx",
                        ".csv", ".md", ".txt", ".json", ".html"
                    ],
                    label="Arrastra archivos aquí o haz clic"
                )
                upload_status = gr.Textbox(
                    label="Estado de indexación",
                    lines=3, interactive=False
                )

                gr.Markdown("---\n### 📊 Métricas en Vivo")
                metrics_display = gr.Markdown(
                    value=get_metrics_text,
                    every=5,
                    elem_classes="metric-box"
                )

                gr.Markdown("---")
                new_chat_btn = gr.Button(
                    "🗑 Nuevo Chat", variant="secondary", size="sm"
                )
                gr.Markdown(
                    "*AluraAgente RAG v2.0 — Challenge Alura ONE*",
                )

            # ══ PANEL PRINCIPAL DE CHAT ════════════════════════════
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=520,
                    label="Chat con el Agente",
                    type="messages",
                    show_label=True,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=alura&backgroundColor=01696f"
                    ),
                    bubble_full_width=False,
                    render_markdown=True,
                    elem_classes="chat-container"
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Pregunta sobre tus documentos corporativos... (Ej: ¿Cuáles fueron las ventas del Q3?)",
                        scale=5,
                        show_label=False,
                        lines=2,
                        max_lines=4,
                        autofocus=True
                    )
                    send_btn = gr.Button(
                        "Enviar →", variant="primary", scale=1, size="lg"
                    )

                gr.Markdown(
                    "*💡 **Sugerencias:** ¿Cuáles son los KPIs principales? | "
                    "Resume el reporte financiero | "
                    "¿Qué dice la política de vacaciones? | "
                    "Genera un gráfico de ventas*"
                )

        # ── Eventos ───────────────────────────────────────────────
        send_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        upload_btn.upload(
            fn=upload_docs_fn,
            inputs=[upload_btn],
            outputs=[doc_list_display, upload_status]
        )
        new_chat_btn.click(
            fn=new_chat_fn,
            inputs=[],
            outputs=[chatbot, metrics_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True
    )
