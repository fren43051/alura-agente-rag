"""
app.py — Interfaz Gradio con sidebar de documentos, métricas en vivo y chat con historial.
Arquitectura: layout de 2 columnas (panel izq: docs + métricas | panel der: chat).
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import gradio as gr

from .agent import AgentResponse, clear_session, run_agent
from .tools import index_documents, list_indexed_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Estado de sesión (en memoria — compatible con sandboxes Gradio)
# ---------------------------------------------------------------------------
_SESSION: dict = {
    "id": str(uuid.uuid4()),
    "start_time": time.time(),
    "tokens_approx": 0,
    "tools_called": 0,
    "messages_count": 0,
}


def _new_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Funciones de UI
# ---------------------------------------------------------------------------

def chat_fn(message: str, history: list, session_state: dict) -> tuple:
    """
    Procesa el mensaje del usuario, invoca el agente y actualiza el historial.
    Devuelve: ("", historia_actualizada, session_state_actualizado, métricas, docs)
    """
    if not message.strip():
        return "", history, session_state, _get_metrics(session_state), _get_doc_list()

    session_id = session_state.get("id", _new_session_id())

    # Invocar el agente
    response: AgentResponse = run_agent(message, session_id=session_id)

    # Actualizar métricas de sesión
    session_state["tokens_approx"] = session_state.get("tokens_approx", 0) + \
        len(message.split()) + len(response.content.split())
    session_state["tools_called"] = session_state.get("tools_called", 0) + \
        response.session_metadata.get("tools_called", 0)
    session_state["messages_count"] = session_state.get("messages_count", 0) + 1

    # Construir respuesta con referencias si las hay
    display_response = response.content
    if response.referenced_docs:
        refs = ", ".join(f"`{d}`" for d in response.referenced_docs)
        display_response += f"\n\n📎 **Documentos consultados:** {refs}"

    # Agregar imágenes de gráficos si se generaron
    chart_html = ""
    for chart_path in response.chart_paths:
        if Path(chart_path).exists():
            chart_html += f'\n\n<img src="file={chart_path}" style="max-width:100%;border-radius:8px;">'

    if chart_html:
        display_response += chart_html

    history.append((message, display_response))

    return (
        "",
        history,
        session_state,
        _get_metrics(session_state),
        _get_doc_list(),
    )


def upload_fn(files, session_state: dict) -> tuple:
    """Copia los archivos a data/ y los indexa en el vectorstore."""
    if not files:
        return _get_doc_list(), session_state, "No se seleccionaron archivos."

    file_paths = []
    for f in files:
        src = Path(f.name)
        dest = DATA_DIR / src.name
        dest.write_bytes(src.read_bytes())
        file_paths.append(str(dest))
        logger.info("Archivo copiado: %s", dest)

    count = index_documents(file_paths)
    status = f"✅ {len(file_paths)} archivo(s) indexados — {count} fragmentos procesados."
    logger.info(status)

    return _get_doc_list(), session_state, status


def new_chat_fn(session_state: dict) -> tuple:
    """Resetea el historial y crea una nueva sesión."""
    old_id = session_state.get("id")
    if old_id:
        clear_session(old_id)

    new_state = {
        "id": _new_session_id(),
        "start_time": time.time(),
        "tokens_approx": 0,
        "tools_called": 0,
        "messages_count": 0,
    }
    return [], "", new_state, _get_metrics(new_state)


def _get_doc_list() -> str:
    """Retorna los documentos indexados formateados para el sidebar."""
    docs = list_indexed_docs()
    if not docs:
        return "_No hay documentos indexados.\nSube archivos abajo._"
    lines = [f"☑ {d}" for d in docs]
    return "\n".join(lines)


def _get_metrics(session_state: dict) -> str:
    """Retorna métricas de sesión formateadas."""
    elapsed = int(time.time() - session_state.get("start_time", time.time()))
    mins, secs = divmod(elapsed, 60)
    tokens = session_state.get("tokens_approx", 0)
    tools = session_state.get("tools_called", 0)
    messages = session_state.get("messages_count", 0)
    doc_count = len(list_indexed_docs())
    sid = session_state.get("id", "---")[:8]

    return (
        f"🔢 Tokens (aprox): {tokens:,}\n"
        f"📄 Docs indexados: {doc_count}\n"
        f"💬 Mensajes: {messages}\n"
        f"🔧 Herramientas: {tools}\n"
        f"⏱  Sesión: {mins}m {secs:02d}s\n"
        f"🆔 ID: {sid}..."
    )


# ---------------------------------------------------------------------------
# Construcción de la UI con Gradio Blocks
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        ),
        title="AluraAgente RAG — Asistente Corporativo",
        css="""
            .sidebar { background: #0f172a; border-right: 1px solid #1e293b; }
            .chat-panel { background: #0a0f1e; }
            footer { display: none !important; }
            .metric-box { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; }
            .upload-btn { border: 2px dashed #334155 !important; }
            .gr-button-primary { background: #0d9488 !important; }
        """,
    ) as demo:

        # ── Estado de sesión (en memoria, sin localStorage) ──────────────────
        session_state = gr.State({
            "id": _new_session_id(),
            "start_time": time.time(),
            "tokens_approx": 0,
            "tools_called": 0,
            "messages_count": 0,
        })

        # ── Encabezado ────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown(
                """# 🤖 AluraAgente RAG
**Asistente corporativo de documentos** — Claude 3.5 Haiku + LangGraph"""
            )
            exit_btn = gr.Button("🚪 Salir", variant="stop", scale=0, min_width=90)

        gr.Divider()

        # ── Layout principal: sidebar + chat ──────────────────────────────────
        with gr.Row(equal_height=True):

            # ─── Sidebar izquierdo ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=240, elem_classes=["sidebar"]):

                gr.Markdown("### 📁 Documentos Indexados")
                doc_display = gr.Textbox(
                    value=_get_doc_list,
                    label="",
                    lines=9,
                    interactive=False,
                    show_copy_button=False,
                    elem_classes=["metric-box"],
                )

                gr.Markdown("### 📤 Subir Documentos")
                upload_input = gr.File(
                    file_count="multiple",
                    file_types=[
                        ".pdf", ".xlsx", ".xls", ".docx", ".doc",
                        ".csv", ".txt", ".md", ".json", ".html", ".htm",
                    ],
                    label="",
                    elem_classes=["upload-btn"],
                )
                upload_status = gr.Textbox(
                    label="Estado", lines=1, interactive=False
                )

                gr.Markdown("### 📊 Métricas de Sesión")
                metrics_display = gr.Textbox(
                    value=lambda: _get_metrics(_SESSION),
                    label="",
                    lines=7,
                    interactive=False,
                    elem_classes=["metric-box"],
                )

                new_chat_btn = gr.Button("🗑  Nuevo Chat", variant="secondary")

            # ─── Panel de chat ───────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["chat-panel"]):
                chatbot = gr.Chatbot(
                    value=[],
                    height=520,
                    label="Chat con el Agente",
                    show_copy_button=True,
                    bubble_full_width=False,
                    render_markdown=True,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/8.x/bottts-neutral/svg?seed=alura&backgroundColor=0d9488",
                    ),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Pregunta sobre tus documentos corporativos... (Enter para enviar)",
                        show_label=False,
                        scale=5,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Enviar →", variant="primary", scale=1, min_width=100)

                gr.Markdown(
                    "_💡 Puedes preguntar: '¿Cuáles son las ventas del Q3?', "
                    "'Resume el manual de HR', 'Muestra un gráfico de ventas por mes'_",
                    elem_classes=["metric-box"],
                )

        # ── Eventos ───────────────────────────────────────────────────────────

        # Enviar mensaje
        send_inputs = [msg_input, chatbot, session_state]
        send_outputs = [msg_input, chatbot, session_state, metrics_display, doc_display]

        send_btn.click(fn=chat_fn, inputs=send_inputs, outputs=send_outputs)
        msg_input.submit(fn=chat_fn, inputs=send_inputs, outputs=send_outputs)

        # Subir documentos
        upload_input.upload(
            fn=upload_fn,
            inputs=[upload_input, session_state],
            outputs=[doc_display, session_state, upload_status],
        )

        # Nuevo chat
        new_chat_btn.click(
            fn=new_chat_fn,
            inputs=[session_state],
            outputs=[chatbot, msg_input, session_state, metrics_display],
        )

        # Salir
        exit_btn.click(fn=None, js="() => window.close()")

        # Auto-refresh de métricas cada 15 s
        demo.load(fn=_get_doc_list, outputs=doc_display, every=15)

    return demo


def launch():
    """Punto de entrada para lanzar la aplicación."""
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    launch()
