# src/app.py — UI Gradio Profesional con Sidebar + Métricas en Vivo

import os
import time
import uuid
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from src.agent import run_agent, reset_session
from src.tools import index_documents, list_indexed_docs, DOCS_DIR

load_dotenv()

# ─── Estado de sesión (en memoria — sin localStorage) ─────────
session_state = {
    "id": str(uuid.uuid4()),
    "start_time": time.time(),
    "tokens_used": 0,
    "tools_called": 0,
    "queries_count": 0,
    "referenced_docs": set(),
}

# ─── Helpers ─────────────────────────────────────────────────
def get_doc_list_text() -> str:
    docs = list_indexed_docs()
    if not docs:
        return "_No hay documentos indexados aún._\n\nSube archivos usando el panel de abajo."
    return "\n".join([f"☑ `{d}`" for d in docs])

def get_metrics_text() -> str:
    elapsed = int(time.time() - session_state["start_time"])
    mins, secs = divmod(elapsed, 60)
    docs_count = len(list_indexed_docs())
    return (
        f"🔢 **Tokens estimados:** {session_state['tokens_used']:,}\n"
        f"📄 **Docs indexados:** {docs_count}\n"
        f"💬 **Consultas:** {session_state['queries_count']}\n"
        f"🔧 **Herramientas usadas:** {session_state['tools_called']}\n"
        f"⏱ **Sesión:** {mins}m {secs}s\n"
        f"🆔 **Session ID:** `{session_state['id'][:8]}...`"
    )

def get_cited_docs_text() -> str:
    docs = session_state["referenced_docs"]
    if not docs:
        return "_Las referencias aparecerán aquí durante la conversación._"
    return "\n".join([f"📎 `{d}`" for d in sorted(docs)])

# ─── Función principal del chat ───────────────────────────────
def chat_fn(message: str, history: list):
    if not message.strip():
        return "", history
    # Ejecutar agente
    response, cited_docs, tools_used = run_agent(message, session_state["id"])
    # Actualizar métricas
    session_state["tokens_used"] += len(message.split()) * 2 + len(str(response).split())
    session_state["tools_called"] += len(tools_used)
    session_state["queries_count"] += 1
    session_state["referenced_docs"].update(cited_docs)
    # Agregar referencia visual si hay docs citados
    if cited_docs:
        refs_text = "\n\n---\n📎 **Documentos consultados:** " + " | ".join([f"`{d}`" for d in cited_docs])
        response = str(response) + refs_text
    history.append((message, response))
    return "", history

# ─── Upload de documentos ─────────────────────────────────────
def upload_and_index(files):
    if not files:
        return get_doc_list_text(), "⚠️ No se seleccionaron archivos."
    paths = []
    for f in files:
        dest = os.path.join(DOCS_DIR, Path(f.name).name)
        import shutil
        shutil.copy(f.name, dest)
        paths.append(dest)
    result = index_documents(paths)
    return get_doc_list_text(), f"✅ {result}"

# ─── Nuevo chat ───────────────────────────────────────────────
def new_chat():
    new_id = reset_session(session_state["id"])
    session_state["id"] = new_id
    session_state["start_time"] = time.time()
    session_state["tokens_used"] = 0
    session_state["tools_called"] = 0
    session_state["queries_count"] = 0
    session_state["referenced_docs"] = set()
    return [], ""

# ─── Construcción de la UI ────────────────────────────────────
css = """
.sidebar { background: #1e1e2e !important; border-radius: 12px; padding: 16px; }
.chat-area { border-radius: 12px; }
.metric-box { background: #2a2a3e; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 13px; }
.doc-list { font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
.upload-status { color: #4ade80; font-size: 12px; }
#send-btn { background: #6366f1 !important; color: white !important; }
#new-chat-btn { background: #374151 !important; color: #d1d5db !important; }
#exit-btn { background: #dc2626 !important; color: white !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    title="AluraAgente RAG",
    css=css
) as demo:

    # ── Encabezado ──────────────────────────────────────────────
    with gr.Row():
        gr.Markdown(
            """# 🤖 AluraAgente RAG
            **Asistente Corporativo de Documentos** · LangGraph + Claude 3.5 Haiku · Challenge Alura Latam"""
        )
        exit_btn = gr.Button("🚪 Salir", elem_id="exit-btn", scale=0, min_width=80)

    gr.Divider()

    with gr.Row(equal_height=False):

        # ══ SIDEBAR IZQUIERDO ════════════════════════════════════
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):

            gr.Markdown("### 📁 Documentos Indexados")
            doc_list_display = gr.Markdown(
                value=get_doc_list_text,
                every=8,
                elem_classes="doc-list"
            )

            gr.Markdown("### 📤 Subir Documentos")
            gr.Markdown(
                "_Formatos: PDF · DOCX · XLSX · CSV · MD · JSON · HTML_",
                elem_classes="upload-status"
            )
            file_upload = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".xlsx", ".xls",
                             ".csv", ".md", ".json", ".html"],
                label="",
                height=100
            )
            upload_status = gr.Markdown("", elem_classes="upload-status")

            gr.Divider()

            gr.Markdown("### 📊 Métricas de Sesión")
            metrics_display = gr.Markdown(
                value=get_metrics_text,
                every=5,
                elem_classes="metric-box"
            )

            gr.Divider()

            gr.Markdown("### 📎 Referencias Citadas")
            cited_display = gr.Markdown(
                value=get_cited_docs_text,
                every=5
            )

            gr.Divider()

            new_chat_btn = gr.Button(
                "🗑 Nuevo Chat",
                elem_id="new-chat-btn",
                variant="secondary"
            )

        # ══ PANEL PRINCIPAL DE CHAT ══════════════════════════════
        with gr.Column(scale=3, elem_classes="chat-area"):

            chatbot = gr.Chatbot(
                height=540,
                label="",
                bubble_full_width=False,
                show_label=False,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=alura"
                ),
                placeholder=(
                    "### 👋 ¡Hola! Soy AluraAgente\n"
                    "Tengo acceso a tus documentos corporativos indexados.\n\n"
                    "Puedo ayudarte con:\n"
                    "- 🔍 Buscar información en documentos\n"
                    "- 📊 Analizar datos y generar gráficos automáticamente\n"
                    "- 📋 Resumir documentos largos\n"
                    "- 🔗 Cruzar información entre múltiples archivos\n\n"
                    "**¿Qué necesitas saber hoy?**"
                )
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Escribe tu pregunta sobre los documentos corporativos...",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=4,
                    autofocus=True
                )
                send_btn = gr.Button(
                    "Enviar →",
                    elem_id="send-btn",
                    variant="primary",
                    scale=1
                )

            gr.Markdown(
                "_💡 **Tips:** Pregunta sobre cualquier documento indexado · "
                "Pide gráficos automáticos · Cruza información entre archivos_",
            )

    # ── Eventos ──────────────────────────────────────────────────
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
    file_upload.upload(
        fn=upload_and_index,
        inputs=[file_upload],
        outputs=[doc_list_display, upload_status]
    )
    new_chat_btn.click(
        fn=new_chat,
        outputs=[chatbot, msg_input]
    )
    exit_btn.click(
        fn=None,
        js="() => { window.close(); }"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
        favicon_path=None
    )
