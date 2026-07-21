#!/usr/bin/env python3
"""
AluraAgente RAG — Versión Avanzada v2.0
Entrypoint principal del proyecto

Uso:
    python main.py           # Inicia la UI Gradio
    python main.py --cli     # Modo línea de comandos
    python main.py --index   # Solo indexa documentos en ./documents/
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def check_env():
    """Verifica que las variables de entorno requeridas estén configuradas."""
    required = ["ANTHROPIC_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print("❌ Variables de entorno faltantes:")
        for k in missing:
            print(f"   - {k}")
        print("\n📋 Copia .env.example a .env y completa los valores.")
        sys.exit(1)
    print("✅ Variables de entorno configuradas correctamente.")


def run_ui():
    """Inicia la interfaz Gradio."""
    print("🚀 Iniciando AluraAgente RAG UI...")
    print("   Accede en: http://localhost:7860")
    from src.app import create_ui
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    )


def run_cli():
    """Modo interactivo de línea de comandos."""
    import uuid
    from src.agent import run_agent
    print("🤖 AluraAgente RAG — Modo CLI")
    print("   Escribe 'salir' para terminar\n")
    session_id = str(uuid.uuid4())
    while True:
        try:
            user_input = input("Tú: ").strip()
            if user_input.lower() in ("salir", "exit", "quit"):
                print("👋 ¡Hasta luego!")
                break
            if not user_input:
                continue
            print("🔄 Procesando...")
            response, docs = run_agent(user_input, session_id)
            print(f"\n🤖 Agente: {response}")
            if docs:
                print(f"📎 Fuentes: {', '.join(set(docs))}")
            print()
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break


def run_index():
    """Indexa todos los documentos en el directorio ./documents/."""
    from pathlib import Path
    from src.tools import index_documents
    docs_dir = Path("./documents")
    if not docs_dir.exists():
        print("📁 Creando directorio ./documents/")
        docs_dir.mkdir()
    extensions = (".pdf", ".xlsx", ".xls", ".docx", ".csv", ".md", ".txt", ".json", ".html")
    files = [str(f) for f in docs_dir.iterdir() if f.suffix.lower() in extensions]
    if not files:
        print("⚠️  No hay documentos en ./documents/ para indexar.")
        print("   Agrega archivos PDF, Excel, Word, CSV, etc. y vuelve a ejecutar.")
        return
    print(f"📄 Indexando {len(files)} documento(s)...")
    result = index_documents(files)
    print(result)


if __name__ == "__main__":
    check_env()
    if "--cli" in sys.argv:
        run_cli()
    elif "--index" in sys.argv:
        run_index()
    else:
        run_ui()
