# main.py — Punto de entrada principal
# Uso: python main.py

import os
import sys
from pathlib import Path

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def check_environment():
    """Verifica que las variables de entorno estén configuradas."""
    required = ["ANTHROPIC_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        print("❌ Variables de entorno faltantes:")
        for key in missing:
            print(f"   - {key}")
        print("\n💡 Solución: Copia .env.example a .env y completa los valores.")
        print("   cp .env.example .env")
        sys.exit(1)
    print("✅ Variables de entorno OK")

def main():
    print("="*60)
    print("🤖 AluraAgente RAG — Versión Avanzada Profesional")
    print("   LangGraph + Claude 3.5 Haiku + Gradio UI")
    print("   Challenge Alura Latam — ONE IA FOR TECH")
    print("="*60)
    # Verificar entorno
    check_environment()
    # Crear directorios necesarios
    for directory in ["documents", "vectorstore", "charts"]:
        os.makedirs(directory, exist_ok=True)
    print(f"📁 Directorios listos: documents/, vectorstore/, charts/")
    # Lanzar UI
    print("\n🚀 Iniciando servidor Gradio...")
    print(f"   URL local: http://localhost:{os.getenv('PORT', '7860')}")
    print("   Presiona Ctrl+C para detener\n")
    from src.app import demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True
    )

if __name__ == "__main__":
    main()
