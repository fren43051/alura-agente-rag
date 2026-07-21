# src/tools.py — 7 Herramientas RAG (4 originales + 3 avanzadas)
import os
import json
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from typing import Optional
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Configuración global ──────────────────────────────────────
VECTORSTORE_DIR = "./vectorstore"
DOCUMENTS_DIR = "./documents"
INDEXED_DOCS_FILE = "./vectorstore/indexed_docs.json"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Path(VECTORSTORE_DIR).mkdir(exist_ok=True)
Path(DOCUMENTS_DIR).mkdir(exist_ok=True)


def get_vectorstore():
    """Retorna o crea el vectorstore ChromaDB persistente."""
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )


def list_indexed_docs() -> list:
    """Lista los documentos ya indexados."""
    if not os.path.exists(INDEXED_DOCS_FILE):
        return []
    with open(INDEXED_DOCS_FILE, "r") as f:
        return json.load(f)


def _save_indexed_doc(name: str):
    docs = list_indexed_docs()
    if name not in docs:
        docs.append(name)
    with open(INDEXED_DOCS_FILE, "w") as f:
        json.dump(docs, f)


def index_documents(file_paths: list) -> str:
    """Indexa una lista de archivos en ChromaDB."""
    vs = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    results = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            elif ext == ".csv":
                loader = CSVLoader(path)
            elif ext in (".xlsx", ".xls"):
                loader = UnstructuredExcelLoader(path)
            elif ext == ".html":
                loader = UnstructuredHTMLLoader(path)
            elif ext in (".md", ".txt", ".json"):
                loader = TextLoader(path)
            else:
                results.append(f"⚠️ Formato no soportado: {ext}")
                continue
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            vs.add_documents(chunks)
            _save_indexed_doc(Path(path).name)
            results.append(f"✅ Indexado: {Path(path).name} ({len(chunks)} chunks)")
        except Exception as e:
            results.append(f"❌ Error en {Path(path).name}: {str(e)}")
    return "\n".join(results)


# ══════════════════════════════════════════════════════════════════
# HERRAMIENTAS ORIGINALES (del proyecto base)
# ══════════════════════════════════════════════════════════════════

@tool
def search_documents(query: str) -> str:
    """Busca información en los documentos corporativos indexados usando RAG.
    Retorna los fragmentos más relevantes con referencia al documento fuente."""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=4)
        if not results:
            return "No se encontraron documentos relevantes para esa consulta."
        output = []
        for doc, score in results:
            source = doc.metadata.get("source", "Desconocido")
            source_name = Path(source).name if source != "Desconocido" else source
            relevance = round((1 - score) * 100, 1)
            output.append(
                f"📄 **{source_name}** (relevancia: {relevance}%)\n"
                f"{doc.page_content[:500]}..."
            )
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Error buscando documentos: {str(e)}"


@tool
def analyze_data(query: str) -> str:
    """Analiza datos tabulares (CSV/Excel) en los documentos indexados.
    Retorna estadísticas descriptivas, tendencias y observaciones clave."""
    try:
        data_files = [
            f for f in Path(DOCUMENTS_DIR).iterdir()
            if f.suffix.lower() in (".csv", ".xlsx", ".xls")
        ]
        if not data_files:
            return "No hay archivos de datos (CSV/Excel) disponibles para analizar."
        results = []
        for file in data_files[:3]:  # máximo 3 archivos
            try:
                if file.suffix.lower() == ".csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                summary = [
                    f"📊 **{file.name}**",
                    f"  Filas: {len(df):,} | Columnas: {len(df.columns)}",
                    f"  Columnas: {', '.join(df.columns.tolist()[:8])}",
                    f"  Datos nulos: {df.isnull().sum().sum():,}",
                    "  **Estadísticas numéricas:**",
                    df.describe().round(2).to_string()
                ]
                results.append("\n".join(summary))
            except Exception as e:
                results.append(f"  Error leyendo {file.name}: {e}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error en análisis: {str(e)}"


@tool
def generate_chart(chart_request: str) -> str:
    """Genera gráficos a partir de datos disponibles usando Matplotlib/Seaborn.
    Especifica qué archivo y qué columnas graficar. Retorna la ruta del gráfico."""
    try:
        data_files = list(Path(DOCUMENTS_DIR).glob("*.csv")) + \
                     list(Path(DOCUMENTS_DIR).glob("*.xlsx"))
        if not data_files:
            return "No hay datos disponibles para graficar."
        file = data_files[0]
        df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return "No hay columnas numéricas para graficar."
        fig, ax = plt.subplots(figsize=(10, 6))
        df[numeric_cols[:3]].plot(ax=ax)
        ax.set_title(f"Análisis de {file.name}")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        plt.tight_layout()
        output_path = f"./documents/chart_{int(time.time())}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return f"✅ Gráfico generado: {output_path}"
    except Exception as e:
        return f"Error generando gráfico: {str(e)}"


@tool
def summarize_text(document_name: str) -> str:
    """Genera un resumen ejecutivo de un documento específico indexado.
    Proporciona el nombre del archivo (ej: reporte.pdf)."""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(
            f"resumen contenido principal {document_name}",
            k=6,
            filter={"source": {"$contains": document_name}} if document_name else None
        )
        if not results:
            results = vs.similarity_search(document_name, k=6)
        if not results:
            return f"No se encontró el documento: {document_name}"
        combined = " ".join([r.page_content for r in results[:4]])
        preview = combined[:2000]
        return (
            f"📋 **Resumen de {document_name}:**\n\n"
            f"{preview}\n\n"
            f"*(Basado en {len(results)} fragmentos del documento)*"
        )
    except Exception as e:
        return f"Error resumiendo documento: {str(e)}"


# ══════════════════════════════════════════════════════════════════
# HERRAMIENTAS AVANZADAS (versión profesional)
# ══════════════════════════════════════════════════════════════════

@tool
def smart_chart_detector(data_description: str) -> str:
    """Detecta automáticamente el mejor tipo de gráfico según los datos disponibles
    y lo genera con Plotly (interactivo). Analiza estructura de datos y elige entre
    bar, line, pie, scatter, heatmap, histogram automáticamente."""
    try:
        data_files = list(Path(DOCUMENTS_DIR).glob("*.csv")) + \
                     list(Path(DOCUMENTS_DIR).glob("*.xlsx"))
        if not data_files:
            return "No hay datos disponibles para análisis automático de chart."
        file = data_files[0]
        df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        n_rows = len(df)
        # Lógica de detección automática
        if len(numeric_cols) >= 2 and n_rows > 50:
            chart_type = "scatter"
            fig = px.scatter(
                df, x=numeric_cols[0], y=numeric_cols[1],
                title=f"Dispersión: {numeric_cols[0]} vs {numeric_cols[1]}",
                template="plotly_white"
            )
        elif categorical_cols and numeric_cols:
            if df[categorical_cols[0]].nunique() <= 6:
                chart_type = "pie"
                fig = px.pie(
                    df, names=categorical_cols[0], values=numeric_cols[0],
                    title=f"Distribución por {categorical_cols[0]}",
                    template="plotly_white"
                )
            else:
                chart_type = "bar"
                fig = px.bar(
                    df.head(20), x=categorical_cols[0], y=numeric_cols[0],
                    title=f"Top 20: {numeric_cols[0]} por {categorical_cols[0]}",
                    template="plotly_white"
                )
        elif len(numeric_cols) >= 1 and n_rows <= 50:
            chart_type = "line"
            fig = px.line(
                df, y=numeric_cols[:3],
                title=f"Tendencia: {', '.join(numeric_cols[:3])}",
                template="plotly_white"
            )
        else:
            chart_type = "histogram"
            fig = px.histogram(
                df, x=numeric_cols[0],
                title=f"Distribución de {numeric_cols[0]}",
                template="plotly_white"
            )
        output_path = f"./documents/smart_chart_{int(time.time())}.html"
        fig.write_html(output_path)
        return (
            f"🤖 **Chart detectado automáticamente: {chart_type.upper()}**\n"
            f"  Archivo analizado: {file.name}\n"
            f"  Filas: {n_rows:,} | Cols numéricas: {len(numeric_cols)} | Cats: {len(categorical_cols)}\n"
            f"  Razón: {_explain_chart_choice(chart_type, numeric_cols, categorical_cols, n_rows)}\n"
            f"  Gráfico interactivo guardado: {output_path}"
        )
    except Exception as e:
        return f"Error en detección automática de chart: {str(e)}"


def _explain_chart_choice(chart_type, num_cols, cat_cols, n_rows):
    explanations = {
        "scatter": f"2+ columnas numéricas con {n_rows} filas → ideal para ver correlaciones",
        "pie": f"Variable categórica con ≤6 categorías → ideal para proporciones",
        "bar": f"Categorías múltiples vs valor numérico → ideal para comparar grupos",
        "line": f"Pocos registros ({n_rows}) con tendencia temporal → ideal para series",
        "histogram": f"Una variable numérica → ideal para ver distribución"
    }
    return explanations.get(chart_type, "Elección basada en estructura de datos")


@tool
def cross_document_analyst(query: str) -> str:
    """Cruza información entre MÚLTIPLES documentos simultáneamente.
    Ideal para preguntas comparativas que involucran datos de más de un archivo.
    Retorna hallazgos consolidados con referencias cruzadas."""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=8)
        if not results:
            return "No se encontraron documentos relevantes para análisis cruzado."
        # Agrupar por documento fuente
        docs_by_source = {}
        for doc, score in results:
            source = Path(doc.metadata.get("source", "Desconocido")).name
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append((doc.page_content, round((1 - score) * 100, 1)))
        if len(docs_by_source) == 1:
            return (
                f"ℹ️ Solo se encontró información en 1 documento para esta consulta.\n"
                f"Usa 'search_documents' para una búsqueda más detallada.\n\n"
                f"Fuente única: **{list(docs_by_source.keys())[0]}**"
            )
        output = [f"🔀 **Análisis Cruzado — {len(docs_by_source)} documentos:**\n"]
        for source, fragments in docs_by_source.items():
            output.append(f"\n### 📄 {source}")
            for content, relevance in fragments[:2]:
                output.append(f"  *(relevancia: {relevance}%)*")
                output.append(f"  {content[:400]}...")
        output.append(f"\n---\n**Documentos consultados:** {', '.join(docs_by_source.keys())}")
        return "\n".join(output)
    except Exception as e:
        return f"Error en análisis cruzado: {str(e)}"


@tool
def error_retry_handler(failed_query: str, error_context: str = "") -> str:
    """Maneja errores del agente con retry automático (máximo 3 intentos).
    Reformula la consulta fallida y reintenta con estrategia diferente.
    Úsala cuando otra herramienta haya fallado o retornado vacío."""
    strategies = [
        f"Información sobre: {failed_query}",
        f"Buscar en documentos: {' '.join(failed_query.split()[:5])}",
        f"Datos relacionados con {failed_query.split()[0] if failed_query else 'empresa'}"
    ]
    vs = get_vectorstore()
    for attempt, reformulated in enumerate(strategies, 1):
        try:
            results = vs.similarity_search(reformulated, k=3)
            if results:
                sources = list(set([
                    Path(r.metadata.get("source", "?")).name for r in results
                ]))
                content_preview = results[0].page_content[:300]
                return (
                    f"🔄 **Retry exitoso (intento {attempt}/3)**\n"
                    f"  Consulta reformulada: '{reformulated}'\n"
                    f"  Documentos encontrados: {', '.join(sources)}\n\n"
                    f"  Resultado: {content_preview}..."
                )
        except Exception:
            continue
    return (
        f"❌ **Todos los reintentos fallaron (3/3)**\n"
        f"  Consulta original: '{failed_query}'\n"
        f"  Contexto del error: {error_context}\n\n"
        f"  **Sugerencias:**\n"
        f"  1. Verifica que existan documentos indexados\n"
        f"  2. Usa términos más generales en tu pregunta\n"
        f"  3. Sube nuevos documentos relacionados con el tema"
    )


def get_all_tools():
    """Retorna la lista completa de las 7 herramientas RAG."""
    return [
        search_documents,
        analyze_data,
        generate_chart,
        summarize_text,
        smart_chart_detector,
        cross_document_analyst,
        error_retry_handler
    ]
