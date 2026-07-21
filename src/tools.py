# src/tools.py — 7 Herramientas RAG Profesionales
# 4 originales + 3 nuevas avanzadas

import os
import json
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Optional
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredMarkdownLoader,
    JSONLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ─── Configuración vectorstore ────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = "./vectorstore"
DOCS_DIR = "./documents"
CHARTS_DIR = "./charts"

for d in [VECTORSTORE_DIR, DOCS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

# ─── Utilidades ───────────────────────────────────────────────
def load_document(file_path: str) -> list[Document]:
    """Carga un documento según su extensión."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
        ".json": lambda p: JSONLoader(p, jq_schema=".", text_content=False),
    }
    loader_class = loaders.get(ext)
    if not loader_class:
        raise ValueError(f"Formato no soportado: {ext}")
    loader = loader_class(file_path)
    return loader.load()

def index_documents(file_paths: list[str]) -> str:
    """Indexa una lista de archivos en el vectorstore."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_docs = []
    indexed = []
    for path in file_paths:
        try:
            docs = load_document(path)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source_file"] = Path(path).name
            all_docs.extend(chunks)
            indexed.append(Path(path).name)
        except Exception as e:
            print(f"Error indexando {path}: {e}")
    if all_docs:
        vectorstore.add_documents(all_docs)
        vectorstore.persist()
    return f"Indexados: {', '.join(indexed)}"

def list_indexed_docs() -> list[str]:
    """Retorna lista de documentos únicos indexados."""
    try:
        data = vectorstore.get()
        sources = set()
        for meta in data.get("metadatas", []):
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(list(sources))
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 1: Búsqueda RAG en documentos
# ══════════════════════════════════════════════════════════════
@tool
def search_documents(query: str, k: int = 5) -> str:
    """
    Busca información relevante en los documentos corporativos indexados.
    Usa RAG (Retrieval-Augmented Generation) para encontrar respuestas
    precisas con referencias a los documentos fuente.
    Args:
        query: La pregunta o término a buscar.
        k: Número de fragmentos a recuperar (default: 5).
    Returns:
        Texto relevante con referencias al documento fuente.
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return "No se encontró información relevante en los documentos indexados."
        output = []
        for doc, score in results:
            source = doc.metadata.get("source_file", "Desconocido")
            relevance = round((1 - score) * 100, 1)
            output.append(
                f"📄 **Fuente:** {source} (relevancia: {relevance}%)\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 2: Análisis estadístico de datos
# ══════════════════════════════════════════════════════════════
@tool
def analyze_data(file_name: str, query: str) -> str:
    """
    Analiza datos estadísticos de archivos CSV o Excel en la carpeta documents/.
    Proporciona estadísticas descriptivas, correlaciones y insights numéricos.
    Args:
        file_name: Nombre del archivo CSV o Excel en documents/.
        query: Qué analizar (ej: 'ventas por mes', 'top 5 productos').
    Returns:
        Análisis estadístico detallado del dataset.
    """
    try:
        file_path = os.path.join(DOCS_DIR, file_name)
        if not os.path.exists(file_path):
            return f"Archivo no encontrado: {file_name}. Verifica que esté en la carpeta documents/."
        ext = Path(file_name).suffix.lower()
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        analysis = [
            f"📊 **Análisis de:** {file_name}",
            f"- Filas: {len(df):,} | Columnas: {len(df.columns)}",
            f"- Columnas numéricas: {', '.join(numeric_cols) if numeric_cols else 'Ninguna'}",
            f"- Valores nulos: {df.isnull().sum().sum():,}",
            "\n**Estadísticas descriptivas:**",
            df[numeric_cols].describe().round(2).to_string() if numeric_cols else "Sin columnas numéricas",
        ]
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(3)
            analysis.append("\n**Correlaciones principales:**")
            analysis.append(corr.to_string())
        return "\n".join(analysis)
    except Exception as e:
        return f"Error analizando datos: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 3: Generación de gráficos
# ══════════════════════════════════════════════════════════════
@tool
def generate_chart(file_name: str, chart_type: str, x_col: str, y_col: str, title: str = "") -> str:
    """
    Genera gráficos a partir de archivos CSV o Excel.
    Tipos disponibles: bar, line, scatter, pie, hist, box.
    Args:
        file_name: Nombre del archivo en documents/.
        chart_type: Tipo de gráfico (bar, line, scatter, pie, hist, box).
        x_col: Columna para el eje X (o categorías en pie).
        y_col: Columna para el eje Y (o valores en pie).
        title: Título del gráfico (opcional).
    Returns:
        Ruta del archivo de imagen generado.
    """
    try:
        file_path = os.path.join(DOCS_DIR, file_name)
        ext = Path(file_name).suffix.lower()
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        plot_title = title or f"{chart_type.capitalize()} — {y_col} por {x_col}"
        chart_funcs = {
            "bar": lambda: sns.barplot(data=df, x=x_col, y=y_col, ax=ax),
            "line": lambda: sns.lineplot(data=df, x=x_col, y=y_col, ax=ax),
            "scatter": lambda: sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax),
            "pie": lambda: ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%'),
            "hist": lambda: sns.histplot(data=df, x=y_col, ax=ax, bins=20),
            "box": lambda: sns.boxplot(data=df, x=x_col, y=y_col, ax=ax),
        }
        if chart_type not in chart_funcs:
            return f"Tipo de gráfico no válido. Opciones: {', '.join(chart_funcs.keys())}"
        chart_funcs[chart_type]()
        ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        chart_path = os.path.join(CHARTS_DIR, f"chart_{chart_type}_{x_col}_{y_col}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path
    except Exception as e:
        return f"Error generando gráfico: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 4: Resumen de documentos
# ══════════════════════════════════════════════════════════════
@tool
def summarize_document(file_name: str, max_chunks: int = 5) -> str:
    """
    Genera un resumen estructurado de un documento indexado.
    Recupera los fragmentos más representativos del documento.
    Args:
        file_name: Nombre del archivo a resumir.
        max_chunks: Máximo de fragmentos a incluir (default: 5).
    Returns:
        Resumen estructurado con los puntos principales del documento.
    """
    try:
        data = vectorstore.get()
        doc_chunks = []
        for content, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            if meta and meta.get("source_file") == file_name:
                doc_chunks.append(content)
        if not doc_chunks:
            return f"Documento '{file_name}' no encontrado en el índice. ¿Ya fue indexado?"
        selected = doc_chunks[:max_chunks]
        summary = [
            f"📋 **Resumen de:** {file_name}",
            f"- Total de fragmentos indexados: {len(doc_chunks)}",
            f"- Mostrando: {len(selected)} fragmentos principales",
            "\n**Contenido principal:**"
        ]
        for i, chunk in enumerate(selected, 1):
            summary.append(f"\n**Sección {i}:**\n{chunk.strip()[:500]}...")
        return "\n".join(summary)
    except Exception as e:
        return f"Error resumiendo documento: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 5 (NUEVA): Detector automático del mejor chart
# ══════════════════════════════════════════════════════════════
@tool
def smart_chart_detector(file_name: str, objective: str) -> str:
    """
    Analiza los datos y detecta automáticamente el tipo de gráfico más
    adecuado según la estructura del dataset y el objetivo del análisis.
    Luego genera el gráfico óptimo automáticamente.
    Args:
        file_name: Nombre del archivo CSV o Excel en documents/.
        objective: Qué quieres visualizar (ej: 'tendencia de ventas mensuales').
    Returns:
        Tipo de gráfico seleccionado, justificación y ruta de la imagen.
    """
    try:
        file_path = os.path.join(DOCS_DIR, file_name)
        ext = Path(file_name).suffix.lower()
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
        n_unique_cat = df[categorical_cols[0]].nunique() if categorical_cols else 0
        # Lógica de detección
        if datetime_cols or any(c.lower() in ['fecha', 'date', 'mes', 'año', 'year', 'month'] for c in df.columns):
            chart_type, reason = "line", "Datos con dimensión temporal → gráfico de línea para mostrar tendencias"
        elif categorical_cols and numeric_cols and n_unique_cat <= 8:
            chart_type, reason = "bar", f"Variable categórica con {n_unique_cat} categorías → barras para comparación"
        elif categorical_cols and numeric_cols and n_unique_cat <= 5:
            chart_type, reason = "pie", "Pocas categorías con valores → pastel para mostrar proporciones"
        elif len(numeric_cols) >= 2:
            chart_type, reason = "scatter", "Dos variables numéricas → dispersión para detectar correlación"
        elif numeric_cols:
            chart_type, reason = "hist", "Variable numérica única → histograma para distribución"
        else:
            return "No se encontraron columnas numéricas para graficar."
        x_col = categorical_cols[0] if categorical_cols else numeric_cols[0]
        y_col = numeric_cols[0]
        chart_path = generate_chart.invoke({
            "file_name": file_name,
            "chart_type": chart_type,
            "x_col": x_col,
            "y_col": y_col,
            "title": f"[Auto-detectado] {objective}"
        })
        return (
            f"🔍 **Chart detector — Análisis automático**\n"
            f"- Dataset: {len(df):,} filas × {len(df.columns)} columnas\n"
            f"- Tipo seleccionado: **{chart_type.upper()}**\n"
            f"- Razón: {reason}\n"
            f"- Ejes: X={x_col}, Y={y_col}\n"
            f"- Gráfico guardado: {chart_path}"
        )
    except Exception as e:
        return f"Error en detección automática: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 6 (NUEVA): Análisis cruzado entre documentos
# ══════════════════════════════════════════════════════════════
@tool
def cross_document_analyst(query: str, doc_names: str) -> str:
    """
    Busca y cruza información de múltiples documentos simultáneamente.
    Ideal para comparar políticas, consolidar reportes o triangular datos.
    Args:
        query: Qué información buscar en todos los documentos.
        doc_names: Nombres de documentos separados por coma (ej: 'reporte.pdf,ventas.xlsx').
    Returns:
        Análisis comparativo con hallazgos de cada documento y síntesis.
    """
    try:
        docs = [d.strip() for d in doc_names.split(",")]
        results_by_doc = {}
        for doc_name in docs:
            doc_results = vectorstore.similarity_search(
                query, k=3,
                filter={"source_file": doc_name}
            )
            if doc_results:
                results_by_doc[doc_name] = " ".join([r.page_content[:300] for r in doc_results])
            else:
                results_by_doc[doc_name] = "❌ Sin resultados relevantes para esta consulta."
        output = [f"🔗 **Análisis cruzado — Query:** '{query}'\n"]
        for doc, content in results_by_doc.items():
            output.append(f"### 📄 {doc}\n{content}\n")
        found = [d for d, c in results_by_doc.items() if "❌" not in c]
        if len(found) > 1:
            output.append(
                f"\n✅ **Síntesis:** Información sobre '{query}' encontrada en "
                f"{len(found)}/{len(docs)} documentos: {', '.join(found)}."
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error en análisis cruzado: {str(e)}"

# ══════════════════════════════════════════════════════════════
# HERRAMIENTA 7 (NUEVA): Manejador de errores con retry
# ══════════════════════════════════════════════════════════════
@tool
def error_retry_handler(failed_query: str, error_message: str, attempt: int = 1) -> str:
    """
    Maneja errores del agente con retry automático y reformulación de prompts.
    Detecta el tipo de error y sugiere una estrategia de recuperación.
    Máximo 3 intentos antes de escalar al usuario.
    Args:
        failed_query: La consulta original que falló.
        error_message: El mensaje de error recibido.
        attempt: Número del intento actual (1-3).
    Returns:
        Consulta reformulada o instrucciones de escalación.
    """
    if attempt > 3:
        return (
            f"⚠️ **Máximo de reintentos alcanzado (3/3)**\n"
            f"- Query original: '{failed_query}'\n"
            f"- Último error: {error_message}\n"
            f"- **Acción requerida:** Por favor reformula tu pregunta o verifica "
            f"que el documento esté indexado correctamente."
        )
    error_strategies = {
        "not found": f"Busca información sobre: {failed_query.replace('exactamente', '').replace('específicamente', '')}",
        "timeout": f"Versión simplificada: {' '.join(failed_query.split()[:10])}",
        "no data": f"¿Existe algún dato relacionado con {failed_query.split()[0]}?",
        "column": f"Lista las columnas disponibles en el dataset y su tipo de dato",
        "file": f"Lista todos los documentos disponibles en el índice",
    }
    reformulated = failed_query
    detected_error = "genérico"
    for key, strategy in error_strategies.items():
        if key.lower() in error_message.lower():
            reformulated = strategy
            detected_error = key
            break
    return (
        f"🔄 **Retry automático — Intento {attempt}/3**\n"
        f"- Error detectado: `{detected_error}`\n"
        f"- Query original: '{failed_query}'\n"
        f"- Query reformulada: '{reformulated}'\n"
        f"- Estrategia: Simplificación y búsqueda alternativa\n"
        f"\n➡️ Ejecutando con nueva query: **{reformulated}**"
    )

# ─── Registro de todas las herramientas ──────────────────────
def get_all_tools():
    """Retorna la lista completa de las 7 herramientas del agente."""
    return [
        search_documents,
        analyze_data,
        generate_chart,
        summarize_document,
        smart_chart_detector,
        cross_document_analyst,
        error_retry_handler,
    ]
