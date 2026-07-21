"""
tools.py — 7 herramientas del agente RAG corporativo
Base (4): search_documents, analyze_data, generate_chart, summarize_text
Avanzadas (3): smart_chart_detector, cross_document_analyst, error_retry_handler
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend no-GUI para servidores
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool

from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# HERRAMIENTAS BASE (4)
# ===========================================================================

@tool
def search_documents(query: str, k: int = 5, filter_filename: Optional[str] = None) -> str:
    """
    Busca información relevante en los documentos corporativos indexados.

    Args:
        query: Pregunta o término de búsqueda.
        k: Número máximo de fragmentos a recuperar (default 5).
        filter_filename: Filtrar por nombre de archivo específico (opcional).

    Returns:
        Fragmentos relevantes con metadatos de fuente.
    """
    try:
        vs = get_vectorstore()
        search_kwargs: dict = {"k": k}
        if filter_filename:
            search_kwargs["filter"] = {"source": filter_filename}

        results = vs.similarity_search_with_relevance_scores(query, **search_kwargs)

        if not results:
            return "No se encontraron documentos relevantes para la consulta."

        output_parts = []
        for doc, score in results:
            source = doc.metadata.get("source", "desconocido")
            page = doc.metadata.get("page", "")
            page_str = f", página {page}" if page else ""
            output_parts.append(
                f"📎 **Fuente:** `{source}`{page_str} (relevancia: {score:.2f})\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as exc:
        logger.error("search_documents failed: %s", exc)
        return f"Error al buscar en documentos: {exc}"


@tool
def analyze_data(filename: str, question: str) -> str:
    """
    Analiza datos tabulares (CSV, Excel) y responde preguntas cuantitativas.

    Args:
        filename: Nombre del archivo en la carpeta data/ (ej: 'ventas.xlsx').
        question: Pregunta sobre los datos.

    Returns:
        Análisis estadístico en texto estructurado.
    """
    try:
        data_path = Path("data") / filename
        if not data_path.exists():
            return f"Archivo `{filename}` no encontrado en data/."

        ext = data_path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(data_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(data_path)
        else:
            return f"Formato `{ext}` no soportado para análisis tabular."

        # Análisis descriptivo automático
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        summary = df[numeric_cols].describe().round(2).to_string() if numeric_cols else "Sin columnas numéricas"

        rows, cols = df.shape
        col_names = ", ".join(df.columns.tolist()[:15])  # máx 15 columnas en display

        return (
            f"**Archivo:** `{filename}` — {rows:,} filas × {cols} columnas\n"
            f"**Columnas:** {col_names}\n\n"
            f"**Estadísticas descriptivas:**\n```\n{summary}\n```\n\n"
            f"**Pregunta:** {question}\n"
            f"*Usa generate_chart o smart_chart_detector para visualizar los datos.*"
        )

    except Exception as exc:
        logger.error("analyze_data failed: %s", exc)
        return f"Error al analizar `{filename}`: {exc}"


@tool
def generate_chart(
    filename: str,
    x_column: str,
    y_column: str,
    chart_type: str = "bar",
    title: str = "",
) -> str:
    """
    Genera un gráfico desde datos tabulares y lo guarda como imagen PNG.

    Args:
        filename: Nombre del archivo CSV/Excel en data/.
        x_column: Columna para el eje X.
        y_column: Columna para el eje Y.
        chart_type: Tipo de gráfico: 'bar', 'line', 'scatter', 'pie', 'area'.
        title: Título del gráfico (opcional).

    Returns:
        Ruta al archivo PNG generado con prefijo 'CHART:' para detección automática.
    """
    try:
        data_path = Path("data") / filename
        ext = data_path.suffix.lower()
        df = pd.read_csv(data_path) if ext == ".csv" else pd.read_excel(data_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        color_palette = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]

        if chart_type == "bar":
            df.plot(kind="bar", x=x_column, y=y_column, ax=ax,
                    color=color_palette[0], legend=False)
        elif chart_type == "line":
            df.plot(kind="line", x=x_column, y=y_column, ax=ax,
                    color=color_palette[0], linewidth=2.5, marker="o", legend=False)
        elif chart_type == "scatter":
            ax.scatter(df[x_column], df[y_column], color=color_palette[0],
                       alpha=0.7, s=60, edgecolors="#ffffff", linewidth=0.5)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        elif chart_type == "pie":
            df.set_index(x_column)[y_column].plot(
                kind="pie", ax=ax, autopct="%1.1f%%",
                colors=color_palette, legend=False)
            ax.set_ylabel("")
        elif chart_type == "area":
            df.plot(kind="area", x=x_column, y=y_column, ax=ax,
                    color=color_palette[0], alpha=0.6, legend=False)
        else:
            return f"Tipo de gráfico `{chart_type}` no soportado."

        chart_title = title or f"{y_column} por {x_column}"
        ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()

        timestamp = int(time.time())
        out_path = CHARTS_DIR / f"chart_{timestamp}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Gráfico generado: %s", out_path)
        return f"CHART:{out_path}"

    except Exception as exc:
        logger.error("generate_chart failed: %s", exc)
        return f"Error al generar gráfico: {exc}"


@tool
def summarize_text(filename: str, max_length: int = 800) -> str:
    """
    Genera un resumen estructurado de un documento corporativo.

    Args:
        filename: Nombre del documento en data/ (PDF, DOCX, TXT, MD).
        max_length: Longitud máxima del resumen en caracteres.

    Returns:
        Resumen con puntos clave, estructura y metadatos del documento.
    """
    try:
        data_path = Path("data") / filename
        if not data_path.exists():
            return f"Archivo `{filename}` no encontrado."

        ext = data_path.suffix.lower()
        text = ""

        if ext == ".txt" or ext == ".md":
            text = data_path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(data_path))
                text = " ".join(page.extract_text() or "" for page in reader.pages[:10])
            except ImportError:
                return "pypdf no instalado. Ejecuta: pip install pypdf"
        elif ext in (".docx",):
            try:
                from docx import Document
                doc = Document(str(data_path))
                text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return "python-docx no instalado. Ejecuta: pip install python-docx"
        else:
            return f"Formato `{ext}` no soportado para summarize_text."

        # Estadísticas básicas
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # Extracto representativo
        excerpt = text[:max_length].rsplit(" ", 1)[0] + "..." if len(text) > max_length else text

        return (
            f"**Documento:** `{filename}`\n"
            f"**Estadísticas:** {word_count:,} palabras | {char_count:,} caracteres\n\n"
            f"**Extracto:**\n{excerpt}\n\n"
            f"*Para análisis completo, usa search_documents con una pregunta específica.*"
        )

    except Exception as exc:
        logger.error("summarize_text failed: %s", exc)
        return f"Error al resumir `{filename}`: {exc}"


# ===========================================================================
# HERRAMIENTAS AVANZADAS (3)
# ===========================================================================

@tool
def smart_chart_detector(filename: str, analysis_goal: str) -> str:
    """
    Detecta automáticamente el mejor tipo de gráfico para los datos y la pregunta,
    y lo genera sin necesidad de especificar columnas manualmente.

    Args:
        filename: Nombre del archivo CSV/Excel en data/.
        analysis_goal: Descripción en lenguaje natural de lo que se quiere visualizar.
                       Ej: 'ventas por mes', 'distribución de edades', 'correlación precio-unidades'.

    Returns:
        Ruta al PNG generado con prefijo 'CHART:', o mensaje explicativo.
    """
    try:
        data_path = Path("data") / filename
        ext = data_path.suffix.lower()
        df = pd.read_csv(data_path) if ext == ".csv" else pd.read_excel(data_path)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        goal_lower = analysis_goal.lower()

        # Lógica de detección heurística
        if any(kw in goal_lower for kw in ["tiempo", "mes", "año", "fecha", "tendencia", "evolución"]):
            chart_type = "line"
            x_col = datetime_cols[0] if datetime_cols else categorical_cols[0] if categorical_cols else numeric_cols[0]
            y_col = numeric_cols[0] if numeric_cols else numeric_cols[1]

        elif any(kw in goal_lower for kw in ["distribución", "proporción", "porcentaje", "parte"]):
            chart_type = "pie"
            x_col = categorical_cols[0] if categorical_cols else numeric_cols[0]
            y_col = numeric_cols[0] if numeric_cols else numeric_cols[1]

        elif any(kw in goal_lower for kw in ["correlación", "relación", "dispersión", "vs"]):
            chart_type = "scatter"
            x_col = numeric_cols[0] if len(numeric_cols) >= 2 else categorical_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) >= 2 else numeric_cols[0]

        elif any(kw in goal_lower for kw in ["acumulado", "área", "total acumulativo"]):
            chart_type = "area"
            x_col = categorical_cols[0] if categorical_cols else numeric_cols[0]
            y_col = numeric_cols[0]

        else:  # default: barras comparativas
            chart_type = "bar"
            x_col = categorical_cols[0] if categorical_cols else numeric_cols[0]
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]

        logger.info("smart_chart_detector: tipo=%s x=%s y=%s", chart_type, x_col, y_col)

        # Delegar a generate_chart
        return generate_chart.invoke({
            "filename": filename,
            "x_column": x_col,
            "y_column": y_col,
            "chart_type": chart_type,
            "title": analysis_goal.capitalize(),
        })

    except Exception as exc:
        logger.error("smart_chart_detector failed: %s", exc)
        return f"Error al detectar gráfico automático: {exc}"


@tool
def cross_document_analyst(query: str, document_list: list[str]) -> str:
    """
    Analiza y cruza información de múltiples documentos simultáneamente,
    identificando consistencias, contradicciones y complementos entre fuentes.

    Args:
        query: Pregunta o tema a investigar en múltiples documentos.
        document_list: Lista de nombres de archivo a comparar (máx. 5).

    Returns:
        Análisis comparativo estructurado con hallazgos por documento.
    """
    if not document_list:
        return "Debes especificar al menos un documento para analizar."

    document_list = document_list[:5]  # limitar a 5 para control de tokens
    results: dict[str, str] = {}

    for doc_name in document_list:
        result = search_documents.invoke({
            "query": query,
            "k": 3,
            "filter_filename": doc_name,
        })
        results[doc_name] = result

    # Formatear análisis comparativo
    output = [f"## Análisis cruzado: *{query}*\n"]

    for doc_name, content in results.items():
        has_content = "No se encontraron" not in content
        status = "✅ Encontrado" if has_content else "❌ No encontrado"
        output.append(f"### `{doc_name}` — {status}")
        if has_content:
            # Tomar solo el primer fragmento para no saturar la respuesta
            first_fragment = content.split("---")[0].strip()
            output.append(first_fragment)
        output.append("")

    docs_with_info = [d for d, c in results.items() if "No se encontraron" not in c]

    if len(docs_with_info) > 1:
        output.append(
            f"---\n**📊 Conclusión:** El tema aparece en {len(docs_with_info)} de "
            f"{len(document_list)} documentos: {', '.join(f'`{d}`' for d in docs_with_info)}."
        )
    elif len(docs_with_info) == 1:
        output.append(
            f"---\n**📊 Conclusión:** Solo `{docs_with_info[0]}` contiene información sobre este tema."
        )
    else:
        output.append("---\n**📊 Conclusión:** Ningún documento contiene información sobre este tema.")

    return "\n".join(output)


@tool
def error_retry_handler(failed_query: str, error_description: str, attempt: int = 1) -> str:
    """
    Maneja errores del agente reformulando la consulta fallida con estrategias
    progresivas: simplificación → descomposición → búsqueda alternativa.

    Args:
        failed_query: La consulta original que falló.
        error_description: Descripción del error o resultado insatisfactorio.
        attempt: Número de intento actual (1-3).

    Returns:
        Consulta reformulada y estrategia sugerida.
    """
    strategies = {
        1: (
            "simplificación",
            f"Versión simplificada de tu consulta:\n"
            f"*{' '.join(failed_query.split()[:8])}*\n\n"
            f"Intenta buscar los términos clave por separado."
        ),
        2: (
            "descomposición",
            f"Descomponiendo la consulta en partes:\n"
            f"1. ¿Cuál es el tema principal? → usa `search_documents` con el tema."
            f"\n2. ¿Hay datos numéricos? → usa `analyze_data` con el archivo.\n"
            f"3. ¿Necesitas comparar? → usa `cross_document_analyst`."
        ),
        3: (
            "búsqueda amplia",
            f"Búsqueda con términos más generales:\n"
            f"Palabras clave extraídas: {', '.join(set(failed_query.lower().split()[:5]))}\n\n"
            f"Si el problema persiste, verifica que los documentos estén correctamente indexados."
        ),
    }

    attempt = min(max(attempt, 1), 3)
    strategy_name, suggestion = strategies[attempt]

    return (
        f"🔄 **Intento {attempt}/3 — Estrategia: {strategy_name}**\n\n"
        f"**Error detectado:** {error_description}\n\n"
        f"**Sugerencia:**\n{suggestion}\n\n"
        f"**Consulta original:** *{failed_query}*"
    )


# ===========================================================================
# API pública
# ===========================================================================

def get_all_tools() -> list:
    """Retorna la lista completa de herramientas disponibles para el agente."""
    return [
        search_documents,
        analyze_data,
        generate_chart,
        summarize_text,
        smart_chart_detector,
        cross_document_analyst,
        error_retry_handler,
    ]


def list_indexed_docs() -> list[str]:
    """Lista los documentos actualmente indexados en el vectorstore."""
    try:
        vs = get_vectorstore()
        collection = vs._collection  # ChromaDB internal
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        sources = sorted({m.get("source", "?") for m in metadatas if m})
        return [Path(s).name for s in sources]
    except Exception as exc:
        logger.warning("list_indexed_docs: %s", exc)
        # Fallback: listar archivos en data/
        data_dir = Path("data")
        if data_dir.exists():
            return sorted(f.name for f in data_dir.iterdir() if f.is_file())
        return []


def index_documents(file_paths: list[str]) -> int:
    """
    Indexa nuevos documentos en el vectorstore.

    Args:
        file_paths: Lista de rutas absolutas a archivos.

    Returns:
        Número de fragmentos indexados.
    """
    from langchain_community.document_loaders import (
        CSVLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredHTMLLoader,
        UnstructuredWordDocumentLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".csv": CSVLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs = []
    for fp in file_paths:
        path = Path(fp)
        loader_cls = LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            logger.warning("Formato no soportado: %s", path.suffix)
            continue
        try:
            loader = loader_cls(str(path))
            docs = loader.load()
            # Enriquecer metadatos
            for doc in docs:
                doc.metadata["source"] = path.name
                doc.metadata["file_path"] = str(path)
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
            logger.info("Indexado: %s (%d fragmentos)", path.name, len(chunks))
        except Exception as exc:
            logger.error("Error indexando %s: %s", path.name, exc)

    if all_docs:
        vs = get_vectorstore()
        vs.add_documents(all_docs)
        logger.info("Total fragmentos indexados: %d", len(all_docs))

    return len(all_docs)
