"""
config.py — Configuración centralizada del proyecto.
Usa Pydantic Settings para validar variables de entorno.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del proyecto cargada desde .env o variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(
        ..., description="Clave de API de Anthropic (Claude)"
    )
    openai_api_key: str | None = Field(
        default=None, description="Clave de OpenAI (opcional, para embeddings)"
    )

    # ── LLM Config ───────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Modelo de Claude a utilizar",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=8192)
    max_retries: int = Field(default=3, ge=1, le=5)

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Modelo de embeddings. Cambia a 'text-embedding-3-small' si usas OpenAI.",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    vectorstore_dir: Path = Field(default=Path("vectordb"))
    charts_dir: Path = Field(default=Path("charts"))

    # ── RAG Config ───────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=800)
    retriever_k: int = Field(default=5, ge=1, le=20)

    # ── UI Config ────────────────────────────────────────────────────────────
    gradio_port: int = Field(default=7860, ge=1024, le=65535)
    gradio_share: bool = Field(default=False)

    def ensure_dirs(self) -> None:
        """Crea los directorios necesarios si no existen."""
        for d in [self.data_dir, self.vectorstore_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Instancia global
settings = Settings()  # type: ignore[call-arg]  # se inyecta desde .env
settings.ensure_dirs()
