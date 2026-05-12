"""Centralized config loaded from .env via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    groq_api_key: str
    langchain_api_key: str | None = None
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_project: str = "documind"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documind"

    # Models
    groq_model_smart: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    groq_model_vision: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # Retrieval
    top_k_retrieve: int = 20
    top_k_rerank: int = 6

    # Paths (derived; not loaded from env)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"


settings = Settings()
