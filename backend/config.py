from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Document QA API"
    openai_api_key: str = Field(default="")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    embedding_batch_size: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.45
    faiss_index_dir: Path = Path("./data/indexes")
    run_live_openai_tests: bool = False
    cors_allow_origins: list[str] = ["http://localhost:5173"]
    max_upload_size_mb: int = 10
    max_chat_history_messages: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
