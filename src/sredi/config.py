from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/sred_db"

    # LLM Settings - Loaded from .env or environment variables
    LLM_PROVIDER: str = "openai"  # "openai" or "gemini"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-pro"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 2

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
