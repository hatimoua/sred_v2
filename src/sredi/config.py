from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/sred_db"

    # LLM Settings (OpenAI)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 2

    # Tournament Config
    SHADOW_MODE: bool = True       # ENABLED: Run both routers
    ROUTER_TYPE: str = "llm"       # PRIMARY: The Context-Aware Router

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
