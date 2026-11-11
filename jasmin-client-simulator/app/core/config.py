import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Digital Twin Client Simulator"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./simulator.db")
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")  # mock | groq | openai | anthropic
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    
    # Groq Provider Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # A simulator turn must fail over to its deterministic client rather than leave the UI waiting on a provider.
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "12"))
    # Used in turn when a model's quota is exhausted (Groq counts tokens per day per model).
    # llama-3.3-70b-versatile was retired by Groq (404 model_not_found), which sent
    # every client to the deterministic templates once gpt-oss quotas ran out.
    GROQ_FALLBACK_MODELS: list = [m.strip() for m in os.getenv(
        "GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",") if m.strip()]
    # Optional last LLM before the templates, through OpenRouter's OpenAI-compatible API.
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")

    # Optional public bridge to the Ozone customer-advisor.  This keeps the
    # simulator independent: it talks to Amira only over the documented API.
    OZONE_ADVISOR_URL: str = os.getenv("OZONE_ADVISOR_URL", "")
    OZONE_TENANT_ID: str = os.getenv("OZONE_TENANT_ID", "ozone-voyages")
    OZONE_AGENT_ID: str = os.getenv("OZONE_AGENT_ID", "customer-advisor")
    OZONE_ADVISOR_TIMEOUT_SECONDS: float = float(os.getenv("OZONE_ADVISOR_TIMEOUT_SECONDS", "45"))
    
    # Everything the app shows and stores is in TND; these rates only convert a budget typed in USD/EUR (questionnaire)
    TND_PER_USD: float = 3.1
    EUR_PER_USD: float = 0.92

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
