from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Service ports
    ANKI_SERVICE_PORT: int = 8001
    NOTION_SERVICE_PORT: int = 8002
    GLUE_SERVICE_PORT: int = 8003
    API_PORT: int = 8000

    # AnkiConnect endpoint
    ANKI_CONNECT_URL: str = "http://localhost:8765"

    # Notion integration token
    NOTION_TOKEN: str = ""

    # Database connection URL (SQLite)
    DATABASE_URL: str = "sqlite:///./mapping.db"

    # Ngrok configuration
    NGROK_AUTHTOKEN: Optional[str] = None
    NGROK_DOMAIN: Optional[str] = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"  # Allow extra fields in environment variables
    }

# Create global settings instance
settings = Settings() 