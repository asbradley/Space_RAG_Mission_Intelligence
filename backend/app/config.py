from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app config, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://space_rag:space_rag@localhost:5433/space_rag"
    raw_storage_dir: Path = Path("../data/raw")
    ntrs_api_base: str = "https://ntrs.nasa.gov/api/citations/search"


settings = Settings()
