from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_api_token: str = ""
    database_url: str = "sqlite:///./backend/data/case_linking.db"
    storage_dir: Path = Path("./backend/private_uploads")
    max_upload_mb: int = 50
    max_pdf_pages: int = 200
    ocr_languages: str = "eng+hin+mar"
    paddle_comparison_language: Literal["en", "hi", "mr"] = "en"
    ocr_dpi: int = 250
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
