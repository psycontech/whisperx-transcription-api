import os
import certifi
from pathlib import Path
from fastapi import Depends
from pydantic import Field
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Annotated, Literal, Union, cast

load_dotenv()

EnvironmentType = Literal["development", "production"]
env = os.getenv("PYTHON_ENV", "development")
PYTHON_ENV: EnvironmentType = cast(EnvironmentType, env)

# Core application paths
_BASE_DIR: Path = Path(__file__).resolve().parent.parent
BASE_DIR: Path = _BASE_DIR
CERTIFICATE: str = os.path.join(os.path.dirname(certifi.__file__), "cacert.pem")
DOTENV: str = os.path.join(_BASE_DIR, ".env")


class APIDocsConfig(BaseSettings):
    """API Documentation configurations."""

    API_DOCS_USERNAME: str = Field("admin", env="API_DOCS_USERNAME")  # type: ignore
    API_DOCS_PASSWORD: str = Field("password", env="API_DOCS_PASSWORD")  # type: ignore
    API_DOCS_URL: str = Field("/docs", env="API_DOCS_URL")  # type: ignore
    API_REDOC_URL: str = Field("/redoc", env="API_REDOC_URL")  # type: ignore
    OPENAPI_URL: str = Field("/openapi.json", env="OPENAPI_URL")  # type: ignore

class GlobalConfig(BaseSettings):
    """Base configuration class with shared settings across environments."""

    APP_NAME: str ="Whisper Transcription API"
    APP_ISS: str = "whisper"
    APP_VERSION: str = "0.0.1"
    APPLICATION_CERTIFICATE: str = Field(default=CERTIFICATE)
    BASE_DIR: Path = Field(default=_BASE_DIR)

    ENVIRONMENT: EnvironmentType = PYTHON_ENV

    HF_TOKEN: str = Field("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")

    WHISPER_MODEL_SIZE: str = Field("small", env="WHISPER_MODEL_SIZE") # type: ignore
    WHISPER_MODEL_DEVICE: str = Field("cpu", env="WHISPER_MODEL_DEVICE") # type: ignore
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE") # type: ignore

    # Paths (computed from BASE_DIR at init)
    UPLOAD_DIR: Path = Field(default=_BASE_DIR / "uploads")

    # Configs
    API_DOCS: APIDocsConfig = Field(default_factory=APIDocsConfig)  # type: ignore


class DevelopmentConfig(GlobalConfig):
    """Development environment specific configurations."""
    DEBUG: bool = True
 


class ProductionConfig(GlobalConfig):
    """Production environment specific configurations."""
    DEBUG: bool = False


ConfigType = Union[DevelopmentConfig, ProductionConfig]

@lru_cache()
def get_settings() -> ConfigType:
    """Factory function to get environment-specific settings."""
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    if not PYTHON_ENV or PYTHON_ENV not in configs:
        raise ValueError(
            f"Invalid deployment environment: `{env}`. Must be one of: {list(configs.keys())}"
        )
    return configs[PYTHON_ENV]()  # type: ignore


settings = get_settings()
SettingsDep = Annotated[ConfigType, Depends(get_settings)]
