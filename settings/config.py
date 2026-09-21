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

    HF_TOKEN: str = Field(..., env="HF_TOKEN") # type: ignore

    WHISPER_MODEL_SIZE: str = Field("small", env="WHISPER_MODEL_SIZE") # type: ignore
    WHISPER_MODEL_DEVICE: str = Field("cpu", env="WHISPER_MODEL_DEVICE") # type: ignore
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE") # type: ignore

    WHISPER_MODEL_SIZE_OR_PATH: str = Field("/models/whisper-german-ct2", env="WHISPER_MODEL_SIZE_OR_PATH") # type: ignore

    # Diarization tuning parameters
    DIARIZATION_CLUSTERING_THRESHOLD: float = Field(0.65, env="DIARIZATION_CLUSTERING_THRESHOLD") # type: ignore
    DIARIZATION_MIN_DURATION_OFF: float = Field(0.1, env="DIARIZATION_MIN_DURATION_OFF") # type: ignore
    DIARIZATION_MIN_CLUSTER_SIZE: int = Field(12, env="DIARIZATION_MIN_CLUSTER_SIZE") # type: ignore

    # Diarization subprocess timeout tuning (kills+respawns the diarization worker
    # process if it hangs — see run_diarization_in_subprocess in app/whisper/service.py).
    # TODO(wisdom): DIARIZATION_WORST_CASE_RATIO is an UNVERIFIED PLACEHOLDER (1.0 = the
    # pipeline processes audio at real-time speed in the worst observed case). This has
    # NOT been measured against production throughput/latency logs. Tune it once you
    # have real worst-case numbers — a wrong value either times out legitimately slow
    # long files, or waits too long to recover from a genuine hang.
    DIARIZATION_WORST_CASE_RATIO: float = Field(1.0, env="DIARIZATION_WORST_CASE_RATIO") # type: ignore
    DIARIZATION_TIMEOUT_SAFETY_MULTIPLIER: float = Field(2.0, env="DIARIZATION_TIMEOUT_SAFETY_MULTIPLIER") # type: ignore
    DIARIZATION_MIN_TIMEOUT_S: float = Field(300.0, env="DIARIZATION_MIN_TIMEOUT_S") # type: ignore
    DIARIZATION_MAX_TIMEOUT_S: float = Field(3600.0, env="DIARIZATION_MAX_TIMEOUT_S") # type: ignore

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
