import json
from typing import List

def process_cors_origins(cors_origins: str | List[str] | None) -> List[str]:
    """
    Process CORS origins into a list of strings.

    Args:
        cors_origins: Either a string (comma-separated or JSON list), a list of strings, or None.

    Returns:
        List of allowed CORS origins.
    """
    default_origins: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:3001",
    ]

    if isinstance(cors_origins, str):
        if cors_origins == "*":
            return ["*"]
        if not cors_origins:
            return default_origins
        if cors_origins.startswith("[") and cors_origins.endswith("]"):
            return default_origins + json.loads(cors_origins) # type: ignore
        else:  # comma-separated
            return default_origins + [
                origin.strip()
                for origin in cors_origins.split(",")
                if origin.strip()
            ]
    elif isinstance(cors_origins, list):
        return default_origins + cors_origins

    return default_origins
