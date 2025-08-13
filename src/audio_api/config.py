import os
from typing import Final


API_TITLE: Final[str] = "Async Audio Analyzer API"
API_DESCRIPTION: Final[str] = (
    "An API to analyze and classify audio files from a URL."
)
API_VERSION: Final[str] = "1.0.0"

API_HOST: Final[str] = os.getenv("API_HOST", "0.0.0.0")
API_PORT: Final[int] = int(os.getenv("API_PORT", 8000))

REDIS_HOST: Final[str] = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: Final[int] = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB: Final[int] = int(os.getenv("REDIS_DB", 0))
REDIS_URL: Final[str] = os.getenv(
    "REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
)

CACHE_EXPIRATION_SECONDS: Final[int] = int(
    os.getenv("CACHE_EXPIRATION_SECONDS", 3600)
)
