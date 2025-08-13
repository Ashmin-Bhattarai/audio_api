from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from loguru import logger

from audio_api.audio_downloader import download_audio_file
from audio_api.audio_processor import extract_audio_features
from audio_api.audio_classifier import classify_audio
from audio_api.models import (
    AnalyzeRequest,
    AudioFeaturesResponse,
    SuccessResponse,
)
from audio_api.ml_classifier import classify_audio_with_model
from audio_api import config
from audio_api.log_config import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.redis = redis.from_url(config.REDIS_URL, decode_responses=True)
    logger.info("Successfully connected to Redis.")
    yield

    await app.state.redis.close()
    logger.info("Redis connection closed.")


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)


def cleanup_file(path: Path):
    """Utility function to remove a file and log it."""
    if path.exists():
        path.unlink()
        logger.info(f"Cleaned up temporary file: {path}")


@app.post("/analyze-audio", response_model=SuccessResponse)
async def analyze_audio_endpoint(
    request: AnalyzeRequest, background_tasks: BackgroundTasks
):
    """
    Accepts an audio file URL, downloads and analyzes it, and returns classification.
    Results are cached in Redis
    """
    cache_key = f"audio_cache:{request.audio_url}"
    temp_file_path: Path | None = None

    try:
        logger.info(f"Received request for URL: {request.audio_url}")
        cached_result = await app.state.redis.get(cache_key)
        if cached_result:
            logger.success(f"Cache hit for URL: {request.audio_url}")
            cached_data = AudioFeaturesResponse.model_validate_json(
                cached_result
            )
            return SuccessResponse(data=cached_data)

        logger.info(
            f"Cache miss for URL: {request.audio_url}. Starting analysis."
        )

        temp_file_path = await download_audio_file(str(request.audio_url))
        background_tasks.add_task(cleanup_file, temp_file_path)

        features, y_mono, sr = await extract_audio_features(temp_file_path)

        classification = await classify_audio_with_model(y_mono, sr)

        response_data = AudioFeaturesResponse(
            duration=features["duration"],
            sample_rate=features["sample_rate"],
            channels=features["channels"],
            classification=classification,
        )

        await app.state.redis.set(
            cache_key,
            response_data.model_dump_json(),
            ex=config.CACHE_EXPIRATION_SECONDS,
        )

        return SuccessResponse(data=response_data)

    except (ValueError, httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error(f"A known error occurred: {e}")
        if temp_file_path:
            cleanup_file(temp_file_path)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"An unexpected internal server error occurred: {e}")
        if temp_file_path:
            cleanup_file(temp_file_path)
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Async Audio Analyzer API. POST to /analyze-audio."
    }


def main():
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    main()