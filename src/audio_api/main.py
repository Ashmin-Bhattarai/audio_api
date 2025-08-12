from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from loguru import logger

from audio_api.audio_downloader import download_audio_file
from audio_api.audio_processor import extract_audio_features
from audio_api.audio_classifier import classify_audio


class AnalyzeRequest(BaseModel):
    """The request model for the API endpoint."""
    audio_url: HttpUrl

class AudioFeaturesResponse(BaseModel):
    """The data model for the successful response."""
    duration: float
    sample_rate: int
    channels: int
    classification: str

class SuccessResponse(BaseModel):
    """The top-level success response model."""
    status: str = "success"
    data: AudioFeaturesResponse


app = FastAPI(
    title="Async Audio Analyzer API",
    description="An API to analyze and classify audio files from a URL.",
    version="1.0.0",
)

def cleanup_file(path: Path):
    """Utility function to remove a file and log it."""
    if path.exists():
        path.unlink()
        logger.info(f"Cleaned up temporary file: {path}")

@app.post("/analyze-audio", response_model=SuccessResponse)
async def analyze_audio_endpoint(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Accepts an audio file URL, downloads and analyzes it, and returns classification.
    """
    temp_file_path: Path | None = None
    try:
        logger.info(f"Received request for URL: {request.audio_url}")
        temp_file_path = await download_audio_file(str(request.audio_url))
        
        background_tasks.add_task(cleanup_file, temp_file_path)

        features, y_mono, sr = await extract_audio_features(temp_file_path)

        classification = await classify_audio(y_mono, sr)

        response_data = AudioFeaturesResponse(
            duration=features["duration"],
            sample_rate=features["sample_rate"],
            channels=features["channels"],
            classification=classification,
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
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Async Audio Analyzer API. POST to /analyze-audio."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)