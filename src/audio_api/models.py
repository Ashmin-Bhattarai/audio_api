from pydantic import BaseModel, HttpUrl

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