# tests/test_main_api.py

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from audio_api.main import app

client = TestClient(app)

TEST_URL = "https://example.com/test.wav"


def test_successful_analysis(monkeypatch):
    """
    Test the full success path of the /analyze-audio endpoint.
    We mock our internal functions to simulate a successful analysis.
    """
    async def mock_download(url):
        return Path("/tmp/fake_audio.wav")
    
    async def mock_extract(path):
        features = {"duration": 5.23, "sample_rate": 44100, "channels": 2}
        fake_audio_data = (None, None) 
        return features, fake_audio_data[0], fake_audio_data[1]

    async def mock_classify(y, sr):
        return "music"
    
    def mock_cleanup(path):
        print(f"Mock cleanup called for {path}")
        pass

    monkeypatch.setattr("audio_api.main.download_audio_file", mock_download)
    monkeypatch.setattr("audio_api.main.extract_audio_features", mock_extract)
    monkeypatch.setattr("audio_api.main.classify_audio", mock_classify)
    monkeypatch.setattr("audio_api.main.cleanup_file", mock_cleanup)

    response = client.post("/analyze-audio", json={"audio_url": TEST_URL})

    assert response.status_code == 200
    expected_data = {
        "status": "success",
        "data": {
            "duration": 5.23,
            "sample_rate": 44100,
            "channels": 2,
            "classification": "music",
        }
    }
    assert response.json() == expected_data


def test_download_fails(monkeypatch):
    """
    Test the case where the download fails, expecting a 400 error.
    """
    async def mock_download_fails(url):
        raise ValueError("Download failed: 404 Not Found")

    monkeypatch.setattr("audio_api.main.download_audio_file", mock_download_fails)

    response = client.post("/analyze-audio", json={"audio_url": TEST_URL})

    assert response.status_code == 400
    assert response.json() == {"detail": "Download failed: 404 Not Found"}