# Async Audio Analyzer & Classifier API

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-blue)](https://fastapi.tiangolo.com/)
[![ML Model](https://img.shields.io/badge/Model-Hugging_Face_AST-yellow)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

An asynchronous microservice that analyzes an audio file from a URL. It extracts metadata, classifies the audio content using a state-of-the-art machine learning model, and caches the results in Redis.

## 🚀 Features

-   **Async Processing**: Built with FastAPI and `asyncio` for high-performance, non-blocking I/O.
-   **Audio Feature Extraction**: Uses `librosa` to calculate duration, sample rate, and channels.
-   **ML-Powered Classification**: Employs a pre-trained **Audio Spectrogram Transformer (AST)** model from Hugging Face to classify audio into one of four high-level categories: `speech`, `music`, `noise`, or `silence`.
-   **Intelligent Caching**: Caches results in Redis with a configurable expiry time to avoid re-processing repeated URLs.
-   **Containerized**: Fully containerized with Docker and Docker Compose for easy setup and deployment.
-   **Structured Logging**: Logs are saved to a rotating file in the `logs/` directory for easy monitoring.

---

## 🛠️ Tech Stack

-   **Backend**: FastAPI, Uvicorn
-   **Audio Processing**: `librosa`
-   **Machine Learning**: `torch`, Hugging Face `transformers`
-   **Caching**: `redis-py`
-   **Async HTTP**: `httpx`, `aiofiles`
-   **Containerization**: Docker, Docker Compose
-   **Testing**: `pytest`

---

## 🏁 Getting Started

### Prerequisites

-   [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
-   An internet connection (for the initial model download)

### 🚀 Running with Docker (Recommended)

This is the simplest way to get the entire application stack (API + Redis) running.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Build and run the services:**
    ```bash
    docker compose up --build
    ```

    -   The `--build` flag is only needed the first time you run it.
    -   The first launch will be slow as it downloads Docker images, Python dependencies, and the ~330MB ML model. Subsequent launches will be much faster.

3.  **The API is now live!**
    -   **API URL**: `http://localhost:8000`
    -   **Interactive Docs (Swagger UI)**: `http://localhost:8000/docs`

---

## 🔬 API Usage

### Analyze Audio

Send a `POST` request to the `/analyze-audio` endpoint with a JSON payload containing the audio URL.

#### Request

-   **Endpoint**: `POST /analyze-audio`
-   **Body**:
    ```json
    {
      "audio_url": "https://www.learningcontainer.com/download/wav-file-sample/?wpdmdl=1679&refresh=68814a6666d771753303654"
    }
    ```

#### Success Response (200 OK)

```json
{
  "status": "success",
  "data": {
    "duration": 348.06,
    "sample_rate": 22050,
    "channels": 2,
    "classification": "music"
  }
}
```

-   The first request to a new URL will be slower as it performs the full analysis.
-   Subsequent requests to the same URL will be served instantly from the Redis cache.

---

## 📁 Project Structure

```
├── docker-compose.yaml
├── Dockerfile
├── logs
│   └── audio_api.log
├── pyproject.toml
├── README.md
├── src
│   └── audio_api
│       ├── audio_classifier.py
│       ├── audio_downloader.py
│       ├── audio_processor.py
│       ├── config.py
│       ├── __init__.py
│       ├── log_config.py
│       ├── main.py
│       ├── ml_classifier.py
│       ├── models.py
├── tests
│   ├── test_audio_classifier.py
│   ├── test_audio_downloader.py
│   ├── test_audio_processor.py
│   └── test_ml_classifier.py
└── uv.lock
```
