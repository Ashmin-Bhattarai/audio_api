import asyncio
from pathlib import Path
from typing import Dict, Any

import librosa
import numpy as np
from loguru import logger


async def extract_audio_features(file_path: Path) -> Dict[str, Any]:
    logger.info(f"Extracting features from {file_path} using librosa")

    def _blocking_operation():
        """A helper function with the synchronous, CPU-bound librosa code."""

        try:
            y, sr = librosa.load(file_path, sr=None, mono=False)

            y_mono = librosa.to_mono(y) if y.ndim > 1 else y

            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)

            # Get channels
            if y.ndim == 1:
                channels = 1
            else:
                channels = y.shape[0]

            features = {
                "duration": round(duration, 2),
                "sample_rate": sr,
                "channels": channels,
            }
            return features, y_mono, sr

        except Exception as e:
            raise ValueError(
                f"Librosa failed to load or process file: {repr(e)}"
            )

    try:
        features, y_mono, sr = await asyncio.to_thread(_blocking_operation)
        logger.success(f"Successfully extracted features: {features}")
        return features, y_mono, sr

    except Exception as e:
        logger.error(f"Feature extraction failed for {file_path}: {e}")
        raise e
