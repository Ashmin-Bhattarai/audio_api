import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple

import librosa
import numpy as np
from loguru import logger

MODEL_TARGET_SR = 16000

async def extract_audio_features(file_path: Path) -> Tuple[Dict[str, Any], np.ndarray, int]:
    """
    Asynchronously extracts features and loads audio data, ensuring the audio
    is resampled to the model's required sample rate (16000 Hz).

    Returns:
        A tuple containing:
        - A dictionary of the *original* audio features (duration, sample_rate, channels).
        - The audio time series (y) as a numpy array, resampled to 16000 Hz and mono.
        - The target sample rate (16000) as an integer.
    """
    logger.info(f"Extracting features and loading data from {file_path}")

    def _blocking_operation():
        try:
            y_orig, sr_orig = librosa.load(file_path, sr=None, mono=False)
            
            duration = librosa.get_duration(y=y_orig, sr=sr_orig)
            channels = y_orig.shape[0] if y_orig.ndim > 1 else 1
            
            original_features = {
                "duration": round(duration, 2),
                "sample_rate": sr_orig,
                "channels": channels,
            }

            # Convert to mono first
            y_mono = librosa.to_mono(y_orig.copy()) if y_orig.ndim > 1 else y_orig
            
            if sr_orig != MODEL_TARGET_SR:
                logger.debug(f"Resampling audio from {sr_orig}Hz to {MODEL_TARGET_SR}Hz.")
                y_resampled = librosa.resample(y=y_mono.copy(), orig_sr=sr_orig, target_sr=MODEL_TARGET_SR)
            else:
                y_resampled = y_mono
            
            return original_features, y_resampled, sr_orig

        except Exception as e:
            raise ValueError(f"Librosa failed to load or process file: {e}")

    try:
        features, y_resampled, target_sr = await asyncio.to_thread(_blocking_operation)
        logger.success(f"Successfully extracted features: {features}")
        return features, y_resampled, target_sr
    except ValueError as e:
        logger.error(f"Feature extraction failed for {file_path}: {e}")
        raise e
