import asyncio
import librosa
import numpy as np
from loguru import logger


async def classify_audio(y: np.ndarray, sr: str):
    def _blocking_classification_():

        rms_energy = np.mean(librosa.feature.rms(y=y))
        
        # A very low energy level indicates silence.
        if rms_energy < 0.005:
            return 'silence'
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        logger.debug(f"Classifier metrics: RMS={rms_energy:.4f}, ZCR={zcr:.4f}, Centroid={spectral_centroid:.2f}")
        
        if spectral_centroid < 1000 and zcr < 0.1:
            return "speech"
        elif spectral_centroid > 1200 and spectral_centroid < 3000 and zcr > 0.15:
            return "music"
        else:
            # Default to noise for sounds that don't fit other categories.
            return "noise"
        
    def _blocking_classification():
        rms_energy = np.mean(librosa.feature.rms(y=y))
        if rms_energy < 0.005:
            return "silence"

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        logger.debug(f"--- STARTING CLASSIFICATION ---")
        logger.debug(f"Metrics: Centroid={spectral_centroid:.2f}, ZCR={zcr:.4f}")

        is_speech_centroid = spectral_centroid < 1000
        is_speech_zcr = zcr < 0.1
        
        logger.debug(f"Checking SPEECH: (Centroid < 1000 -> {is_speech_centroid}) AND (ZCR < 0.1 -> {is_speech_zcr})")
        if is_speech_centroid and is_speech_zcr:
            logger.debug("Result: Matched SPEECH.")
            return "speech"

        is_music_centroid = spectral_centroid > 1200 and spectral_centroid < 3500
        is_music_zcr = zcr < 0.12
        logger.debug(f"Checking MUSIC: (Centroid in [1200, 3500] -> {is_music_centroid}) AND (ZCR < 0.12 -> {is_music_zcr})")
        
        if is_music_centroid and is_music_zcr:
            logger.debug("Result: Matched MUSIC.")
            return "music"

        logger.debug("Result: No match found. Falling back to NOISE.")
        return "noise"

    classification = await asyncio.to_thread(_blocking_classification)
    logger.info(f"Audio classified as: {classification}")
    return classification
