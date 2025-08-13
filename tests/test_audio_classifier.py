import pytest
import numpy as np

from audio_api.audio_classifier import classify_audio

SAMPLE_RATE = 22050
DURATION_S = 1

pytestmark = pytest.mark.asyncio


async def test_classify_silence():
    """
    Tests that an array of zeros is correctly classified as 'silence'.
    """
    y_silence = np.zeros(SAMPLE_RATE * DURATION_S)

    classification = await classify_audio(y_silence, SAMPLE_RATE)

    assert classification == "silence"


async def test_classify_speech():
    """
    Tests that a low-frequency tonal signal is classified as 'speech'.
    Our heuristic looks for low zero-crossing rate and low spectral centroid.
    A simple sine wave at a typical human voice frequency (e.g., 440Hz) fits this.
    """
    # Arrange: Create a 440Hz sine wave (simulates a vowel sound)
    t = np.linspace(0., DURATION_S, int(SAMPLE_RATE * DURATION_S))
    y_speech = 0.5 * np.sin(2 * np.pi * 440. * t)

    # Act
    classification = await classify_audio(y_speech, SAMPLE_RATE)

    # Assert
    assert classification == "speech"


async def test_classify_music():
    """
    Tests that a complex signal with high frequencies is classified as 'music'.
    Our heuristic looks for high spectral centroid and a reasonably high ZCR.
    Combining a low tone with a high tone creates this complexity.
    """
    # Arrange: Create a signal with low and high frequency components
    t = np.linspace(0., DURATION_S, int(SAMPLE_RATE * DURATION_S))
    y_low = np.sin(2 * np.pi * 440. * t)
    y_high = 0.3 * np.sin(2 * np.pi * 4000. * t) # High freq pushes centroid up
    y_music = y_low + y_high

    # Act
    classification = await classify_audio(y_music, SAMPLE_RATE)

    # Assert
    assert classification == "music"


async def test_classify_noise():
    """
    Tests that a signal that doesn't fit other categories falls back to 'noise'.
    We can create a mid-range signal that has a higher ZCR than speech but a
    lower spectral centroid than music, failing both specific checks.
    """
    # Arrange: Create a mid-frequency tone that fits neither speech nor music rules
    t = np.linspace(0., DURATION_S, int(SAMPLE_RATE * DURATION_S))
    y_noise = 0.5 * np.sin(2 * np.pi * 1500. * t)

    # Act
    classification = await classify_audio(y_noise, SAMPLE_RATE)

    # Assert
    assert classification == "noise"