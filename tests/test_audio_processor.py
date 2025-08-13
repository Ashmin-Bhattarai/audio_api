import pytest
from pathlib import Path
import struct
import numpy as np

from audio_api.audio_processor import extract_audio_features


def create_fake_wav_file(
    path: Path, duration_s: int, channels: int, sample_rate: int
):
    """
    Creates a minimal, valid WAV file for testing purposes.
    """

    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    num_samples = sample_rate * duration_s
    data_size = num_samples * channels * bits_per_sample // 8
    chunk_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", chunk_size))
        f.write(b"WAVE")
        # "fmt " sub-chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # "data" sub-chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        # Write empty audio data (silence)
        f.write(b"\0" * data_size)


@pytest.mark.asyncio
async def test_extract_features_sucess(tmp_path: Path):
    """
    Tests that features are correctly extracted from a valid WAV file.
    """

    sample_rate = 44100
    channels = 2
    duration = 2.0
    fake_audio_file = tmp_path / "test.wav"
    create_fake_wav_file(fake_audio_file, int(duration), channels, sample_rate)

    features, y_mono, sr = await extract_audio_features(fake_audio_file)
    assert isinstance(features, dict)
    assert features["duration"] == duration
    assert features["sample_rate"] == sample_rate
    assert features["channels"] == channels
    assert fake_audio_file.exists()

    assert isinstance(y_mono, np.ndarray)
    assert y_mono.ndim == 1
    assert isinstance(sr, int)
    assert sr == sample_rate

    assert fake_audio_file.exists()
    fake_audio_file.unlink()



@pytest.mark.asyncio
async def test_extract_features_invalid_file(tmp_path: Path):
    """
    Tests that the function raises a ValueError for a non-audio file.
    """
    
    invalid_file = tmp_path / "not_audio.txt"
    invalid_file.write_text("this is just a text file, not audio")

    with pytest.raises(ValueError):
        await extract_audio_features(invalid_file)

    invalid_file.unlink()