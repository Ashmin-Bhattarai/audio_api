import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from audio_api.ml_classifier import (
    AudioClassificationModel,
    classify_audio_with_model,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_huggingface_model(monkeypatch):
    """
    A pytest fixture to mock the transformers library's from_pretrained methods.
    This prevents the actual download of the large model during testing.
    """
    mock_extractor = MagicMock()
    mock_model = MagicMock()

    mock_model.config.id2label = {
        0: "Speech",
        1: "Violin, fiddle",
        2: "Siren",
        3: "Silence",
        4: "Cat",
    }

    monkeypatch.setattr(
        "transformers.AutoFeatureExtractor.from_pretrained",
        lambda model_id: mock_extractor,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForAudioClassification.from_pretrained",
        lambda model_id: mock_model,
    )

    AudioClassificationModel._instance = None

    return mock_model, mock_extractor


async def test_class_mapping_creation(mock_huggingface_model):
    """

    Verify that our custom mapping from specific to general classes is created correctly.
    """
    model_instance = AudioClassificationModel()

    # Assert
    mapping = model_instance.specific_to_general_mapping
    assert mapping["Speech"] == "speech"
    assert mapping["Violin, fiddle"] == "music"
    assert mapping["Siren"] == "noise"
    assert "Cat" not in mapping


async def test_classify_music(mock_huggingface_model):
    """
    Test that audio is correctly classified as 'music' when the model's top
    prediction is a music-related subclass.
    """
    mock_model, _ = mock_huggingface_model

    fake_logits = torch.tensor([[0.1, 5.0, 0.2, 0.3, 0.4]])
    mock_model.return_value.logits = (
        fake_logits  # Configure the mock to return these logits
    )

    dummy_audio = np.random.randn(16000)
    classification = await classify_audio_with_model(dummy_audio, 16000)

    assert classification == "music"


async def test_classify_noise_from_unmapped_class(mock_huggingface_model):
    """
    Test that if the top prediction is a class we don't map (like 'Cat'),
    the final classification is the one with the highest score among our
    required classes (or a reasonable fallback like 'noise').
    """
    mock_model, _ = mock_huggingface_model

    fake_logits = torch.tensor([[0.1, 0.2, 0.5, 0.3, 10.0]])
    mock_model.return_value.logits = fake_logits

    dummy_audio = np.random.randn(16000)
    classification = await classify_audio_with_model(dummy_audio, 16000)

    assert classification == "noise"
