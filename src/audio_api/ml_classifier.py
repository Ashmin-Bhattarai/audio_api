# ml_classifier.py

import asyncio
import collections
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from loguru import logger

REQUIRED_CLASSES = ["music", "speech", "noise", "silence"]

LABEL_MAPPING = {
    "music": [
        "music",
        "musical",
        "instrument",
        "singing",
        "choir",
        "song",
        "guitar",
        "piano",
        "drum",
        "orchestra",
        "symphony",
        "cello",
        "violin",
        "flute",
    ],
    "speech": [
        "speech",
        "speaking",
        "speech synthesizer",
        "chatter",
        "narration",
        "vocal music",
        "acapella",
    ],
    "noise": [
        "noise",
        "engine",
        "wind",
        "crackle",
        "siren",
        "gunshot",
        "explosion",
        "machine",
        "hiss",
        "hum",
        "rumble",
        "vehicle",
    ],
    "silence": ["silence"],
}


class AudioClassificationModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info(
                "Creating and loading AudioClassificationModel instance..."
            )
            cls._instance = super(AudioClassificationModel, cls).__new__(cls)
            cls._instance._load_model_and_mapping()
        return cls._instance

    def _load_model_and_mapping(self):
        """Loads the model, feature extractor, and creates the custom class mapping."""
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_id
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id
            )
            self._create_class_mapping()
            logger.success(
                "Hugging Face model, extractor, and class mapping loaded successfully."
            )
        except Exception as e:
            logger.critical(f"Failed to load Hugging Face model: {e}")
            raise

    def _create_class_mapping(self):
        """Builds a lookup table from the model's specific labels to our general classes."""
        self.specific_to_general_mapping = {}
        for i, label in self.model.config.id2label.items():
            label_lower = label.lower()
            for general_class, keywords in LABEL_MAPPING.items():
                if any(keyword in label_lower for keyword in keywords):
                    self.specific_to_general_mapping[label] = general_class
                    break

    def classify(self, y: np.ndarray, sr: int = 16000) -> str:
        """
        Performs targeted classification by aggregating probabilities.
        """
        inputs = self.feature_extractor(
            y, sampling_rate=sr, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        all_probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        class_probabilities = collections.defaultdict(float)

        for i, prob in enumerate(all_probs):
            specific_label = self.model.config.id2label[i]
            general_class = self.specific_to_general_mapping.get(
                specific_label, "other"
            )
            class_probabilities[general_class] += prob

        relevant_probs = {
            cls: class_probabilities.get(cls, 0.0) for cls in REQUIRED_CLASSES
        }

        log_probs = " | ".join(
            [f"{k}: {v:.2%}" for k, v in relevant_probs.items()]
        )
        logger.debug(f"Aggregated probabilities: {log_probs}")

        if not relevant_probs:
            return "noise"

        best_class = max(relevant_probs, key=relevant_probs.get)

        return best_class


async def classify_audio_with_model(y: np.ndarray, sr: int) -> str:
    """
    Asynchronous wrapper for the ML classification model.
    """
    model_instance = AudioClassificationModel()
    classification = await asyncio.to_thread(model_instance.classify, y)
    logger.info(f"Audio classified via ML model as: {classification}")
    return classification
