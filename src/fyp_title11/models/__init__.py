"""Models for Hum2Tune melody recognition.

Available architectures:
    CNNLSTMModel     — CNN + bidirectional LSTM + temporal attention baseline.
    AudioTransformer — Transformer-encoder model for ablation comparison (RQ3).
    PitchCNN         — Lightweight CNN operating on raw pitch contours.
    FusionModel      — Late-fusion of CNNLSTMModel and PitchCNN embeddings.

Usage:
    from fyp_title11.models import CNNLSTMModel, AudioTransformer
"""

from .cnn_lstm import CNNLSTMModel, PitchCNN, FusionModel
from .audio_transformer import AudioTransformer

__all__ = [
    "CNNLSTMModel",
    "AudioTransformer",
    "PitchCNN",
    "FusionModel",
]
