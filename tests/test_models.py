"""Model tests for Hum2Tune.

Tests cover:
- CNNLSTMModel forward shapes for both return_features modes
- AudioTransformer forward shapes
- PitchCNN and FusionModel shapes
- Parametrised channel/class/length combinations for ablation coverage
"""

import warnings
import pytest
import torch

# Suppress harmless PyTorch nested-tensor warning triggered by Pre-LN Transformer.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True", category=UserWarning)

from src.fyp_title11.models.cnn_lstm import CNNLSTMModel, PitchCNN, FusionModel
from src.fyp_title11.models.audio_transformer import AudioTransformer


# -----------------------------------------------------------------------------
#  CNN-LSTM
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("n_channels,num_classes,seq_len", [
    (39, 10, 431),   # default: 39-channel MFCC+delta, 431 frames @ 22050Hz/512hop
    (13, 10, 128),   # legacy 13-channel MFCC only
    (39, 99, 431),   # full song database size
])
def test_cnn_lstm_logits_shape(n_channels, num_classes, seq_len):
    """CNNLSTMModel default forward returns logits tensor only."""
    model = CNNLSTMModel(input_channels=n_channels, num_classes=num_classes)
    x = torch.randn(2, n_channels, seq_len)
    logits = model(x)                     # return_features=False by default
    assert logits.shape == (2, num_classes), (
        f"Expected logits shape (2, {num_classes}), got {logits.shape}"
    )


def test_cnn_lstm_return_features():
    """return_features=True yields a tuple (logits, dict) with context_vector."""
    model = CNNLSTMModel(input_channels=39, num_classes=7)
    x = torch.randn(2, 39, 128)
    logits, features = model(x, return_features=True)   # explicit flag required

    assert logits.shape == (2, 7)
    assert "context_vector" in features
    assert features["context_vector"].shape == (2, model.embedding_dim)
    assert "attention_weights" in features             # use_attention=True by default


def test_cnn_lstm_get_embedding():
    """get_embedding() returns a (B, embedding_dim) tensor."""
    model = CNNLSTMModel(input_channels=39, num_classes=7)
    x = torch.randn(3, 39, 128)
    emb = model.get_embedding(x)
    assert emb.shape == (3, model.embedding_dim)


def test_cnn_lstm_predict_proba_sums_to_one():
    """predict_proba() outputs valid probability distributions."""
    model = CNNLSTMModel(input_channels=39, num_classes=10)
    x = torch.randn(4, 39, 128)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 10)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


# -----------------------------------------------------------------------------
#  Audio Transformer
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("n_channels,num_classes,seq_len", [
    (39, 10, 431),
    (39, 99, 256),
    (13, 10, 128),
])
def test_audio_transformer_logits_shape(n_channels, num_classes, seq_len):
    """AudioTransformer default forward returns logits tensor only."""
    model = AudioTransformer(input_channels=n_channels, num_classes=num_classes)
    x = torch.randn(2, n_channels, seq_len)
    logits = model(x)
    assert logits.shape == (2, num_classes), (
        f"Expected logits shape (2, {num_classes}), got {logits.shape}"
    )


def test_audio_transformer_return_features():
    """return_features=True yields (logits, dict) with context_vector."""
    model = AudioTransformer(input_channels=39, num_classes=7)
    x = torch.randn(2, 39, 128)
    logits, features = model(x, return_features=True)

    assert logits.shape == (2, 7)
    assert "context_vector" in features
    assert features["context_vector"].shape == (2, model.d_model)
    assert "encoder_output" in features
    assert features["encoder_output"].shape[0] == 2   # (B, T, d_model)


def test_audio_transformer_same_api_as_cnn_lstm():
    """Both models accept (B, C, T) input and return (B, num_classes) logits.

    This is the core requirement for the fair ablation study in the dissertation.
    """
    B, C, T, K = 3, 39, 200, 10
    x = torch.randn(B, C, T)

    cnn_lstm = CNNLSTMModel(input_channels=C, num_classes=K)
    transformer = AudioTransformer(input_channels=C, num_classes=K)

    out_cnn = cnn_lstm(x)
    out_tf = transformer(x)

    assert out_cnn.shape == (B, K)
    assert out_tf.shape == (B, K)


# -----------------------------------------------------------------------------
#  PitchCNN and FusionModel
# -----------------------------------------------------------------------------

def test_pitch_cnn_forward_shape():
    model = PitchCNN(input_size=1, num_classes=7)
    x = torch.randn(2, 1, 128)
    logits = model(x)
    assert logits.shape == (2, 7)


def test_fusion_model_forward_shape():
    model = FusionModel(num_classes=7, acoustic_channels=39)
    x_acoustic = torch.randn(2, 39, 128)
    x_pitch = torch.randn(2, 1, 128)
    logits, features = model(x_acoustic, x_pitch)
    assert logits.shape == (2, 7)
    assert "fusion_embedding" in features
    assert features["fusion_embedding"].shape[0] == 2
