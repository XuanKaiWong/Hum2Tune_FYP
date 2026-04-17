import torch

from src.fyp_title11.models.cnn_lstm import CNNLSTMModel, PitchCNN, FusionModel


def test_cnn_lstm_forward_shape():
    model = CNNLSTMModel(input_channels=13, num_classes=7)
    x = torch.randn(2, 13, 128)
    logits, features = model(x)

    assert logits.shape == (2, 7)
    assert "context_vector" in features
    assert features["context_vector"].shape[0] == 2


def test_pitch_cnn_forward_shape():
    model = PitchCNN(input_size=1, num_classes=7)
    x = torch.randn(2, 1, 128)
    logits = model(x)

    assert logits.shape == (2, 7)


def test_fusion_model_forward_shape():
    model = FusionModel(num_classes=7, acoustic_channels=13)
    x_acoustic = torch.randn(2, 13, 128)
    x_pitch = torch.randn(2, 1, 128)

    logits, features = model(x_acoustic, x_pitch)

    assert logits.shape == (2, 7)
    assert "fusion_embedding" in features
    assert features["fusion_embedding"].shape[0] == 2