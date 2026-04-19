"""Prediction script for Hum2Tune (CNN-LSTM only, small-data aligned)."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fyp_title11.data.dataset import load_class_map
from fyp_title11.models.cnn_lstm import CNNLSTMModel
from fyp_title11.models.audio_transformer import AudioTransformer
from fyp_title11.tokenization.feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

_utf8_stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "predict.log", encoding="utf-8"),
        logging.StreamHandler(stream=_utf8_stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Defaults aligned with the current small-data training setup
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "cnn_lstm"
DEFAULT_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.30


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def load_training_config() -> dict[str, Any]:
    """
    Load config/training_config.yaml if available.
    Fall back to the same small-data defaults used by training.
    """
    config_path = project_root / "config" / "training_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "training" in loaded and isinstance(loaded["training"], dict):
                return loaded["training"]

    return {
        "input_channels": 39,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.4,
        "bidirectional": True,
        "use_attention": True,
        "norm_type": "group",
    }


def build_model(model_name: str, num_classes: int, train_conf: dict[str, Any]):
    """Instantiate the correct architecture based on model_name."""
    if model_name == "audio_transformer":
        t = train_conf.get("transformer", {})
        return AudioTransformer(
            input_channels=int(train_conf.get("input_channels", 39)),
            num_classes=num_classes,
            d_model=int(t.get("d_model", 128)),
            nhead=int(t.get("nhead", 4)),
            num_layers=int(t.get("num_layers", 2)),
            dim_feedforward=int(t.get("dim_feedforward", 256)),
            dropout=float(t.get("dropout", 0.3)),
        )
    # Default: cnn_lstm
    return CNNLSTMModel(
        input_channels=int(train_conf.get("input_channels", 39)),
        num_classes=num_classes,
        hidden_size=int(train_conf.get("hidden_size", 128)),
        num_layers=int(train_conf.get("num_layers", 1)),
        dropout=float(train_conf.get("dropout", 0.35)),
        bidirectional=bool(train_conf.get("bidirectional", True)),
        use_attention=bool(train_conf.get("use_attention", True)),
        attn_hidden_dim=int(train_conf.get("attn_hidden_dim", 32)),
        norm_type=str(train_conf.get("norm_type", "group")),
    )


def load_class_names() -> list[str]:
    classes_path = project_root / "data" / "processed" / "datasets" / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"classes.json not found: {classes_path}. "
            "Run dataset preparation first."
        )

    class_map = load_class_map(classes_path)
    return [class_map[str(i)] for i in range(len(class_map))]


def prepare_features(
    audio_path: str,
    expected_channels: int,
) -> np.ndarray:
    """
    Load audio, extract features, and coerce them into shape (C, T).
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    extractor = FeatureExtractor(
        config_path=str(project_root / "config" / "audio_config.yaml")
    )

    audio, _ = librosa.load(
        str(audio_file),
        sr=extractor.sr,
        duration=extractor.duration,
        mono=True,
    )

    features = extractor.process_segment(audio)
    if features is None:
        raise RuntimeError(f"Feature extraction failed for {audio_file}")

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features shaped like (C, T) or (T, C), got {features.shape}"
        )

    # Try to auto-correct orientation if needed
    if features.shape[0] != expected_channels and features.shape[1] == expected_channels:
        features = features.T

    if features.shape[0] != expected_channels:
        raise ValueError(
            f"Feature channel mismatch: model expects {expected_channels}, "
            f"but extracted features have shape {features.shape}. "
            "Check audio_config.yaml and training_config.yaml."
        )

    return features


def format_results(
    class_names: list[str],
    probs: torch.Tensor,
    top_k: int,
    confidence_threshold: float,
) -> None:
    top_k = min(top_k, len(class_names))
    top_prob, top_idx = torch.topk(probs, top_k)

    best_confidence = float(top_prob[0, 0].item())
    best_index = int(top_idx[0, 0].item())
    best_name = class_names[best_index]

    print("\nMATCH RESULTS")
    print("-" * 50)

    if best_confidence < confidence_threshold:
        print(
            f"Low confidence ({best_confidence:.1%}). "
            f"Best guess: {best_name}"
        )
        return

    for rank in range(top_k):
        idx = int(top_idx[0, rank].item())
        name = class_names[idx]
        confidence = float(top_prob[0, rank].item())
        print(f"{rank + 1}. {name:<30} {confidence:.1%}")


# ---------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------

def predict(
    audio_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    supported = ("cnn_lstm", "audio_transformer")
    if model_name not in supported:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {supported}"
        )

    train_conf = load_training_config()
    expected_channels = int(train_conf.get("input_channels", 39))
    class_names = load_class_names()

    features = prepare_features(
        audio_path=audio_path,
        expected_channels=expected_channels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running prediction on device: %s", device)

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        train_conf=train_conf,
    ).to(device)

    model_path = project_root / "models" / model_name / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            "Train the model first."
        )

    state_dict = torch_load(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = extract_logits(model(tensor))
        probs = torch.softmax(logits, dim=1)

    format_results(
        class_names=class_names,
        probs=probs,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict song from a hummed audio recording."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the audio file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=[DEFAULT_MODEL_NAME],
        help="Model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top predictions to show",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence required before accepting the top result",
    )

    args = parser.parse_args()

    predict(
        audio_path=args.audio,
        model_name=args.model,
        top_k=args.top_k,
        confidence_threshold=args.threshold,
    )