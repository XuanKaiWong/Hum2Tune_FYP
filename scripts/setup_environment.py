"""Project setup helpers for Hum2Tune."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

import pandas as pd


CONFIG_YAML = """# Hum2Tune Configuration

project:
  name: "Hum2Tune"
  version: "1.1.0"
  author: "FYP Student"
  description: "Humming-based music recognition with DTW baseline and CNN-LSTM model"

audio:
  sample_rate: 22050
  duration: 30.0
  n_mels: 128
  n_mfcc: 13
  n_fft: 2048
  hop_length: 512
  min_freq: 80
  max_freq: 8000

models:
  cnn_lstm:
    input_channels: 39
    hidden_size: 64
    num_layers: 1
    dropout: 0.4
    bidirectional: true
    use_attention: true
    norm_type: "group"

  dtw:
    distance_metric: "euclidean"
    top_k: 5

training:
  batch_size: 4
  learning_rate: 0.0003
  epochs: 80
  early_stopping_patience: 10
  gradient_clip: 1.0
  weight_decay: 0.001
  scheduler_patience: 4
  scheduler_factor: 0.5
  min_lr: 0.00001
  seed: 42

evaluation:
  metrics: ["accuracy", "top_k_accuracy", "mrr", "macro_f1", "weighted_f1"]
  top_k: 3
"""

MODEL_CONFIG_YAML = """# Model configurations

cnn_lstm:
  input_channels: 39
  num_classes: 10
  hidden_size: 64
  num_layers: 1
  dropout: 0.4
  bidirectional: true
  use_attention: true
  norm_type: "group"

pitch_cnn:
  input_size: 1
  num_classes: 10
  dropout: 0.4
  norm_type: "group"

fusion:
  num_classes: 10
  acoustic_channels: 39
  hidden_size: 64
  dropout: 0.4
  norm_type: "group"

dtw:
  window_size: null
  distance_metric: "euclidean"
  top_k: 5
"""

PATH_YAML = """paths:
  data:
    raw: "data/raw"
    processed: "data/processed"
    hummed: "data/hummed"

  raw:
    songs: "data/raw/original_songs"
    metadata: "data/raw/metadata"

  hummed:
    recordings: "data/hummed/recordings"
    metadata: "data/hummed/metadata"

  processed:
    features: "data/processed/features"
    tokens: "data/processed/tokens"
    datasets: "data/processed/datasets"

  manifests:
    root: "data/manifests"

  queries:
    humming: "data/humming_queries"
    references: "data/song_references"

  models:
    cnn_lstm: "models/cnn_lstm"
    dtw: "models/dtw"
    checkpoints: "models/checkpoints"

  logs: "logs"
  results: "results"
  visualizations: "results/visualizations"
  predictions: "results/predictions"
  evaluations: "results/evaluations"

  assets:
    root: "assets"

  config: "config"
  src: "src"
  scripts: "scripts"
  tests: "tests"
"""

AUDIO_CONFIG_YAML = """audio:
  sample_rate: 22050
  duration: 30.0
  n_mfcc: 13
  n_mels: 128
  n_fft: 2048
  hop_length: 512
  win_length: 2048
  min_freq: 80.0
  max_freq: 8000.0
  normalize_audio: true
  normalize_features: true
"""

TRAINING_CONFIG_YAML = """training:
  batch_size: 4
  learning_rate: 3e-4
  epochs: 80
  early_stopping_patience: 10
  gradient_clip: 1.0
  weight_decay: 1e-3
  input_channels: 39
  seed: 42
  hidden_size: 64
  num_layers: 1
  dropout: 0.4
  bidirectional: true
  use_attention: true
  norm_type: "group"
  scheduler_patience: 4
  scheduler_factor: 0.5
  min_lr: 1e-5
  label_smoothing: 0.1
  num_workers: 0
"""

ENV_EXAMPLE = """# Hum2Tune Environment Variables

PROJECT_NAME=Hum2Tune
PROJECT_VERSION=1.1.0
DATA_PATH=./data
MODELS_PATH=./models
LOGS_PATH=./logs
RESULTS_PATH=./results
SAMPLE_RATE=22050
MAX_DURATION=30
DEFAULT_MODEL=cnn_lstm
BATCH_SIZE=4
LEARNING_RATE=0.0003
LOG_LEVEL=INFO
LOG_FILE=./logs/hum2tune.log
"""


def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  [OK] {path.relative_to(path.parent.parent if path.parent.parent.exists() else path.parent)} created")
    else:
        print(f"  [OK] {path.name} already exists")


def setup_project() -> None:
    print("\n" + "=" * 70)
    print("[Hum2Tune] HUM2TUNE - PROJECT SETUP")
    print("=" * 70)

    project_root = Path(__file__).parent.parent

    print(f"\n[Folder] Project Root: {project_root}")
    print(f"[Python] Python Version: {platform.python_version()}")
    print(f"[OS] OS: {platform.system()} {platform.release()}")

    print("\n[1/5] Creating directory structure...")
    directories = [
        "data/raw/original_songs",
        "data/raw/metadata",
        "data/hummed/recordings",
        "data/hummed/metadata",
        "data/humming_queries",
        "data/song_references",
        "data/manifests",
        "data/processed/features",
        "data/processed/tokens",
        "data/processed/datasets",
        "models/cnn_lstm",
        "models/dtw",
        "models/checkpoints",
        "logs",
        "results/evaluations",
        "results/visualizations",
        "results/predictions",
        "config",
        "assets",
    ]
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {directory}")

    print("\n[2/5] Checking Python dependencies...")
    required_packages = {
        "torch": "torch",
        "librosa": "librosa",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "yaml": "pyyaml",
    }
    missing = []
    for import_name, display_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  [OK] {display_name}")
        except ImportError:
            missing.append(display_name)
            print(f"  [X] {display_name} (missing)")

    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")

    print("\n[3/5] Creating configuration files...")
    write_if_missing(project_root / "config" / "config.yaml", CONFIG_YAML)
    write_if_missing(project_root / "config" / "model_config.yaml", MODEL_CONFIG_YAML)
    write_if_missing(project_root / "config" / "path.yaml", PATH_YAML)
    write_if_missing(project_root / "config" / "audio_config.yaml", AUDIO_CONFIG_YAML)
    write_if_missing(project_root / "config" / "training_config.yaml", TRAINING_CONFIG_YAML)

    print("\n[4/5] Creating sample metadata...")
    songs_df = pd.DataFrame(
        {
            "song_id": ["song_001", "song_002", "song_003"],
            "title": ["Melody One", "Happy Tune", "Sad Song"],
            "artist": ["Artist A", "Artist B", "Artist C"],
            "duration": [180, 210, 195],
            "bpm": [120, 140, 80],
            "key": ["C", "G", "D"],
            "genre": ["Pop", "Rock", "Jazz"],
        }
    )
    songs_df.to_csv(project_root / "data/raw/metadata/songs_metadata.csv", index=False)

    hummed_df = pd.DataFrame(
        {
            "hum_id": ["hum_001_01", "hum_001_02", "hum_002_01"],
            "song_id": ["song_001", "song_001", "song_002"],
            "user_id": ["user_01", "user_02", "user_01"],
            "duration": [10.5, 9.8, 11.2],
            "pitch_accuracy": [0.85, 0.78, 0.92],
            "rhythm_accuracy": [0.80, 0.75, 0.88],
        }
    )
    hummed_df.to_csv(project_root / "data/hummed/metadata/hummed_metadata.csv", index=False)
    print("  [OK] Sample metadata created")

    print("\n[5/5] Creating environment file...")
    env_example = project_root / ".env.example"
    if not env_example.exists():
        env_example.write_text(ENV_EXAMPLE, encoding="utf-8")
        print("  [OK] .env.example created")
    else:
        print("  [OK] .env.example already exists")

    actual_env = project_root / ".env"
    if not actual_env.exists():
        shutil.copy(env_example, actual_env)
        print("  [OK] .env created (copy of .env.example)")
    else:
        print("  [OK] .env already exists")

    print("\n" + "=" * 70)
    print("[OK] SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n[Report] Next Steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Create dataset / features as needed")
    print("  3. Train model: python scripts/train_model.py")
    print("  4. Predict: python scripts/predict.py --audio your_file.wav")
    print("  5. Evaluate: python scripts/evaluate.py")
    print("\n[Hum2Tune] Happy coding! [Hum2Tune]")


if __name__ == "__main__":
    try:
        setup_project()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERR] Setup failed: {exc}")
        sys.exit(1)