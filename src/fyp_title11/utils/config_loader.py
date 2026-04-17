"""
Configuration loader for Hum2Tune.

Loads and manages project configuration files with sensible defaults for the
cleaned final FYP scope:
- CNN-LSTM as the active neural model
- DTW as the traditional baseline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Configuration loader for the Hum2Tune project."""

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        """
        Load all configuration files.

        Notes:
        - The project uses `path.yaml`, not `paths.yaml`.
        - If a config file is missing, a default one is created automatically.
        """
        config_files = [
            ("main", "config.yaml"),
            ("model", "model_config.yaml"),
            ("audio", "audio_config.yaml"),
            ("training", "training_config.yaml"),
            ("paths", "path.yaml"),
        ]

        for name, filename in config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.configs[name] = self.load_yaml(config_path)
            else:
                self.configs[name] = self._get_default_config(name)
                self.save_config(name, config_path)

    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Return default configuration for a named config section."""
        defaults: Dict[str, Dict[str, Any]] = {
            "main": {
                "project": {
                    "name": "Hum2Tune",
                    "version": "1.1.0",
                    "author": "FYP Student",
                    "description": "Humming-based music recognition",
                },
                "audio": {
                    "sample_rate": 22050,
                    "duration": 30.0,
                    "n_mels": 128,
                    "n_mfcc": 13,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "min_freq": 80,
                    "max_freq": 8000,
                },
                "models": {
                    "cnn_lstm": {
                        "input_channels": 13,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "dropout": 0.4,
                        "bidirectional": True,
                        "use_attention": True,
                        "norm_type": "group",
                    },
                    "dtw": {
                        "distance_metric": "euclidean",
                        "top_k": 5,
                    },
                },
                "training": {
                    "batch_size": 4,
                    "learning_rate": 3e-4,
                    "epochs": 80,
                    "early_stopping_patience": 10,
                    "gradient_clip": 1.0,
                    "weight_decay": 1e-3,
                    "scheduler_patience": 4,
                    "scheduler_factor": 0.5,
                    "min_lr": 1e-5,
                    "seed": 42,
                },
                "evaluation": {
                    "metrics": [
                        "accuracy",
                        "top_k_accuracy",
                        "mrr",
                        "macro_f1",
                        "weighted_f1",
                    ],
                    "top_k": 3,
                },
            },
            "model": {
                "cnn_lstm": {
                    "input_channels": 13,
                    "num_classes": 10,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "dropout": 0.4,
                    "bidirectional": True,
                    "use_attention": True,
                    "norm_type": "group",
                },
                "pitch_cnn": {
                    "input_size": 1,
                    "num_classes": 10,
                    "dropout": 0.4,
                    "norm_type": "group",
                },
                "fusion": {
                    "num_classes": 10,
                    "acoustic_channels": 13,
                    "hidden_size": 64,
                    "dropout": 0.4,
                    "norm_type": "group",
                },
                "dtw": {
                    "window_size": None,
                    "distance_metric": "euclidean",
                    "top_k": 5,
                },
            },
            "audio": {
                "sample_rate": 22050,
                "duration": 30.0,
                "n_mels": 128,
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 512,
                "min_freq": 80.0,
                "max_freq": 8000.0,
            },
            "training": {
                "training": {
                    "batch_size": 4,
                    "learning_rate": 3e-4,
                    "epochs": 80,
                    "early_stopping_patience": 10,
                    "gradient_clip": 1.0,
                    "weight_decay": 1e-3,
                    "input_channels": 13,
                    "seed": 42,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "dropout": 0.4,
                    "bidirectional": True,
                    "use_attention": True,
                    "norm_type": "group",
                    "scheduler_patience": 4,
                    "scheduler_factor": 0.5,
                    "min_lr": 1e-5,
                    "label_smoothing": 0.1,
                    "num_workers": 0,
                }
            },
            "paths": {
                "paths": {
                    "data": {
                        "raw": "data/raw",
                        "processed": "data/processed",
                        "hummed": "data/hummed",
                    },
                    "raw": {
                        "songs": "data/raw/original_songs",
                        "metadata": "data/raw/metadata",
                    },
                    "hummed": {
                        "recordings": "data/hummed/recordings",
                        "metadata": "data/hummed/metadata",
                    },
                    "processed": {
                        "features": "data/processed/features",
                        "tokens": "data/processed/tokens",
                        "datasets": "data/processed/datasets",
                    },
                    "manifests": {
                        "root": "data/manifests",
                    },
                    "queries": {
                        "humming": "data/humming_queries",
                        "references": "data/song_references",
                    },
                    "models": {
                        "cnn_lstm": "models/cnn_lstm",
                        "dtw": "models/dtw",
                        "checkpoints": "models/checkpoints",
                    },
                    "logs": "logs",
                    "results": "results",
                    "visualizations": "results/visualizations",
                    "predictions": "results/predictions",
                    "evaluations": "results/evaluations",
                    "assets": {
                        "root": "assets",
                    },
                    "config": "config",
                    "src": "src",
                    "scripts": "scripts",
                    "tests": "tests",
                }
            },
        }

        return defaults.get(config_name, {})

    def load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"Error loading YAML file {filepath}: {exc}")
            return {}

    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"Error loading JSON file {filepath}: {exc}")
            return {}

    def save_config(self, config_name: str, filepath: Path) -> None:
        """Save configuration to file."""
        config = self.configs.get(config_name, {})
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix in {".yaml", ".yml"}:
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        elif filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def get_config(self, config_name: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value.

        Supports nested access using dot notation.
        Example:
            get_config("paths", "paths.models.cnn_lstm")
        """
        config = self.configs.get(config_name, {})

        if key is None:
            return config

        value: Any = config
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
        return value

    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Deep-update a configuration section."""
        if config_name not in self.configs:
            self.configs[config_name] = {}

        def deep_update(current: Dict[str, Any], incoming: Dict[str, Any]) -> None:
            for key, value in incoming.items():
                if (
                    key in current
                    and isinstance(current[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(current[key], value)
                else:
                    current[key] = value

        deep_update(self.configs[config_name], updates)

    def get_audio_config(self) -> Dict[str, Any]:
        return self.get_config("audio")

    def get_model_config(self, model_type: str = "cnn_lstm") -> Dict[str, Any]:
        model_configs = self.get_config("model")
        return model_configs.get(model_type, {}) if isinstance(model_configs, dict) else {}

    def get_training_config(self) -> Dict[str, Any]:
        training = self.get_config("training")
        if isinstance(training, dict) and "training" in training and isinstance(training["training"], dict):
            return training["training"]
        return training if isinstance(training, dict) else {}

    def get_paths(self) -> Dict[str, Any]:
        paths = self.get_config("paths")
        if isinstance(paths, dict) and "paths" in paths and isinstance(paths["paths"], dict):
            return paths["paths"]
        return paths if isinstance(paths, dict) else {}

    def create_project_structure(self) -> None:
        """Create project directory structure from path configuration."""
        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, str) and value.strip():
                path = Path(value)
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {path}")

        walk(self.get_paths())

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten all config sections into a single dot-key dictionary."""
        flat_config: Dict[str, Any] = {}

        def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            items: Dict[str, Any] = {}
            for key, value in d.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    items.update(flatten_dict(value, new_key))
                else:
                    items[new_key] = value
            return items

        for config_name, config in self.configs.items():
            flat_config.update(flatten_dict(config, config_name))

        return flat_config

    def print_config(self, config_name: Optional[str] = None) -> None:
        """Pretty-print one config section or all configs."""
        if config_name:
            config = self.get_config(config_name)
            print(f"\n📋 Configuration: {config_name}")
            print(json.dumps(config, indent=2, default=str))
            return

        for name, config in self.configs.items():
            print(f"\n📋 Configuration: {name}")
            print(json.dumps(config, indent=2, default=str))