"""Hum2Tune — Melody-based music recognition from hummed input.

This package provides the core components for the Hum2Tune FYP system:
  - data:         dataset loading and chunked storage utilities
  - evaluation:   retrieval and classification metrics (Top-K, MRR, MAP, NDCG)
  - models:       CNNLSTMModel, AudioTransformer, PitchCNN, FusionModel
  - tokenization: FeatureExtractor (39-channel MFCC+delta), PitchDetector
  - utils:        config loader, visualisation helpers

Submodules are imported lazily (on demand) rather than eagerly here.
Eager top-level imports from subpackages with heavy dependencies (torch,
librosa) cause unnecessary import-time overhead and can create circular
import chains when any submodule imports from this parent package.

Usage:
    from fyp_title11.models import CNNLSTMModel, AudioTransformer
    from fyp_title11.tokenization.feature_extractor import FeatureExtractor
    from fyp_title11.evaluation.metrics import compute_all_metrics
"""

__version__ = "1.1.0"
__author__ = "Wong Xuan Kai"
__description__ = "Hum2Tune: Melody-Based Music Recognition Using Frequency Analysis"

# Declare subpackages — no eager imports so submodules load only when needed.
__all__ = [
    "data",
    "evaluation",
    "models",
    "tokenization",
    "utils",
]
