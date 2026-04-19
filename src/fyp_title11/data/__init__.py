"""Data loading utilities for Hum2Tune.

Exports:
    RealAudioDataset:   Load features from a single .npz file.
    ChunkedAudioDataset: Lazy-loading dataset backed by chunked .npy files.
    resolve_split_dataset: Auto-detect and load either format by split name.
    load_class_map:     Load the class index -> song name mapping.
"""

from .dataset import (
    ChunkedAudioDataset,
    RealAudioDataset,
    load_class_map,
    resolve_split_dataset,
)

__all__ = [
    "ChunkedAudioDataset",
    "RealAudioDataset",
    "load_class_map",
    "resolve_split_dataset",
]
