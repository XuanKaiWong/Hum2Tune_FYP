"""Prepare chunked train/val/test datasets without file leakage.

Key fixes in this version:
- split by source file *before* window extraction and augmentation
- apply augmentation to the training split only
- save incrementally in chunked `.npy` format to avoid very large writes
- write split manifests for reproducibility
"""

from __future__ import annotations

import io
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fyp_title11.tokenization.feature_extractor import FeatureExtractor

log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
_utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "prepare_dataset.log", encoding="utf-8"),
        logging.StreamHandler(stream=_utf8_stdout),
    ],
)
logger = logging.getLogger(__name__)

SEED = 42
NOISE_LEVEL = 0.005
PITCH_SHIFT_STEPS = [-3, -2, -1, 1, 2, 3]
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.m4a", "*.flac")


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    song_name: str
    label: int


class ChunkedDatasetWriter:
    """Incrementally writes dataset chunks to disk."""

    def __init__(self, base_path: Path, chunk_size: int = 1024):
        self.base_path = base_path
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self.buffer_x: list[np.ndarray] = []
        self.buffer_y: list[int] = []
        self.chunk_index = 0
        self.total = 0
        self._clear_existing()

    def _clear_existing(self) -> None:
        for old in self.base_path.parent.glob(f"{self.base_path.stem}_chunk*_*.npy"):
            old.unlink()
        meta_path = self.base_path.parent / f"{self.base_path.stem}_meta.json"
        if meta_path.exists():
            meta_path.unlink()

    def append(self, feature: np.ndarray, label: int) -> None:
        self.buffer_x.append(np.asarray(feature, dtype=np.float32))
        self.buffer_y.append(int(label))
        if len(self.buffer_x) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer_x:
            return

        x_arr = np.stack(self.buffer_x).astype(np.float32)
        y_arr = np.asarray(self.buffer_y, dtype=np.int64)

        x_path = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_X.npy"
        y_path = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_y.npy"

        # IMPORTANT: np.save() auto-appends .npy when the path does not already
        # end with .npy. Using *.tmp would silently create *.tmp.npy, which then
        # breaks the final move on Windows.
        tmp_x = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_X.tmp.npy"
        tmp_y = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_y.tmp.npy"

        np.save(tmp_x, x_arr)
        np.save(tmp_y, y_arr)

        tmp_x.replace(x_path)
        tmp_y.replace(y_path)

        logger.info("Saved %s chunk %d with %d samples", self.base_path.stem, self.chunk_index, len(x_arr))
        self.total += len(x_arr)
        self.chunk_index += 1
        self.buffer_x.clear()
        self.buffer_y.clear()

    def finalize(self, extra_meta: dict | None = None) -> None:
        self.flush()
        meta = {
            "stem": self.base_path.stem,
            "n_chunks": self.chunk_index,
            "total": self.total,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(self.base_path.parent / f"{self.base_path.stem}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def augment_audio(audio: np.ndarray, sr: int, rng: random.Random) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []
    noise = np.random.normal(0, NOISE_LEVEL, audio.shape).astype(np.float32)
    augmented.append((audio + noise).clip(-1.0, 1.0))
    pool = PITCH_SHIFT_STEPS.copy()
    rng.shuffle(pool)
    for n_steps in pool[:2]:
        try:
            augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps))
        except Exception:
            logger.warning("Pitch shift failed for n_steps=%s", n_steps)
    return augmented


def discover_audio_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in AUDIO_EXTENSIONS:
        files.extend(raw_dir.rglob(pattern))
    return sorted([p for p in files if p.is_file() and not p.name.startswith(".")])


def split_records_by_file(records: list[AudioRecord], seed: int = SEED) -> dict[str, list[AudioRecord]]:
    """Split at file level so windows from the same file never cross splits."""
    rng = random.Random(seed)
    grouped: dict[int, list[AudioRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.label].append(rec)

    splits = {"train": [], "val": [], "test": []}
    warnings = []
    for label, items in grouped.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        # For tiny per-song datasets, prefer keeping more files in training.
        # This is a pragmatic compromise for development experiments.
        if n >= 5:
            n_test, n_val = 1, 1
        elif n == 4:
            n_test, n_val = 1, 1
        elif n == 3:
            n_test, n_val = 0, 1
            warnings.append(f"Class {items[0].song_name!r} has only 3 files; no test example created.")
        elif n == 2:
            n_test, n_val = 0, 1
            warnings.append(f"Class {items[0].song_name!r} has only 2 files; no test example created.")
        else:
            n_test, n_val = 0, 0
            warnings.append(f"Class {items[0].song_name!r} has only 1 file; kept in train only.")

        train_cut = n - n_val - n_test
        splits["train"].extend(items[:train_cut])
        splits["val"].extend(items[train_cut:train_cut + n_val])
        splits["test"].extend(items[train_cut + n_val:])

    for msg in warnings:
        logger.warning(msg)
    return splits


def build_class_map(audio_files: list[Path]) -> tuple[dict[str, int], dict[str, str]]:
    unique_songs = sorted({f.parent.name for f in audio_files})
    class_to_idx = {name: idx for idx, name in enumerate(unique_songs)}
    idx_to_class = {str(idx): name for idx, name in enumerate(unique_songs)}
    return class_to_idx, idx_to_class


def make_manifest(records: list[AudioRecord]) -> dict:
    per_class_files: dict[str, int] = defaultdict(int)
    for rec in records:
        per_class_files[rec.song_name] += 1
    return {
        "num_files": len(records),
        "per_class_files": dict(sorted(per_class_files.items())),
        "files": [str(rec.path) for rec in records],
    }


def process_split(
    split_name: str,
    records: list[AudioRecord],
    extractor: FeatureExtractor,
    processed_dir: Path,
    seed: int = SEED,
) -> None:
    writer = ChunkedDatasetWriter(processed_dir / f"{split_name}_data")
    rng = random.Random(seed + hash(split_name) % 1000)
    skipped = 0

    for rec in tqdm(records, desc=f"Processing {split_name}", unit="file"):
        try:
            audio, _ = librosa.load(str(rec.path), sr=extractor.sr, mono=True)
            variants = [audio]
            if split_name == "train":
                variants.extend(augment_audio(audio, extractor.sr, rng))

            for variant in variants:
                feats = extractor.process_audio(variant)
                for feat in feats:
                    writer.append(feat, rec.label)
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping %s: %s", rec.path.name, exc)

    writer.finalize(extra_meta={
        "split": split_name,
        "num_source_files": len(records),
        "skipped_files": skipped,
    })
    logger.info("%s split complete: %d source files, %d skipped", split_name, len(records), skipped)


def prepare_data() -> None:
    data_root = project_root / "data"
    raw_dir = data_root / "Humming Audio"
    processed_dir = data_root / "processed" / "datasets"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for old in processed_dir.glob("*"):
        if old.is_file() and (old.suffix in {".npz", ".npy", ".json"} or old.name.endswith(".tmp")):
            old.unlink()

    extractor = FeatureExtractor(config_path=str(project_root / "config" / "audio_config.yaml"))
    audio_files = discover_audio_files(raw_dir)
    if not audio_files:
        logger.error("No audio files found in %s", raw_dir)
        return

    class_to_idx, idx_to_class = build_class_map(audio_files)
    records = [AudioRecord(path=f, song_name=f.parent.name, label=class_to_idx[f.parent.name]) for f in audio_files]
    splits = split_records_by_file(records)

    with open(processed_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    split_manifest = {name: make_manifest(split_records) for name, split_records in splits.items()}
    with open(processed_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info("Found %d audio files across %d song classes", len(audio_files), len(class_to_idx))
    for split_name, split_records in splits.items():
        logger.info("%s files: %d", split_name, len(split_records))
        process_split(split_name, split_records, extractor, processed_dir)

    logger.info("Dataset ready in %s", processed_dir)


if __name__ == "__main__":
    prepare_data()
