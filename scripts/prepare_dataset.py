from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fyp_title11.tokenization.feature_extractor import FeatureExtractor  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_UTF8_STDOUT = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "prepare_dataset.log", encoding="utf-8"),
        logging.StreamHandler(stream=_UTF8_STDOUT),
    ],
)
LOGGER = logging.getLogger("prepare_dataset")

SEED = 42
NOISE_LEVEL = 0.005
PITCH_SHIFT_STEPS = [-3, -2, -1, 1, 2, 3]
TIME_STRETCH_RATES = [0.85, 0.90, 1.10, 1.15]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MIN_FILES_PER_CLASS = 5
BENCHMARK_SONGS: list[str] = []


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    song_name: str
    label: int
    source: str = "unknown"
    singer_id: str = "unknown"
    split: str | None = None
    query_id: str | None = None


class ChunkedDatasetWriter:
    """Incrementally write feature tensors as memory-safe chunks."""

    def __init__(self, base_path: Path, chunk_size: int = 1024) -> None:
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
        meta = self.base_path.parent / f"{self.base_path.stem}_meta.json"
        if meta.exists():
            meta.unlink()

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
        tmp_x = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_X.tmp.npy"
        tmp_y = self.base_path.parent / f"{self.base_path.stem}_chunk{self.chunk_index}_y.tmp.npy"

        np.save(tmp_x, x_arr)
        np.save(tmp_y, y_arr)
        tmp_x.replace(x_path)
        tmp_y.replace(y_path)

        LOGGER.info("Saved %s chunk %d with %d samples", self.base_path.stem, self.chunk_index, len(x_arr))
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
            json.dump(meta, f, indent=2, ensure_ascii=False)


def stable_split_seed(split_name: str, seed: int = SEED) -> int:
    digest = hashlib.sha1(f"{split_name}|{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else (root / path).resolve()


def discover_audio_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith(".")
    )


def build_class_map(song_names: Iterable[str]) -> tuple[dict[str, int], dict[str, str]]:
    unique = sorted(set(str(s) for s in song_names))
    class_to_idx = {name: idx for idx, name in enumerate(unique)}
    idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_manifest_records(manifest_dir: Path) -> tuple[list[AudioRecord], dict[str, int], dict[str, str]] | None:
    """Load records from songs.csv + queries.csv + optional splits.csv.

    Returns None when the required manifest files are not available, allowing
    fallback to directory scanning.
    """
    songs_path = manifest_dir / "songs.csv"
    queries_path = manifest_dir / "queries.csv"
    splits_path = manifest_dir / "splits.csv"

    if not songs_path.exists() or not queries_path.exists():
        return None

    songs = pd.read_csv(songs_path, dtype=str).fillna("")
    queries = pd.read_csv(queries_path, dtype=str).fillna("")

    required_songs = {"song_id", "title"}
    required_queries = {"query_id", "song_id", "query_path"}
    if not required_songs.issubset(songs.columns):
        raise ValueError(f"songs.csv must contain columns: {sorted(required_songs)}")
    if not required_queries.issubset(queries.columns):
        raise ValueError(f"queries.csv must contain columns: {sorted(required_queries)}")

    if "split" not in queries.columns:
        if splits_path.exists():
            splits = pd.read_csv(splits_path, dtype=str).fillna("")
            if not {"query_id", "split"}.issubset(splits.columns):
                raise ValueError("splits.csv must contain query_id and split columns")
            queries = queries.merge(splits[["query_id", "split"]], on="query_id", how="left")
        else:
            queries["split"] = ""

    songs = songs.drop_duplicates("song_id")
    merged = queries.merge(songs[["song_id", "title"]], on="song_id", how="left", validate="many_to_one")
    if merged["title"].isna().any() or (merged["title"] == "").any():
        missing = merged.loc[(merged["title"].isna()) | (merged["title"] == ""), "song_id"].unique().tolist()
        raise ValueError(f"queries.csv contains song_id values not found in songs.csv: {missing}")

    class_to_idx, idx_to_class = build_class_map(merged["title"].tolist())

    records: list[AudioRecord] = []
    skipped = 0
    for row in merged.itertuples(index=False):
        path = resolve_path(str(row.query_path))
        if not path.exists():
            LOGGER.warning("Manifest query path not found: %s", path)
            skipped += 1
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            LOGGER.warning("Unsupported audio extension in manifest: %s", path)
            skipped += 1
            continue

        split = str(getattr(row, "split", "") or "").strip().lower() or None
        if split is not None and split not in {"train", "val", "test"}:
            LOGGER.warning("Invalid split '%s' for %s; auto-splitting this record", split, row.query_id)
            split = None

        title = str(row.title)
        records.append(
            AudioRecord(
                path=path,
                song_name=title,
                label=class_to_idx[title],
                source=str(getattr(row, "source", "unknown") or "unknown"),
                singer_id=str(getattr(row, "singer_id", "unknown") or "unknown"),
                split=split,
                query_id=str(row.query_id),
            )
        )

    LOGGER.info(
        "Loaded manifest records: %d usable, %d skipped, %d classes",
        len(records),
        skipped,
        len(class_to_idx),
    )
    return records, class_to_idx, idx_to_class


def records_from_directory(raw_dir: Path) -> tuple[list[AudioRecord], dict[str, int], dict[str, str]]:
    files = discover_audio_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No audio files found under {raw_dir}. Add files or create data/manifests/*.csv."
        )

    class_to_idx, idx_to_class = build_class_map(p.parent.name for p in files)
    records: list[AudioRecord] = []
    for path in files:
        parts = {p.lower() for p in path.parts}
        if "user" in parts or "human" in parts:
            source = "user"
        elif "kaggle" in parts or "dataset" in parts:
            source = "kaggle"
        elif "ai" in parts or "synthetic" in parts or "generated" in parts:
            source = "ai"
        else:
            source = "unknown"

        records.append(
            AudioRecord(
                path=path,
                song_name=path.parent.name,
                label=class_to_idx[path.parent.name],
                source=source,
            )
        )

    LOGGER.info("Directory scan records: %d recordings, %d classes", len(records), len(class_to_idx))
    return records, class_to_idx, idx_to_class


def filter_records_for_classifier(
    records: list[AudioRecord],
    *,
    min_files: int,
    benchmark_songs: list[str] | None = None,
) -> list[AudioRecord]:
    counts = Counter(r.song_name for r in records)
    allowed = {name for name, count in counts.items() if count >= min_files}
    if benchmark_songs:
        allowed &= set(benchmark_songs)

    filtered = [r for r in records if r.song_name in allowed]
    dropped = sorted((name, count) for name, count in counts.items() if name not in allowed)

    LOGGER.info(
        "classifier_subset: kept %d/%d records across %d songs; dropped %d songs",
        len(filtered),
        len(records),
        len(set(r.song_name for r in filtered)),
        len(dropped),
    )
    for name, count in dropped[:20]:
        LOGGER.info("  dropped: %s (%d recordings)", name, count)
    if len(dropped) > 20:
        LOGGER.info("  ... %d more dropped songs", len(dropped) - 20)

    return filtered


def relabel_records(records: list[AudioRecord]) -> tuple[list[AudioRecord], dict[str, int], dict[str, str]]:
    class_to_idx, idx_to_class = build_class_map(r.song_name for r in records)
    relabelled = [
        AudioRecord(
            path=r.path,
            song_name=r.song_name,
            label=class_to_idx[r.song_name],
            source=r.source,
            singer_id=r.singer_id,
            split=r.split,
            query_id=r.query_id,
        )
        for r in records
    ]
    return relabelled, class_to_idx, idx_to_class


def split_records(records: list[AudioRecord], seed: int = SEED) -> dict[str, list[AudioRecord]]:
    """Split records by manifest where available, otherwise by file per class."""
    if records and all(r.split in {"train", "val", "test"} for r in records):
        LOGGER.info("Using manifest-defined splits.")
        splits = {"train": [], "val": [], "test": []}
        for r in records:
            splits[r.split or "train"].append(r)
        return splits

    LOGGER.info("Using deterministic file-level auto split.")
    rng = random.Random(seed)
    grouped: dict[int, list[AudioRecord]] = defaultdict(list)
    for r in records:
        grouped[r.label].append(r)

    splits = {"train": [], "val": [], "test": []}
    for _, items in sorted(grouped.items()):
        items = items.copy()
        rng.shuffle(items)
        n = len(items)

        if n >= 5:
            n_val, n_test = 1, 1
        elif n >= 3:
            n_val, n_test = 1, 0
        elif n == 2:
            n_val, n_test = 1, 0
        else:
            n_val, n_test = 0, 0
            LOGGER.warning("Only one recording for '%s'; keeping it in train only.", items[0].song_name)

        train_cut = n - n_val - n_test
        splits["train"].extend(items[:train_cut])
        splits["val"].extend(items[train_cut:train_cut + n_val])
        splits["test"].extend(items[train_cut + n_val:])

    return splits


def augment_audio(audio: np.ndarray, sr: int, rng: random.Random) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []

    noise_seed = rng.randrange(0, 2**32 - 1)
    noise_arr = np.random.default_rng(noise_seed).normal(
        0.0, NOISE_LEVEL, size=audio.shape
    ).astype(np.float32)
    augmented.append(np.clip(audio + noise_arr, -1.0, 1.0).astype(np.float32))

    pitch_steps = PITCH_SHIFT_STEPS.copy()
    rng.shuffle(pitch_steps)
    for n_steps in pitch_steps[:2]:
        try:
            augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32))
        except Exception as exc:
            LOGGER.warning("Pitch shift failed (%s semitones): %s", n_steps, exc)

    rates = TIME_STRETCH_RATES.copy()
    rng.shuffle(rates)
    for rate in rates[:2]:
        try:
            augmented.append(librosa.effects.time_stretch(audio, rate=rate).astype(np.float32))
        except Exception as exc:
            LOGGER.warning("Time stretch failed (rate %.2f): %s", rate, exc)

    return augmented


def process_split(
    split_name: str,
    records: list[AudioRecord],
    extractor: FeatureExtractor,
    processed_dir: Path,
    *,
    seed: int = SEED,
) -> None:
    writer = ChunkedDatasetWriter(processed_dir / f"{split_name}_data")
    rng = random.Random(stable_split_seed(split_name, seed))
    skipped = 0
    source_files = 0

    for record in tqdm(records, desc=f"Processing {split_name}", unit="file"):
        try:
            audio, _ = librosa.load(str(record.path), sr=extractor.sr, mono=True)
            source_files += 1
            variants = [audio.astype(np.float32)]
            if split_name == "train":
                variants.extend(augment_audio(audio.astype(np.float32), extractor.sr, rng))

            for variant in variants:
                for feature in extractor.process_audio(variant):
                    writer.append(feature, record.label)
        except Exception as exc:
            skipped += 1
            LOGGER.warning("Skipping %s: %s", record.path, exc)

    writer.finalize(
        extra_meta={
            "split": split_name,
            "num_source_files": source_files,
            "skipped_files": skipped,
            "feature_channels": extractor.input_channels,
            "target_len": extractor.target_len,
        }
    )
    LOGGER.info(
        "%s complete: %d source files, %d skipped, %d windows",
        split_name,
        source_files,
        skipped,
        writer.total,
    )


def write_split_manifest(
    processed_dir: Path,
    splits: dict[str, list[AudioRecord]],
    idx_to_class: dict[str, str],
) -> None:
    split_manifest: dict[str, dict] = {
        "_classes": idx_to_class,
    }

    for split_name, records in splits.items():
        per_class = Counter(r.song_name for r in records)
        per_source = Counter(r.source for r in records)
        split_manifest[split_name] = {
            "num_files": len(records),
            "per_class_files": dict(sorted(per_class.items())),
            "sources": dict(sorted(per_source.items())),
            "files": [
                {
                    "query_id": r.query_id,
                    "path": str(r.path),
                    "song_name": r.song_name,
                    "label": r.label,
                    "source": r.source,
                    "singer_id": r.singer_id,
                }
                for r in records
            ],
        }

    with open(processed_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)


def clear_processed_dir(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for old in processed_dir.glob("*"):
        if old.is_file() and (
            old.suffix in {".npz", ".npy", ".json", ".csv"} or old.name.endswith(".tmp")
        ):
            old.unlink()


def prepare_data(
    mode: str = "retrieval_full",
    min_files: int = MIN_FILES_PER_CLASS,
    benchmark_songs: list[str] | None = None,
) -> None:
    if mode not in {"retrieval_full", "classifier_subset"}:
        raise ValueError("mode must be 'retrieval_full' or 'classifier_subset'")

    data_root = PROJECT_ROOT / "data"
    manifest_dir = data_root / "manifests"
    raw_dir = data_root / "Humming Audio"

    processed_dir = (
        data_root / "processed" / "classifier_subset"
        if mode == "classifier_subset"
        else data_root / "processed" / "datasets"
    )
    clear_processed_dir(processed_dir)

    manifest_result = load_manifest_records(manifest_dir)
    if manifest_result is not None:
        records, class_to_idx, idx_to_class = manifest_result
        LOGGER.info("Manifest-first dataset construction enabled.")
    else:
        records, class_to_idx, idx_to_class = records_from_directory(raw_dir)
        LOGGER.info("Directory-scan fallback enabled.")

    if mode == "classifier_subset":
        records = filter_records_for_classifier(
            records,
            min_files=min_files,
            benchmark_songs=benchmark_songs or BENCHMARK_SONGS or None,
        )
        if not records:
            raise RuntimeError(
                f"No songs meet min_files={min_files}. Add more recordings or use retrieval_full."
            )
        records, class_to_idx, idx_to_class = relabel_records(records)

    splits = split_records(records, seed=SEED)

    LOGGER.info(
        "Final dataset: %d songs, %d recordings | train=%d val=%d test=%d",
        len(class_to_idx),
        len(records),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    with open(processed_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

    write_split_manifest(processed_dir, splits, idx_to_class)

    extractor = FeatureExtractor(config_path=str(PROJECT_ROOT / "config" / "audio_config.yaml"))
    for split_name in ("train", "val", "test"):
        process_split(split_name, splits[split_name], extractor, processed_dir, seed=SEED)

    LOGGER.info("Dataset ready: %s", processed_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe Hum2Tune datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["retrieval_full", "classifier_subset"], default="retrieval_full")
    parser.add_argument("--min-files", type=int, default=MIN_FILES_PER_CLASS)
    parser.add_argument("--songs", type=str, nargs="*", default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_data(mode=args.mode, min_files=args.min_files, benchmark_songs=args.songs)
