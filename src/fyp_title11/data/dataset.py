from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

PathLike = Union[str, Path]


class RealAudioDataset(Dataset):
    def __init__(self, source: Union[PathLike, Tuple[np.ndarray, np.ndarray]]):
        self.X: torch.Tensor
        self.y: torch.Tensor

        if isinstance(source, tuple):
            X, y = source
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.long)
            return

        path = Path(source)
        if path.suffix != ".npz":
            raise ValueError(
                f"RealAudioDataset expects a .npz file or (X, y) tuple, got: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            data = np.load(path)
            self.X = torch.as_tensor(data["X"], dtype=torch.float32)
            self.y = torch.as_tensor(data["y"], dtype=torch.long)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset {path}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ChunkedAudioDataset(Dataset):
    """Lazy dataset backed by chunked `.npy` files.

    The dataset reads chunk arrays with `mmap_mode='r'` so large datasets do not
    need to be fully loaded into RAM.
    """

    def __init__(self, meta_path: PathLike, mmap_mode: Optional[str] = "r"):
        self.meta_path = Path(meta_path)
        if self.meta_path.suffix != ".json":
            raise ValueError(f"ChunkedAudioDataset expects a *_meta.json file, got: {self.meta_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.data_dir = self.meta_path.parent
        self.stem = self.meta.get("stem", self.meta_path.stem.replace("_meta", ""))
        self.n_chunks = int(self.meta.get("n_chunks", 0))
        if self.n_chunks <= 0:
            raise ValueError(f"Invalid chunk metadata in {self.meta_path}")

        self._x_chunks = []
        self._y_chunks = []
        self._lengths = []
        self._offsets = [0]

        for i in range(self.n_chunks):
            x_path = self.data_dir / f"{self.stem}_chunk{i}_X.npy"
            y_path = self.data_dir / f"{self.stem}_chunk{i}_y.npy"
            if not x_path.exists() or not y_path.exists():
                raise FileNotFoundError(f"Missing chunk pair: {x_path.name}, {y_path.name}")
            x_arr = np.load(x_path, mmap_mode=mmap_mode)
            y_arr = np.load(y_path, mmap_mode=mmap_mode)
            if len(x_arr) != len(y_arr):
                raise ValueError(f"Chunk length mismatch in {x_path.name}")
            self._x_chunks.append(x_arr)
            self._y_chunks.append(y_arr)
            self._lengths.append(len(x_arr))
            self._offsets.append(self._offsets[-1] + len(x_arr))

        self.total = self._offsets[-1]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        chunk_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[chunk_idx]
        x = torch.as_tensor(np.array(self._x_chunks[chunk_idx][local_idx]), dtype=torch.float32)
        y = torch.as_tensor(int(self._y_chunks[chunk_idx][local_idx]), dtype=torch.long)
        return x, y

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self._x_chunks[0].shape[1:])


def resolve_split_dataset(data_dir: PathLike, stem: str) -> Dataset:
    """Load a split dataset from either chunked or legacy format."""
    data_dir = Path(data_dir)
    meta_path = data_dir / f"{stem}_meta.json"
    if meta_path.exists():
        return ChunkedAudioDataset(meta_path)

    npz_path = data_dir / f"{stem}.npz"
    if npz_path.exists():
        return RealAudioDataset(npz_path)

    raise FileNotFoundError(
        f"Could not find dataset split '{stem}' in {data_dir}. "
        "Expected either *_meta.json + chunks or a .npz file."
    )


def load_class_map(path: PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Class map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
