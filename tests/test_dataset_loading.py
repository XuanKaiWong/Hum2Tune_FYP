import json
from pathlib import Path

import numpy as np

from src.fyp_title11.data.dataset import ChunkedAudioDataset, RealAudioDataset, resolve_split_dataset


def test_real_audio_dataset_from_npz(tmp_path: Path):
    x = np.random.randn(4, 40, 10).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    np.savez(tmp_path / "sample.npz", X=x, y=y)
    ds = RealAudioDataset(tmp_path / "sample.npz")
    assert len(ds) == 4
    fx, fy = ds[0]
    assert fx.shape == (40, 10)
    assert int(fy) in {0, 1}


def test_chunked_dataset(tmp_path: Path):
    x0 = np.random.randn(2, 40, 10).astype(np.float32)
    y0 = np.array([0, 1], dtype=np.int64)
    x1 = np.random.randn(1, 40, 10).astype(np.float32)
    y1 = np.array([1], dtype=np.int64)
    np.save(tmp_path / "train_data_chunk0_X.npy", x0)
    np.save(tmp_path / "train_data_chunk0_y.npy", y0)
    np.save(tmp_path / "train_data_chunk1_X.npy", x1)
    np.save(tmp_path / "train_data_chunk1_y.npy", y1)
    with open(tmp_path / "train_data_meta.json", "w", encoding="utf-8") as f:
        json.dump({"stem": "train_data", "n_chunks": 2, "total": 3}, f)

    ds = ChunkedAudioDataset(tmp_path / "train_data_meta.json")
    assert len(ds) == 3
    fx, fy = ds[2]
    assert fx.shape == (40, 10)
    assert int(fy) == 1

    ds2 = resolve_split_dataset(tmp_path, "train_data")
    assert len(ds2) == 3
