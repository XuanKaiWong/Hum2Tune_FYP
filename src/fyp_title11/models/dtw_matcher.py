from __future__ import annotations

import numpy as np
import librosa


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    X = seq1[np.newaxis, :]   # (1, T1) -- librosa expects (dim, time)
    Y = seq2[np.newaxis, :]   # (1, T2)
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    path_len = max(len(wp), 1)
    return float(D[-1, -1] / path_len)


def subseq_dtw_distance(query: np.ndarray, reference: np.ndarray) -> float:
    X = query[np.newaxis, :]
    Y = reference[np.newaxis, :]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean", subseq=True)
    end_j = int(np.argmin(D[-1, :]))
    best_cost = float(D[-1, end_j])
    path_len = max(len(wp), 1)
    return best_cost / path_len
