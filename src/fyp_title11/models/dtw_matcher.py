"""DTW sequence matching utility for Hum2Tune.

NOTE: This module was previously dead code (never called from the pipeline).
It has been consolidated to use librosa.sequence.dtw -- the same backend
as hybrid_retrieval.py -- so there is one consistent DTW implementation.

The fastdtw dependency has been removed: it normalized by (len(a)+len(b))
which is inconsistent with the path-length normalization used everywhere
else in this project.

Usage:
    from fyp_title11.models.dtw_matcher import dtw_distance, subseq_dtw_distance
"""

from __future__ import annotations

import numpy as np
import librosa


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance between two 1-D pitch sequences.

    Normalises by warping path length (not input length sum) to make the
    score independent of sequence duration. This is consistent with the
    normalization used in hybrid_retrieval.py.

    Args:
        seq1: Query pitch sequence, shape (T1,).
        seq2: Reference pitch sequence, shape (T2,).

    Returns:
        Path-length-normalised DTW distance (lower = more similar).
    """
    X = seq1[np.newaxis, :]   # (1, T1) -- librosa expects (dim, time)
    Y = seq2[np.newaxis, :]   # (1, T2)
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    path_len = max(len(wp), 1)
    return float(D[-1, -1] / path_len)


def subseq_dtw_distance(query: np.ndarray, reference: np.ndarray) -> float:
    """Subsequence DTW: find the best-matching segment within reference.

    More robust than global DTW when the query (humming clip) is much
    shorter than the reference (full song vocal track). Locates the
    position in the reference that minimises alignment cost.

    Args:
        query:     Query pitch sequence, shape (T_q,).
        reference: Reference pitch sequence, shape (T_r,), T_r >= T_q.

    Returns:
        Path-length-normalised cost of the best-matching subsequence.
    """
    X = query[np.newaxis, :]
    Y = reference[np.newaxis, :]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean", subseq=True)
    end_j = int(np.argmin(D[-1, :]))
    best_cost = float(D[-1, end_j])
    path_len = max(len(wp), 1)
    return best_cost / path_len
