import numpy as np
import pytest

from src.fyp_title11.models.dtw_matcher import dtw_distance, subseq_dtw_distance


def _sine(freq_hz: float, duration_s: float = 1.0, sr: int = 100) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_dtw_distance_identical_sequences():
    """DTW distance of a sequence with itself should be zero."""
    seq = _sine(5.0)
    assert dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-5)


def test_dtw_distance_is_non_negative():
    a = _sine(3.0)
    b = _sine(7.0)
    assert dtw_distance(a, b) >= 0.0


def test_dtw_distance_similar_less_than_dissimilar():
    """Sequences with the same frequency should match better than different ones."""
    ref    = _sine(5.0, duration_s=2.0)
    close  = _sine(5.5, duration_s=2.0)   # small frequency shift
    far    = _sine(20.0, duration_s=2.0)  # large frequency shift
    assert dtw_distance(ref, close) < dtw_distance(ref, far)


def test_subseq_dtw_distance_embedded_query():
    """Subsequence DTW should find a near-zero cost when query is embedded."""
    # Build a reference that contains the query as a subsequence
    query = _sine(4.0, duration_s=1.0)
    padding = np.zeros(50, dtype=np.float32)
    reference = np.concatenate([padding, query, padding])
    dist = subseq_dtw_distance(query, reference)
    assert dist < 0.5, f"Subsequence match cost unexpectedly high: {dist:.4f}"


def test_subseq_dtw_distance_non_negative():
    a = _sine(6.0)
    b = _sine(6.0, duration_s=3.0)
    assert subseq_dtw_distance(a, b) >= 0.0


def test_dtw_normalisation_independent_of_length():
    """Path-length normalisation makes score comparable across different lengths."""
    short_ref = _sine(5.0, duration_s=1.0)
    long_ref  = _sine(5.0, duration_s=4.0)
    query     = _sine(5.0, duration_s=1.0)

    dist_short = dtw_distance(query, short_ref)
    dist_long  = dtw_distance(query, long_ref)
    # Both should be close to zero since frequencies match
    assert dist_short < 1.0
    assert dist_long  < 1.0
