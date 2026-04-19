"""Tests for evaluation metrics in Hum2Tune.

Covers: Top-K accuracy, MRR, MAP@K, NDCG@K, confusion matrix,
and most_confused_pairs. All functions use immutable default args.
"""

import warnings
import numpy as np
import pytest

from src.fyp_title11.evaluation.metrics import (
    compute_top_k_accuracy,
    compute_mrr,
    compute_map,
    compute_ndcg,
    compute_confusion_matrix,
    most_confused_pairs,
    compute_all_metrics,
)

# Suppress PyTorch nested-tensor UserWarning emitted by TransformerEncoder
# when norm_first=True. This is a known harmless warning in PyTorch < 2.1.
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
    category=UserWarning,
)


@pytest.fixture
def perfect_probs():
    """5 samples, 5 classes, model always ranks the correct class first.

    y_true and y_prob.shape[1] must agree on the number of classes so
    sklearn's top_k_accuracy_score does not raise a ValueError.
    """
    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5, dtype=float)   # diagonal: correct class gets score 1.0
    return y_true, y_prob


@pytest.fixture
def worst_probs():
    """5 samples, 5 classes: model gives the correct class the lowest score."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.full((5, 5), 0.25, dtype=float)
    for i, cls in enumerate(y_true):
        y_prob[i, cls] = 0.0   # correct class is scored 0 -> always ranked last
    return y_true, y_prob


# -- Top-K accuracy --------------------------------------------------------

def test_top1_perfect(perfect_probs):
    y_true, y_prob = perfect_probs
    result = compute_top_k_accuracy(y_true, y_prob, k_values=[1, 3, 5])
    assert result["top_1_accuracy"] == pytest.approx(1.0)


def test_top1_worst(worst_probs):
    y_true, y_prob = worst_probs
    result = compute_top_k_accuracy(y_true, y_prob, k_values=[1])
    assert result["top_1_accuracy"] == pytest.approx(0.0)


def test_top_k_default_k_values():
    """Default k_values should be [1, 3, 5] when not specified."""
    y_true = np.array([0, 1, 2, 3, 4])   # 5 classes -- matches y_prob cols
    y_prob = np.eye(5)                    # shape (5, 5)
    result = compute_top_k_accuracy(y_true, y_prob)
    assert "top_1_accuracy" in result
    assert "top_3_accuracy" in result
    assert "top_5_accuracy" in result


def test_top_k_no_mutation_of_default():
    """Calling the function twice with identical inputs gives identical results.

    Uses 3 classes (not 2) to avoid sklearn's binary-mode edge case where
    a 2D score matrix is rejected when y_true has exactly 2 unique values.
    """
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3, dtype=float)   # 3x3: classes match y_prob columns
    r1 = compute_top_k_accuracy(y_true, y_prob, k_values=[1, 3])
    r2 = compute_top_k_accuracy(y_true, y_prob, k_values=[1, 3])
    assert r1 == r2


# -- MRR ------------------------------------------------------------------

def test_mrr_perfect(perfect_probs):
    y_true, y_prob = perfect_probs
    assert compute_mrr(y_true, y_prob) == pytest.approx(1.0)


def test_mrr_bounded():
    y_true = np.array([0, 1])
    y_prob = np.array([[0.1, 0.9], [0.9, 0.1]])   # both wrong at rank 1
    mrr = compute_mrr(y_true, y_prob)
    assert 0.0 <= mrr <= 1.0


# -- MAP ------------------------------------------------------------------

def test_map_perfect(perfect_probs):
    y_true, y_prob = perfect_probs
    assert compute_map(y_true, y_prob, max_k=5) == pytest.approx(1.0)


def test_map_zero_when_answer_beyond_cutoff(worst_probs):
    y_true, y_prob = worst_probs
    # Correct answer is always last (rank 5), beyond max_k=3 cutoff
    result = compute_map(y_true, y_prob, max_k=3)
    assert result == pytest.approx(0.0)


# -- NDCG -----------------------------------------------------------------

def test_ndcg_perfect(perfect_probs):
    y_true, y_prob = perfect_probs
    assert compute_ndcg(y_true, y_prob, max_k=5) == pytest.approx(1.0)


def test_ndcg_bounded(worst_probs):
    y_true, y_prob = worst_probs
    ndcg = compute_ndcg(y_true, y_prob, max_k=5)
    assert 0.0 <= ndcg <= 1.0


# -- Confusion matrix ------------------------------------------------------

def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 1, 0])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)


def test_confusion_matrix_normalised_rows_sum_to_one():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred, normalise=True)
    row_sums = cm.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


# -- most_confused_pairs ---------------------------------------------------

def test_most_confused_pairs_returns_correct_length():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([1, 1, 0, 2, 1])
    names = ["song_a", "song_b", "song_c"]
    pairs = most_confused_pairs(y_true, y_pred, names, top_n=2)
    assert len(pairs) <= 2
    for true_name, pred_name, count in pairs:
        assert true_name in names
        assert pred_name in names
        assert count > 0


# -- compute_all_metrics ---------------------------------------------------

def test_compute_all_metrics_keys(perfect_probs):
    y_true, y_prob = perfect_probs
    y_pred = y_prob.argmax(axis=1)
    result = compute_all_metrics(y_true, y_pred, y_prob)

    required_keys = [
        "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
        "mrr", "map_at_10", "ndcg_at_10",
        "accuracy", "macro_f1", "confusion_matrix",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_compute_all_metrics_subset_classes_does_not_crash():
    """Critical regression test: when the evaluation split contains only a
    subset of all known classes, metrics must not raise ValueError.

    Real-world scenario:
      - Training set has 100 songs (100 classes).
      - Test split (small) only contains samples for 7 of those 100 classes.
      - class_names list still has 100 entries.

    Before the fix, classification_report raised:
        ValueError: Number of classes, 7, does not match size of target_names, 100
    """
    num_all_classes = 100
    num_present_classes = 7
    n_samples = 20

    rng = np.random.default_rng(42)
    # y_true only draws from first 7 classes (indices 0-6)
    y_true = rng.integers(0, num_present_classes, size=n_samples)
    # y_prob still has 100 columns (one per class in the full training set)
    y_prob = rng.dirichlet(np.ones(num_all_classes), size=n_samples)
    y_pred = y_prob.argmax(axis=1)
    class_names = [f"song_{i}" for i in range(num_all_classes)]

    # Must not raise -- this was the crash reported in v3/v4
    result = compute_all_metrics(y_true, y_pred, y_prob, class_names=class_names)

    assert "accuracy" in result
    assert "macro_f1" in result
    assert "mrr" in result
    # Confusion matrix must be square with one row/col per class
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (num_all_classes, num_all_classes), (
        f"Expected ({num_all_classes}, {num_all_classes}), got {cm.shape}"
    )


def test_confusion_matrix_shape_with_num_classes():
    """compute_confusion_matrix with num_classes always returns (C, C) matrix."""
    from src.fyp_title11.evaluation.metrics import compute_confusion_matrix
    y_true = np.array([0, 0, 1])    # only 2 classes present
    y_pred = np.array([0, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=10)
    assert cm.shape == (10, 10), f"Expected (10, 10), got {cm.shape}"
    # Classes 2-9 are absent -> all zeros in those rows/cols
    assert cm[2:, :].sum() == 0
    assert cm[:, 2:].sum() == 0


def test_classification_metrics_subset_classes():
    """compute_classification_metrics must not crash when only some classes
    appear in y_true but class_names covers all 100."""
    from src.fyp_title11.evaluation.metrics import compute_classification_metrics
    num_classes = 100
    y_true = np.array([0, 1, 2, 0, 1])    # only classes 0, 1, 2 present
    y_pred = np.array([0, 1, 0, 0, 2])
    class_names = [f"song_{i}" for i in range(num_classes)]

    result = compute_classification_metrics(y_true, y_pred, class_names)
    assert "accuracy" in result
    assert "macro_f1" in result
    # per_class should have entries for all 100 classes
    assert len(result["per_class"]) == num_classes
