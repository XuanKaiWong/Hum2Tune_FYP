"""
METRICS.PY -- Evaluation Metrics for Hum2Tune
FYP Hum2Tune: Melody-Based Music Recognition

Provides all metric computation functions used in evaluate.py and notebooks.
Metrics included:
  - Top-K accuracy (Top-1, Top-3, Top-5)
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP@K) -- standard IR metric
  - Normalised Discounted Cumulative Gain (NDCG@K) -- standard IR metric
  - Per-class classification metrics (precision, recall, F1)
  - Confusion matrix and most-confused pair analysis

Design rules applied throughout:
  - No mutable default arguments (all list defaults use Optional[List] + None).
  - Explicit `labels=` passed to every sklearn function so the code handles
    the common real-world case where the evaluation split contains only a
    subset of the full class set (e.g. a 7-class test set out of 100 classes).
    Without `labels=`, sklearn raises:
        ValueError: Number of classes, 7, does not match size of target_names, 100
  - Confusion matrix always shaped to (num_classes, num_classes) when
    class_names is provided, so downstream plot functions always receive a
    square matrix with one row/column per known class.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


# -----------------------------------------------------------------------------
#  Ranking-based metrics
# -----------------------------------------------------------------------------

def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute Top-K accuracy for multiple values of K.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_prob: Predicted probability scores, shape (N, num_classes).
        k_values: K values to evaluate. Defaults to [1, 3, 5].

    Returns:
        Dict mapping 'top_k_accuracy' -> float.
        E.g. {'top_1_accuracy': 0.82, 'top_3_accuracy': 0.95, ...}
    """
    if k_values is None:
        k_values = [1, 3, 5]

    num_classes = y_prob.shape[1]
    # Pass labels= explicitly so sklearn works when not all classes appear in
    # y_true (common with small or stratified test sets).
    all_labels = np.arange(num_classes)
    results = {}
    for k in k_values:
        if k > num_classes:
            continue
        score = top_k_accuracy_score(y_true, y_prob, k=k, labels=all_labels)
        results[f"top_{k}_accuracy"] = float(score)
    return results


def compute_mrr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    MRR = 1 -> correct song always ranked #1.
    MRR = 0.5 -> correct song on average ranked #2.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_prob: Probability scores, shape (N, num_classes).

    Returns:
        MRR as a float in [0, 1].
    """
    ranked = np.argsort(-y_prob, axis=1)   # descending probability order
    reciprocal_ranks = []
    for i in range(len(y_true)):
        positions = np.where(ranked[i] == y_true[i])[0]
        if len(positions) == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (int(positions[0]) + 1))
    return float(np.mean(reciprocal_ranks))


def compute_map(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_k: int = 10,
) -> float:
    """Compute Mean Average Precision (MAP) at rank max_k.

    MAP rewards systems that place the correct answer near the top of the
    ranking. It is the standard retrieval metric at ISMIR and MIREX.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_prob: Probability scores, shape (N, num_classes).
        max_k:  Rank cut-off for AP computation.

    Returns:
        MAP@max_k as a float in [0, 1].
    """
    # Clamp max_k to actual num_classes to avoid shape mismatch.
    actual_k = min(max_k, y_prob.shape[1])
    ranked = np.argsort(-y_prob, axis=1)[:, :actual_k]
    average_precisions = []
    for i in range(len(y_true)):
        hits = (ranked[i] == y_true[i]).astype(float)
        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue
        cumulative_hits = np.cumsum(hits)
        precision_at_k = cumulative_hits / (np.arange(actual_k) + 1)
        ap = float(np.sum(precision_at_k * hits) / hits.sum())
        average_precisions.append(ap)
    return float(np.mean(average_precisions))


def compute_ndcg(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_k: int = 10,
) -> float:
    """Compute Normalised Discounted Cumulative Gain (NDCG) at rank max_k.

    NDCG applies a logarithmic discount to penalise correct answers at lower
    ranks. NDCG = 1 -> correct song always ranked #1.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_prob: Probability scores, shape (N, num_classes).
        max_k:  Rank cut-off.

    Returns:
        NDCG@max_k as a float in [0, 1].
    """
    actual_k = min(max_k, y_prob.shape[1])
    ranked = np.argsort(-y_prob, axis=1)[:, :actual_k]
    discounts = 1.0 / np.log2(np.arange(2, actual_k + 2))
    ndcg_scores = []
    for i in range(len(y_true)):
        hits = (ranked[i] == y_true[i]).astype(float)
        dcg = float(np.sum(hits * discounts))
        idcg = discounts[0]   # ideal: correct answer at rank 1
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores))


# -----------------------------------------------------------------------------
#  Classification metrics
# -----------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute full per-class classification metrics.

    Passes `labels=` to sklearn so the report always covers all known
    classes -- not just the subset that appears in y_true. This prevents
    the crash:
        ValueError: Number of classes, 7, does not match size of target_names, 100

    Args:
        y_true:       Ground-truth class indices.
        y_pred:       Predicted class indices.
        class_names:  Optional list of human-readable class names indexed by
                      class integer. len(class_names) defines the full class set.

    Returns:
        Dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
        weighted_f1, per_class (dict), report_str (formatted string).
    """
    if class_names is not None:
        # labels= forces sklearn to report all num_classes rows even when some
        # classes are absent from the evaluation split.
        labels = list(range(len(class_names)))
        report_dict = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        report_str = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    else:
        report_dict = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0,
        )
        report_str = classification_report(
            y_true, y_pred,
            zero_division=0,
        )

    results: Dict = {
        "accuracy":         float(report_dict["accuracy"]),
        "macro_precision":  float(report_dict["macro avg"]["precision"]),
        "macro_recall":     float(report_dict["macro avg"]["recall"]),
        "macro_f1":         float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1":      float(report_dict["weighted avg"]["f1-score"]),
        "report_str":       report_str,
        "per_class":        {},
    }

    if class_names:
        for name in class_names:
            if name in report_dict:
                results["per_class"][name] = {
                    "precision": round(report_dict[name]["precision"], 4),
                    "recall":    round(report_dict[name]["recall"], 4),
                    "f1":        round(report_dict[name]["f1-score"], 4),
                    "support":   int(report_dict[name]["support"]),
                }

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    normalise: bool = False,
) -> np.ndarray:
    """Compute confusion matrix of shape (num_classes, num_classes).

    Passes `labels=` so the matrix is always square and covers all known
    classes -- not just those present in y_true. This ensures downstream
    plot functions receive a matrix with the expected number of rows and
    columns even when the test split is small.

    Args:
        y_true:       Ground-truth class indices.
        y_pred:       Predicted class indices.
        num_classes:  Total number of classes. Inferred from y_true/y_pred if
                      None, but should always be passed when using class_names.
        normalise:    If True, return row-normalised matrix (proportions).

    Returns:
        Confusion matrix as a 2D numpy array, shape (num_classes, num_classes).
    """
    if num_classes is not None:
        labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        cm = confusion_matrix(y_true, y_pred)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums == 0, 0.0, cm.astype(float) / row_sums)

    return cm


def most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = 5,
) -> List[Tuple[str, str, int]]:
    """Find the most frequently confused class pairs.

    Useful for error analysis -- shows which songs the model struggles to
    distinguish (e.g. songs with similar choruses or tempo profiles).

    Args:
        y_true:       Ground-truth class indices.
        y_pred:       Predicted class indices.
        class_names:  List of class names indexed by class integer.
        top_n:        Number of top confused pairs to return.

    Returns:
        List of (true_class_name, predicted_class_name, count) tuples,
        sorted descending by count. Off-diagonal cells only.
    """
    # Pass labels= so matrix covers all classes, not just those in y_true.
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                true_name = class_names[i] if i < len(class_names) else str(i)
                pred_name = class_names[j] if j < len(class_names) else str(j)
                pairs.append((true_name, pred_name, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


# -----------------------------------------------------------------------------
#  Combined convenience function
# -----------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
) -> Dict:
    """Compute all metrics in one call.

    Args:
        y_true:       Ground-truth class indices, shape (N,).
        y_pred:       Predicted class indices, shape (N,).
        y_prob:       Probability scores, shape (N, num_classes).
        class_names:  Optional list of class name strings.
        top_k_values: K values for Top-K accuracy. Defaults to [1, 3, 5].

    Returns:
        Merged dictionary of all metric results.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5]

    num_classes = y_prob.shape[1]
    results: Dict = {}

    # Ranking-based retrieval metrics
    results.update(compute_top_k_accuracy(y_true, y_prob, k_values=top_k_values))
    results["mrr"]       = compute_mrr(y_true, y_prob)
    results["map_at_10"] = compute_map(y_true, y_prob, max_k=10)
    results["ndcg_at_10"]= compute_ndcg(y_true, y_prob, max_k=10)

    # Per-class classification metrics (labels= handled inside)
    results.update(compute_classification_metrics(y_true, y_pred, class_names))

    # Confusion matrix -- pass num_classes so matrix is always (C, C)
    results["confusion_matrix"] = compute_confusion_matrix(
        y_true, y_pred, num_classes=num_classes
    ).tolist()

    # Most confused pairs
    if class_names:
        results["most_confused_pairs"] = most_confused_pairs(
            y_true, y_pred, class_names
        )

    return results
