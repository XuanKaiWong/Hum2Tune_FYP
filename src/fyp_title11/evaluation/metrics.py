"""
METRICS.PY — Evaluation Metrics for Hum2Tune
FYP Hum2Tune: Melody-Based Music Recognition

Provides all metric computation functions used in evaluate.py and
the Jupyter notebooks. Centralises metric logic to avoid duplication.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Compute Top-K accuracy for multiple values of K.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_prob: Predicted probability scores, shape (N, num_classes).
        k_values: List of K values to evaluate.

    Returns:
        Dictionary mapping 'top_k_accuracy' keys to float scores.
        E.g. {'top_1_accuracy': 0.82, 'top_3_accuracy': 0.95, ...}
    """
    num_classes = y_prob.shape[1]
    results = {}
    for k in k_values:
        if k > num_classes:
            continue
        score = top_k_accuracy_score(y_true, y_prob, k=k)
        results[f"top_{k}_accuracy"] = float(score)
    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute full per-class classification metrics.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Optional list of human-readable class names.

    Returns:
        Dictionary with keys:
          - accuracy (float)
          - macro_precision, macro_recall, macro_f1 (float)
          - weighted_f1 (float)
          - per_class (dict of {class_name: {precision, recall, f1, support}})
          - report_str (formatted string for printing)
    """
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    results = {
        "accuracy": float(report_dict["accuracy"]),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "report_str": report_str,
        "per_class": {},
    }

    if class_names:
        for name in class_names:
            if name in report_dict:
                results["per_class"][name] = {
                    "precision": round(report_dict[name]["precision"], 4),
                    "recall": round(report_dict[name]["recall"], 4),
                    "f1": round(report_dict[name]["f1-score"], 4),
                    "support": int(report_dict[name]["support"]),
                }

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalise: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        normalise: If True, return row-normalised matrix (proportions).

    Returns:
        Confusion matrix as a 2D numpy array.
    """
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
    """
    Find the most frequently confused class pairs.

    Useful for error analysis in the FYP report — shows which songs
    the model struggles to distinguish.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: List of class names indexed by class index.
        top_n: Number of top confused pairs to return.

    Returns:
        List of (true_class, predicted_class, count) tuples,
        sorted by count descending. Off-diagonal only.
    """
    cm = confusion_matrix(y_true, y_pred)
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                true_name = class_names[i] if i < len(class_names) else str(i)
                pred_name = class_names[j] if j < len(class_names) else str(j)
                pairs.append((true_name, pred_name, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Convenience function: compute all metrics in one call.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).
        y_prob: Probability scores, shape (N, num_classes).
        class_names: Optional list of class name strings.
        top_k_values: K values for top-K accuracy.

    Returns:
        Merged dictionary of all metric results.
    """
    results = {}

    # Top-K accuracy
    results.update(compute_top_k_accuracy(y_true, y_prob, k_values=top_k_values))

    # Per-class metrics
    classification = compute_classification_metrics(y_true, y_pred, class_names)
    results.update(classification)

    # Confusion matrix (counts)
    results["confusion_matrix"] = compute_confusion_matrix(y_true, y_pred).tolist()

    # Most confused pairs
    if class_names:
        results["most_confused_pairs"] = most_confused_pairs(
            y_true, y_pred, class_names
        )

    return results