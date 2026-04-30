from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_1d_int_array(values: NDArray[Any] | List[Any]) -> IntArray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _as_2d_float_array(values: NDArray[Any] | List[Any]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D score/probability array, got shape {arr.shape}")
    return arr


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_report_row(report: Mapping[str, Any], row_name: str) -> Mapping[str, Any]:
    row = report.get(row_name, {})
    if isinstance(row, Mapping):
        return cast(Mapping[str, Any], row)
    return {}


def _get_report_metric(
    report: Mapping[str, Any],
    row_name: str,
    metric_name: str,
    default: float = 0.0,
) -> float:
    row = _get_report_row(report, row_name)
    return _safe_float(row.get(metric_name, default), default)


def _validate_classification_inputs(
    y_true: NDArray[Any] | List[Any],
    y_pred: NDArray[Any] | List[Any],
) -> Tuple[IntArray, IntArray]:
    true_arr = _as_1d_int_array(y_true)
    pred_arr = _as_1d_int_array(y_pred)

    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of samples. "
            f"Got {true_arr.shape[0]} and {pred_arr.shape[0]}."
        )

    return true_arr, pred_arr


def _validate_ranking_inputs(
    y_true: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
) -> Tuple[IntArray, FloatArray]:
    true_arr = _as_1d_int_array(y_true)
    prob_arr = _as_2d_float_array(y_prob)

    if true_arr.shape[0] != prob_arr.shape[0]:
        raise ValueError(
            "y_true and y_prob must contain the same number of samples. "
            f"Got {true_arr.shape[0]} and {prob_arr.shape[0]}."
        )

    if prob_arr.shape[1] <= 0:
        raise ValueError("y_prob must contain at least one class column.")

    return true_arr, prob_arr


def compute_top_k_accuracy(
    y_true: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    true_arr, prob_arr = _validate_ranking_inputs(y_true, y_prob)

    if k_values is None:
        k_values = [1, 3, 5]

    num_classes = int(prob_arr.shape[1])
    labels = np.arange(num_classes, dtype=np.int64)

    results: Dict[str, float] = {}

    for k in k_values:
        k = int(k)

        if k <= 0 or k > num_classes:
            continue

        try:
            score = top_k_accuracy_score(true_arr, prob_arr, k=k, labels=labels)
            results[f"top_{k}_accuracy"] = float(score)
        except ValueError:
            ranked = np.argsort(-prob_arr, axis=1)[:, :k]
            hits = np.any(ranked == true_arr[:, None], axis=1)
            results[f"top_{k}_accuracy"] = float(np.mean(hits))

    return results


def compute_mrr(
    y_true: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
) -> float:
    true_arr, prob_arr = _validate_ranking_inputs(y_true, y_prob)
    ranked = np.argsort(-prob_arr, axis=1)

    reciprocal_ranks: List[float] = []

    for i, true_label in enumerate(true_arr):
        positions = np.where(ranked[i] == true_label)[0]

        if len(positions) == 0:
            reciprocal_ranks.append(0.0)
        else:
            rank = int(positions[0]) + 1
            reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_map(
    y_true: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
    max_k: int = 10,
) -> float:
    true_arr, prob_arr = _validate_ranking_inputs(y_true, y_prob)

    if max_k <= 0:
        return 0.0

    actual_k = min(int(max_k), int(prob_arr.shape[1]))
    ranked = np.argsort(-prob_arr, axis=1)[:, :actual_k]

    average_precisions: List[float] = []

    for i, true_label in enumerate(true_arr):
        hits = (ranked[i] == true_label).astype(np.float64)

        if float(hits.sum()) == 0.0:
            average_precisions.append(0.0)
            continue

        cumulative_hits = np.cumsum(hits)
        precision_at_k = cumulative_hits / (np.arange(actual_k) + 1)
        ap = float(np.sum(precision_at_k * hits) / hits.sum())
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_ndcg(
    y_true: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
    max_k: int = 10,
) -> float:
    true_arr, prob_arr = _validate_ranking_inputs(y_true, y_prob)

    if max_k <= 0:
        return 0.0

    actual_k = min(int(max_k), int(prob_arr.shape[1]))
    ranked = np.argsort(-prob_arr, axis=1)[:, :actual_k]

    discounts = 1.0 / np.log2(np.arange(2, actual_k + 2))
    ideal_dcg = float(discounts[0])
    ndcg_scores: List[float] = []

    for i, true_label in enumerate(true_arr):
        hits = (ranked[i] == true_label).astype(np.float64)
        dcg = float(np.sum(hits * discounts))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_classification_metrics(
    y_true: NDArray[Any] | List[Any],
    y_pred: NDArray[Any] | List[Any],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    true_arr, pred_arr = _validate_classification_inputs(y_true, y_pred)

    if class_names is not None:
        labels = list(range(len(class_names)))

        report_raw = classification_report(
            true_arr,
            pred_arr,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_str = classification_report(
            true_arr,
            pred_arr,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    else:
        report_raw = classification_report(
            true_arr,
            pred_arr,
            output_dict=True,
            zero_division=0,
        )

        report_str = classification_report(
            true_arr,
            pred_arr,
            zero_division=0,
        )

    report_dict = cast(Mapping[str, Any], report_raw)

    results: Dict[str, Any] = {
        "accuracy": _safe_float(report_dict.get("accuracy", 0.0)),
        "macro_precision": _get_report_metric(report_dict, "macro avg", "precision"),
        "macro_recall": _get_report_metric(report_dict, "macro avg", "recall"),
        "macro_f1": _get_report_metric(report_dict, "macro avg", "f1-score"),
        "weighted_f1": _get_report_metric(report_dict, "weighted avg", "f1-score"),
        "report_str": report_str,
        "per_class": {},
    }

    per_class: Dict[str, Dict[str, float | int]] = {}

    if class_names is not None:
        for class_name in class_names:
            row = _get_report_row(report_dict, class_name)

            per_class[class_name] = {
                "precision": round(_safe_float(row.get("precision", 0.0)), 4),
                "recall": round(_safe_float(row.get("recall", 0.0)), 4),
                "f1": round(_safe_float(row.get("f1-score", 0.0)), 4),
                "support": _safe_int(row.get("support", 0)),
            }

    results["per_class"] = per_class
    return results


def compute_confusion_matrix(
    y_true: NDArray[Any] | List[Any],
    y_pred: NDArray[Any] | List[Any],
    num_classes: Optional[int] = None,
    normalise: bool = False,
) -> NDArray[Any]:
    true_arr, pred_arr = _validate_classification_inputs(y_true, y_pred)

    if num_classes is not None:
        labels = list(range(int(num_classes)))
        cm = confusion_matrix(true_arr, pred_arr, labels=labels)
    else:
        cm = confusion_matrix(true_arr, pred_arr)

    if not normalise:
        return cm

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_float = cm.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.divide(
            cm_float,
            row_sums,
            out=np.zeros_like(cm_float),
            where=row_sums != 0,
        )

    return normalised


def most_confused_pairs(
    y_true: NDArray[Any] | List[Any],
    y_pred: NDArray[Any] | List[Any],
    class_names: List[str],
    top_n: int = 5,
) -> List[Tuple[str, str, int]]:
    true_arr, pred_arr = _validate_classification_inputs(y_true, y_pred)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(true_arr, pred_arr, labels=labels)

    pairs: List[Tuple[str, str, int]] = []

    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            count = int(cm[true_idx, pred_idx])

            if true_idx == pred_idx or count <= 0:
                continue

            true_name = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

            pairs.append((true_name, pred_name, count))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[: int(top_n)]


def compute_all_metrics(
    y_true: NDArray[Any] | List[Any],
    y_pred: NDArray[Any] | List[Any],
    y_prob: NDArray[Any] | List[Any],
    class_names: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    true_arr, prob_arr = _validate_ranking_inputs(y_true, y_prob)
    pred_arr = _as_1d_int_array(y_pred)

    if pred_arr.shape[0] != true_arr.shape[0]:
        raise ValueError(
            "y_pred must contain the same number of samples as y_true. "
            f"Got {pred_arr.shape[0]} and {true_arr.shape[0]}."
        )

    if top_k_values is None:
        top_k_values = [1, 3, 5]

    num_classes = int(prob_arr.shape[1])
    results: Dict[str, Any] = {}

    results.update(compute_top_k_accuracy(true_arr, prob_arr, k_values=top_k_values))
    results["mrr"] = compute_mrr(true_arr, prob_arr)
    results["map_at_10"] = compute_map(true_arr, prob_arr, max_k=10)
    results["ndcg_at_10"] = compute_ndcg(true_arr, prob_arr, max_k=10)

    results.update(
        compute_classification_metrics(
            true_arr,
            pred_arr,
            class_names=class_names,
        )
    )

    results["confusion_matrix"] = compute_confusion_matrix(
        true_arr,
        pred_arr,
        num_classes=num_classes,
        normalise=False,
    ).tolist()

    if class_names is not None:
        results["most_confused_pairs"] = most_confused_pairs(
            true_arr,
            pred_arr,
            class_names,
        )
    else:
        results["most_confused_pairs"] = []

    return results


if __name__ == "__main__":
    y_true_test = np.array([0, 1, 2, 1, 0])
    y_pred_test = np.array([0, 1, 1, 1, 0])
    y_prob_test = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.20, 0.60, 0.20],
            [0.15, 0.70, 0.15],
            [0.85, 0.10, 0.05],
        ],
        dtype=np.float64,
    )
    class_names_test = ["Song A", "Song B", "Song C"]

    metrics = compute_all_metrics(
        y_true=y_true_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
        class_names=class_names_test,
    )

    print("Accuracy:", metrics["accuracy"])
    print("Top-1:", metrics["top_1_accuracy"])
    print("MRR:", metrics["mrr"])
    print("Most confused:", metrics["most_confused_pairs"])
