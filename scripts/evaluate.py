from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fyp_title11.data.dataset import load_class_map, resolve_split_dataset
from fyp_title11.models.cnn_lstm import CNNLSTMModel

log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

_utf8_stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "evaluate.log", encoding="utf-8"),
        logging.StreamHandler(stream=_utf8_stdout),
    ],
)
logger = logging.getLogger(__name__)

TOP_K_VALUES = [1, 3, 5]


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def load_training_config() -> dict[str, Any]:
    config_path = project_root / "config" / "training_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "training" in loaded and isinstance(loaded["training"], dict):
                return loaded["training"]

    return {
        "input_channels": 13,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.4,
        "bidirectional": True,
        "use_attention": True,
        "norm_type": "group",
        "batch_size": 4,
        "num_workers": 0,
    }


def build_model(num_classes: int, train_conf: dict[str, Any]) -> CNNLSTMModel:
    return CNNLSTMModel(
        input_channels=int(train_conf.get("input_channels", 13)),
        num_classes=num_classes,
        hidden_size=int(train_conf.get("hidden_size", 64)),
        num_layers=int(train_conf.get("num_layers", 1)),
        dropout=float(train_conf.get("dropout", 0.4)),
        bidirectional=bool(train_conf.get("bidirectional", True)),
        use_attention=bool(train_conf.get("use_attention", True)),
        norm_type=str(train_conf.get("norm_type", "group")),
    )


def compute_mrr(targets: np.ndarray, probs: np.ndarray) -> float:
    order = np.argsort(-probs, axis=1)
    reciprocal_ranks = []
    for target, ranked in zip(targets, order):
        rank = int(np.where(ranked == target)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_conf = load_training_config()

    data_dir = project_root / "data" / "processed" / "datasets"
    results_dir = project_root / "results" / "evaluations"
    viz_dir = project_root / "results" / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(data_dir / "classes.json")
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(class_names)

    test_dataset = resolve_split_dataset(data_dir, "test_data")
    batch_size = int(train_conf.get("batch_size", 4))
    num_workers = int(train_conf.get("num_workers", 0))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=num_classes, train_conf=train_conf).to(device)
    model_path = project_root / "models" / "cnn_lstm" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    checkpoint = torch_load(model_path, device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)

            logits = extract_logits(model(X))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())

    if not all_targets:
        raise ValueError("Test set is empty. Rebuild the dataset first.")

    probs_np = np.vstack(all_probs)
    preds_np = np.asarray(all_preds, dtype=np.int64)
    targets_np = np.asarray(all_targets, dtype=np.int64)

    metrics: dict[str, object] = {
        "model": "cnn_lstm",
        "num_classes_total": num_classes,
        "num_test_samples": int(len(all_targets)),
        "accuracy": float(accuracy_score(targets_np, preds_np)),
    }

    all_labels = np.arange(num_classes, dtype=np.int64)
    for k in TOP_K_VALUES:
        kk = min(k, num_classes)
        metrics[f"top_{kk}_accuracy"] = float(
            top_k_accuracy_score(targets_np, probs_np, k=kk, labels=all_labels)
        )
        metrics[f"recall_at_{kk}"] = metrics[f"top_{kk}_accuracy"]

    metrics["mrr"] = compute_mrr(targets_np, probs_np)

    present_labels = np.unique(targets_np)
    present_names = [class_names[i] for i in present_labels]

    report = classification_report(
        targets_np,
        preds_np,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    metrics["num_classes_in_test"] = int(len(present_labels))
    metrics["macro_f1"] = float(report["macro avg"]["f1-score"])
    metrics["weighted_f1"] = float(report["weighted avg"]["f1-score"])

    cm = confusion_matrix(targets_np, preds_np, labels=present_labels)
    plot_confusion_matrix(cm, present_names, viz_dir / "cnn_lstm_confusion_matrix.png")

    out_path = results_dir / "cnn_lstm_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete on device: %s", device)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Top-3: %.4f", metrics["top_3_accuracy"])
    logger.info("Top-5: %.4f", metrics["top_5_accuracy"])
    logger.info("MRR: %.4f", metrics["mrr"])
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    evaluate()