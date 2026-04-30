from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fyp_title11.data.dataset import load_class_map, resolve_split_dataset
from fyp_title11.evaluation.metrics import compute_all_metrics
from fyp_title11.models.cnn_lstm import CNNLSTMModel
from fyp_title11.models.audio_transformer import AudioTransformer

log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

_utf8_stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
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

SUPPORTED_MODELS = ("cnn_lstm", "audio_transformer")


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

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
            if isinstance(loaded, dict) and "training" in loaded:
                return loaded["training"]
    return {"input_channels": 39, "hidden_size": 128, "num_layers": 1,
            "dropout": 0.35, "bidirectional": True, "use_attention": True,
            "norm_type": "group", "batch_size": 8, "num_workers": 0}


def build_model(model_name: str, num_classes: int, conf: dict[str, Any]):
    """Instantiate the requested model architecture from config."""
    if model_name == "cnn_lstm":
        return CNNLSTMModel(
            input_channels=int(conf.get("input_channels", 39)),
            num_classes=num_classes,
            hidden_size=int(conf.get("hidden_size", 128)),
            num_layers=int(conf.get("num_layers", 1)),
            dropout=float(conf.get("dropout", 0.35)),
            bidirectional=bool(conf.get("bidirectional", True)),
            use_attention=bool(conf.get("use_attention", True)),
            attn_hidden_dim=int(conf.get("attn_hidden_dim", 32)),
            norm_type=str(conf.get("norm_type", "group")),
        )
    if model_name == "audio_transformer":
        t = conf.get("transformer", {})
        return AudioTransformer(
            input_channels=int(conf.get("input_channels", 39)),
            num_classes=num_classes,
            d_model=int(t.get("d_model", 128)),
            nhead=int(t.get("nhead", 4)),
            num_layers=int(t.get("num_layers", 2)),
            dim_feedforward=int(t.get("dim_feedforward", 256)),
            dropout=float(t.get("dropout", 0.3)),
        )
    raise ValueError(f"Unsupported model '{model_name}'. Choose from: {SUPPORTED_MODELS}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(8, len(class_names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, pad=12)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ticks = range(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved -> %s", save_path)


# -----------------------------------------------------------------------------
#  Main evaluation loop
# -----------------------------------------------------------------------------

def evaluate(model_name: str = "cnn_lstm", curriculum: bool = False) -> None:
    """Evaluate a trained model on the test (or val) split.

    Args:
        model_name:  Architecture to evaluate ('cnn_lstm' or 'audio_transformer').
        curriculum:  If True, load the curriculum checkpoint (best_model_curriculum.pth)
                     instead of the standard checkpoint (best_model.pth).
                     This keeps standard and curriculum results in separate files
                     so the comparison table is never accidentally overwritten.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {SUPPORTED_MODELS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    conf = load_training_config()
    logger.info("Evaluating '%s' on device: %s (AMP: %s)", model_name, device, use_amp)

    data_dir = project_root / "data" / "processed" / "datasets"
    results_dir = project_root / "results" / "evaluations"
    viz_dir = project_root / "results" / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(data_dir / "classes.json")
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(class_names)
    logger.info("Classes: %d", num_classes)

    # Use test split; fall back to val if test is missing (small datasets)
    try:
        test_dataset = resolve_split_dataset(data_dir, "test_data")
        split_used = "test"
    except FileNotFoundError:
        logger.warning("test_data not found -- evaluating on val_data instead")
        test_dataset = resolve_split_dataset(data_dir, "val_data")
        split_used = "val"

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(conf.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(conf.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # Select checkpoint based on whether curriculum mode was requested.
    # Using separate filenames prevents standard and curriculum results from
    # overwriting each other in results/evaluations/.
    if curriculum:
        model_path = project_root / "models" / model_name / "best_model_curriculum.pth"
        result_stem = f"{model_name}_curriculum"
    else:
        model_path = project_root / "models" / model_name / "best_model.pth"
        result_stem = model_name

    if not model_path.exists():
        hint = (
            f"python main.py train --model {model_name} --curriculum"
            if curriculum else
            f"python main.py train --model {model_name}"
        )
        raise FileNotFoundError(
            f"No checkpoint found at {model_path}.\n"
            f"Train first: {hint}"
        )

    logger.info("Loading checkpoint: %s", model_path)
    model = build_model(model_name, num_classes, conf).to(device)
    checkpoint = torch_load(model_path, device)
    state = (checkpoint.get("model_state") or checkpoint.get("model_state_dict")
             or checkpoint.get("state_dict") or checkpoint
             if isinstance(checkpoint, dict) else checkpoint)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded weights from %s", model_path)

    # Collect predictions
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            from torch.cuda.amp import autocast
            with autocast(enabled=use_amp):
                logits = extract_logits(model(X))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_targets.extend(y.numpy().tolist())

    if not all_targets:
        raise ValueError("Evaluation set is empty. Run prepare_dataset.py first.")

    targets_np = np.asarray(all_targets, dtype=np.int64)
    preds_np = np.asarray(all_preds, dtype=np.int64)
    probs_np = np.vstack(all_probs)

    # Compute all metrics using the shared metrics module
    metrics = compute_all_metrics(
        y_true=targets_np,
        y_pred=preds_np,
        y_prob=probs_np,
        class_names=class_names,
        top_k_values=[1, 3, 5],
    )

    # Add evaluation metadata
    metrics["model"] = model_name
    metrics["curriculum"] = curriculum
    metrics["checkpoint"] = str(model_path)
    metrics["split"] = split_used
    metrics["num_classes"] = num_classes
    metrics["num_samples"] = int(len(targets_np))

    # Log headline metrics
    logger.info("-- Evaluation Results --------------------------")
    for key in ("accuracy", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
                "mrr", "map_at_10", "ndcg_at_10", "macro_f1"):
        if key in metrics:
            logger.info("  %-22s %.4f", key + ":", metrics[key])
    logger.info("------------------------------------------------")

    # Confusion matrix -- already shaped (num_classes, num_classes) thanks to
    # compute_confusion_matrix(num_classes=...) in compute_all_metrics().
    # Row-normalise here for readability in the plot.
    cm_raw = np.array(metrics["confusion_matrix"], dtype=float)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums == 0, 0.0, cm_raw / row_sums)
    plot_confusion_matrix(
        cm_norm, class_names,
        viz_dir / f"{result_stem}_confusion_matrix.png"
    )

    # Most confused pairs (useful for dissertation error analysis section)
    if "most_confused_pairs" in metrics:
        logger.info("Top confused pairs (true -> predicted):")
        for true_c, pred_c, count in metrics["most_confused_pairs"][:5]:
            logger.info("  %s -> %s  (%d samples)", true_c, pred_c, count)

    # Save JSON report with a stem that distinguishes standard from curriculum,
    # preventing accidental overwrites when running both variants.
    # standard CNN-LSTM -> cnn_lstm_evaluation_results.json
    # curriculum CNN-LSTM -> cnn_lstm_curriculum_evaluation_results.json
    report_metrics = {k: v for k, v in metrics.items() if k not in ("report_str",)}
    out_path = results_dir / f"{result_stem}_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_metrics, f, indent=2, default=str)
    logger.info("Results saved -> %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hum2Tune model")
    parser.add_argument(
        "--model", type=str, default="cnn_lstm",
        choices=list(SUPPORTED_MODELS),
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--curriculum", action="store_true", default=False,
        help=(
            "Load the curriculum checkpoint (best_model_curriculum.pth) instead of "
            "the standard checkpoint (best_model.pth). Results are saved to a "
            "separate file so the standard baseline is never overwritten."
        ),
    )
    args = parser.parse_args()
    evaluate(model_name=args.model, curriculum=args.curriculum)
