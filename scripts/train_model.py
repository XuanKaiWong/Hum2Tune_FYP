"""Accuracy-focused training pipeline for Hum2Tune CNN-LSTM."""

from __future__ import annotations

import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(stream=_utf8_stdout),
    ],
)
logger = logging.getLogger(__name__)

SEED = 42

MODEL_REGISTRY: dict[str, Any] = {
    "cnn_lstm": CNNLSTMModel,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config() -> dict:
    """
    Load config/training_config.yaml if available.
    Then upgrade it toward an accuracy-focused setup for larger song counts.
    """
    config_path = project_root / "config" / "training_config.yaml"
    loaded_conf: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "training" in loaded and isinstance(loaded["training"], dict):
                loaded_conf = loaded["training"]

    conf = {
        "batch_size": 8,
        "learning_rate": 1e-3,
        "epochs": 120,
        "early_stopping_patience": 20,
        "gradient_clip": 1.0,
        "weight_decay": 1e-4,
        "input_channels": 13,
        "seed": 42,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
        "norm_type": "group",
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
        "label_smoothing": 0.0,
        "num_workers": 0,
        "use_weighted_sampler": True,
        "use_class_weights": True,
    }

    # Let user config override, then lift clearly weak small-data values.
    conf.update(loaded_conf)

    conf["batch_size"] = max(int(conf.get("batch_size", 8)), 8)
    conf["learning_rate"] = max(float(conf.get("learning_rate", 1e-3)), 1e-3)
    conf["epochs"] = max(int(conf.get("epochs", 120)), 120)
    conf["early_stopping_patience"] = max(int(conf.get("early_stopping_patience", 20)), 20)
    conf["weight_decay"] = min(float(conf.get("weight_decay", 1e-4)), 1e-4)
    conf["hidden_size"] = max(int(conf.get("hidden_size", 128)), 128)
    conf["dropout"] = min(float(conf.get("dropout", 0.2)), 0.25)
    conf["label_smoothing"] = 0.0
    conf["scheduler_patience"] = max(int(conf.get("scheduler_patience", 5)), 5)
    conf["min_lr"] = min(float(conf.get("min_lr", 1e-6)), 1e-6)
    conf["use_weighted_sampler"] = bool(conf.get("use_weighted_sampler", True))
    conf["use_class_weights"] = bool(conf.get("use_class_weights", True))

    return conf


def build_model(model_name: str, num_classes: int, train_conf: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")

    if model_name == "cnn_lstm":
        return CNNLSTMModel(
            input_channels=int(train_conf.get("input_channels", 13)),
            num_classes=num_classes,
            hidden_size=int(train_conf.get("hidden_size", 128)),
            num_layers=int(train_conf.get("num_layers", 1)),
            dropout=float(train_conf.get("dropout", 0.2)),
            bidirectional=bool(train_conf.get("bidirectional", True)),
            use_attention=bool(train_conf.get("use_attention", True)),
            norm_type=str(train_conf.get("norm_type", "group")),
        )

    raise ValueError(f"Unsupported model '{model_name}'.")


def extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def save_training_curves(history: dict, save_dir: Path, model_name: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], marker="o", markersize=3, label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", markersize=3, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], marker="o", markersize=3, label="train")
    ax2.plot(epochs, history["val_acc"], marker="o", markersize=3, label="val")
    if history["val_acc"]:
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        best_acc = max(history["val_acc"])
        ax2.axvline(x=best_epoch, linestyle="--", linewidth=1, label=f"Best epoch: {best_epoch}")
        ax2.set_title(f"Accuracy (best val={best_acc:.3f})")
    else:
        ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"{model_name} training curves")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_epoch(model, loader, device, criterion):
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = extract_logits(model(X))
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def build_balancing(train_dataset, num_classes: int):
    targets = np.array([int(train_dataset[i][1]) for i in range(len(train_dataset))], dtype=np.int64)
    class_counts = np.bincount(targets, minlength=num_classes)

    # Prevent division by zero
    class_weights_np = 1.0 / np.maximum(class_counts, 1)
    class_weights_np = class_weights_np / class_weights_np.mean()

    sample_weights = class_weights_np[targets]

    return targets, class_counts, class_weights_np, sample_weights


def safe_load_resume(model, optimizer, resume_path: Path, device: torch.device, history: dict):
    if not resume_path.exists():
        return 1, 0.0, 0, history

    try:
        logger.info("Resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        no_improve_count = ckpt.get("no_improve_count", 0)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0) + 1
        return start_epoch, best_val_acc, no_improve_count, history
    except Exception as exc:
        logger.warning("Could not resume from checkpoint, starting fresh: %s", exc)
        return 1, 0.0, 0, history


def safe_load_best_weights(model, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        return

    try:
        logger.info("Loading previous best weights from %s", checkpoint_path)
        checkpoint = torch_load(checkpoint_path, device)
        state = extract_state_dict(checkpoint)
        model.load_state_dict(state)
    except Exception as exc:
        logger.warning("Could not load previous best weights, starting fresh: %s", exc)


def train_pipeline(model_name: str = "cnn_lstm") -> None:
    train_conf = load_config()
    seed = int(train_conf.get("seed", SEED))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    data_dir = project_root / "data" / "processed" / "datasets"
    class_map = load_class_map(data_dir / "classes.json")
    num_classes = len(class_map)

    train_dataset = resolve_split_dataset(data_dir, "train_data")
    val_dataset = resolve_split_dataset(data_dir, "val_data")

    batch_size = int(train_conf.get("batch_size", 8))
    num_workers = int(train_conf.get("num_workers", 0))

    targets, class_counts, class_weights_np, sample_weights = build_balancing(train_dataset, num_classes)

    logger.info("Min class count: %d | Max class count: %d | Mean class count: %.2f",
                int(class_counts.min()), int(class_counts.max()), float(class_counts.mean()))

    sampler = None
    shuffle = True
    if bool(train_conf.get("use_weighted_sampler", True)):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        logger.info("Using WeightedRandomSampler for class balancing")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(model_name, num_classes=num_classes, train_conf=train_conf).to(device)

    logger.info("Model: %s", model_name)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))
    logger.info("Number of classes: %d", num_classes)
    logger.info("Final training config: %s", json.dumps(train_conf, indent=2))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_conf.get("learning_rate", 1e-3)),
        weight_decay=float(train_conf.get("weight_decay", 1e-4)),
    )

    if bool(train_conf.get("use_class_weights", True)):
        class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(train_conf.get("label_smoothing", 0.0)),
        )
        logger.info("Using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(train_conf.get("label_smoothing", 0.0))
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(train_conf.get("scheduler_patience", 5)),
        factor=float(train_conf.get("scheduler_factor", 0.5)),
        min_lr=float(train_conf.get("min_lr", 1e-6)),
    )

    model_dir = project_root / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "best_model.pth"
    resume_path = model_dir / "resume_checkpoint.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Try loading old best weights only if resume checkpoint does not load
    safe_load_best_weights(model, checkpoint_path, device)

    start_epoch = 1
    best_val_acc = 0.0
    no_improve_count = 0
    start_epoch, best_val_acc, no_improve_count, history = safe_load_resume(
        model, optimizer, resume_path, device, history
    )

    epochs = int(train_conf.get("epochs", 120))
    patience = int(train_conf.get("early_stopping_patience", 20))
    grad_clip = float(train_conf.get("gradient_clip", 1.0))

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = extract_logits(model(X))
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total_train += y.size(0)
            correct_train += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        val_loss, val_acc = evaluate_epoch(model, val_loader, device, criterion)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %03d/%03d | train_loss %.4f | train_acc %.4f | val_loss %.4f | val_acc %.4f | lr %.6f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, current_lr
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0

            # Save pure state_dict for compatibility with existing app/eval code
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model saved to %s", checkpoint_path)
        else:
            no_improve_count += 1
            logger.info("No improvement count: %d/%d", no_improve_count, patience)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "no_improve_count": no_improve_count,
                "history": history,
                "model_name": model_name,
                "config": train_conf,
            },
            resume_path,
        )

        if no_improve_count >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    save_training_curves(history, project_root / "results" / "visualizations", model_name)

    results_dir = project_root / "results" / "evaluations"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{model_name}_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "model": model_name,
                "epochs_trained": len(history["train_loss"]),
                "best_val_accuracy": best_val_acc,
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else None,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else None,
                "config": train_conf,
            },
            f,
            indent=2,
        )

    logger.info("Training complete. Best validation accuracy: %.2f%%", best_val_acc * 100)


if __name__ == "__main__":
    train_pipeline("cnn_lstm")