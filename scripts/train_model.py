from __future__ import annotations

import io
import json
import logging
import random
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fyp_title11.data.dataset import load_class_map, resolve_split_dataset
from fyp_title11.models.cnn_lstm import CNNLSTMModel
from fyp_title11.models.audio_transformer import AudioTransformer

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
    "cnn_lstm":          CNNLSTMModel,
    "audio_transformer": AudioTransformer,
}


def get_device() -> torch.device:
    """Return the best available device and log GPU info if available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
        logger.info("CUDA version: %s", torch.version.cuda)
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected -- training on CPU (will be slow)")
    return device


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True disables cudnn.benchmark; use only when
        # exact reproducibility matters more than speed.
        torch.backends.cudnn.deterministic = False
        # benchmark=True lets cuDNN auto-tune the fastest conv algorithm
        # for the fixed input sizes used in this project. Gives 10-30%
        # speedup after the first epoch warmup.
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config() -> dict:
    """Load training configuration from config/training_config.yaml.

    The YAML file is the single source of truth. Defaults are only used
    when a key is entirely absent from the file -- not to override explicit
    user choices. This enables clean hyperparameter search.
    """
    config_path = project_root / "config" / "training_config.yaml"
    loaded_conf: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "training" in loaded and isinstance(loaded["training"], dict):
                loaded_conf = loaded["training"]

    # Sensible defaults -- only applied when the key is absent from YAML.
    # num_workers: auto = half the CPU count, capped at 8, 0 on Windows
    # (Windows DataLoader with spawn can silently hang with num_workers > 0).
    import os as _os, platform as _platform
    _auto_workers = min(8, max(0, (_os.cpu_count() or 1) // 2))
    if _platform.system() == "Windows":
        _auto_workers = 0

    defaults = {
        "batch_size": 8,
        "learning_rate": 3e-4,
        "epochs": 120,
        "early_stopping_patience": 20,
        "gradient_clip": 1.0,
        "weight_decay": 1e-3,
        "input_channels": 39,   # n_mfcc(13) * 3 (MFCC + delta + delta-delta)
        "seed": 42,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "bidirectional": True,
        "use_attention": True,
        "attn_hidden_dim": 32,
        "norm_type": "group",
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
        "label_smoothing": 0.1,
        "num_workers": _auto_workers,
        "use_weighted_sampler": True,
        "use_class_weights": True,
        # mixed_precision: uses torch.cuda.amp for 1.5-2x speedup on GPU
        # with Tensor Cores (RTX/Volta and newer). Automatically disabled
        # when running on CPU.
        "mixed_precision": True,
    }
    # YAML values always win over defaults.
    conf = {**defaults, **loaded_conf}
    return conf


def build_model(model_name: str, num_classes: int, train_conf: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")

    if model_name == "cnn_lstm":
        return CNNLSTMModel(
            input_channels=int(train_conf.get("input_channels", 39)),
            num_classes=num_classes,
            hidden_size=int(train_conf.get("hidden_size", 128)),
            num_layers=int(train_conf.get("num_layers", 1)),
            dropout=float(train_conf.get("dropout", 0.3)),
            bidirectional=bool(train_conf.get("bidirectional", True)),
            use_attention=bool(train_conf.get("use_attention", True)),
            attn_hidden_dim=int(train_conf.get("attn_hidden_dim", 32)),
            norm_type=str(train_conf.get("norm_type", "group")),
        )

    if model_name == "audio_transformer":
        # Read Transformer settings from the nested 'transformer' block in
        # training_config.yaml -- consistent with evaluate.py and predict.py.
        t = train_conf.get("transformer", {})
        return AudioTransformer(
            input_channels=int(train_conf.get("input_channels", 39)),
            num_classes=num_classes,
            d_model=int(t.get("d_model", 128)),
            nhead=int(t.get("nhead", 4)),
            num_layers=int(t.get("num_layers", 2)),
            dim_feedforward=int(t.get("dim_feedforward", 256)),
            dropout=float(t.get("dropout", 0.3)),
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


def evaluate_epoch(model, loader, device, criterion, use_amp: bool = False):
    model.eval()

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    running_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = extract_logits(model(X))
                loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            total += y.size(0)

            correct_top1 += (logits.argmax(dim=1) == y).sum().item()

            k = min(5, logits.size(1))
            top5_preds = logits.topk(k, dim=1).indices
            correct_top5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = running_loss / max(total, 1)
    top1_acc = correct_top1 / max(total, 1)
    top5_acc = correct_top5 / max(total, 1)
    return avg_loss, top1_acc, top5_acc


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
        ckpt = torch_load(resume_path, device)   # uses weights_only=True where supported
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

    device = get_device()
    logger.info("Training on device: %s", device)

    # Mixed-precision training (AMP) -- speeds up GPU training by 1.5-2x
    # using float16 for forward/backward and float32 for the optimizer step.
    # Automatically no-op on CPU.
    use_amp = bool(train_conf.get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info("Mixed-precision (AMP): %s", "enabled" if use_amp else "disabled")

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
            weights=np.asarray(sample_weights, dtype=np.float64).tolist(),
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
        pin_memory=(device.type == "cuda"),   # page-lock host memory for faster H->D transfer
        persistent_workers=(num_workers > 0), # keep worker processes alive between epochs
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    logger.info("DataLoader num_workers: %d | pin_memory: %s | persistent_workers: %s",
                num_workers, device.type == "cuda", num_workers > 0)

    model = build_model(model_name, num_classes=num_classes, train_conf=train_conf).to(device)

    logger.info("Model: %s", model_name)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))  # type: ignore[arg-type]
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
        "val_top5_acc": [],
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
            # non_blocking=True overlaps H->D transfer with GPU compute
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # autocast: runs forward pass in float16 on GPU (no-op on CPU)
            with autocast(enabled=use_amp):
                logits = extract_logits(model(X))
                loss = criterion(logits, y)

            # scaler: scales the loss to prevent float16 underflow,
            # then unscales gradients before the optimizer step.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            total_train += y.size(0)
            correct_train += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        val_loss, val_acc, val_top5_acc = evaluate_epoch(model, val_loader, device, criterion, use_amp)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5_acc"].append(val_top5_acc)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %03d/%03d | train_loss %.4f | train_acc %.4f | val_loss %.4f | val_acc %.4f | val_top5 %.4f | lr %.6f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, val_top5_acc, current_lr
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


# -----------------------------------------------------------------------------
#  Curriculum Learning pipeline
# -----------------------------------------------------------------------------

def curriculum_train_pipeline(model_name: str = "cnn_lstm") -> None:
    """Two-stage curriculum learning training pipeline.

    Addresses RQ4: "Can curriculum learning improve model robustness and
    generalisation in cross-domain humming recognition tasks?"

    Curriculum strategy (Bengio et al., 2009):
        Stage 1 -- Easy examples (vocal-only):
            Train on the vocal-isolated split (data/processed/vocal_only/).
            The model learns the fundamental concept of melody matching without
            polyphonic interference. This mirrors how humans learn to sing
            before playing instruments.

        Stage 2 -- Full complexity (polyphonic):
            Fine-tune on the full polyphonic training set. The model now has
            a strong melodic prior from Stage 1 and can generalise better
            to the polyphonic target domain.

    Expected data layout:
        data/processed/vocal_only/train_data_meta.json   <- Stage 1 training set
        data/processed/datasets/train_data_meta.json     <- Stage 2 training set
        data/processed/datasets/val_data_meta.json       <- shared validation set

    If the vocal-only split does not exist, falls back to standard training
    so this function is always safe to call.
    """
    train_conf = load_config()
    seed = int(train_conf.get("seed", SEED))
    set_seed(seed)

    device = get_device()
    logger.info("[Curriculum] Training on device: %s", device)

    use_amp = bool(train_conf.get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info("[Curriculum] Mixed-precision (AMP): %s", "enabled" if use_amp else "disabled")

    data_dir = project_root / "data" / "processed" / "datasets"
    vocal_only_dir = project_root / "data" / "processed" / "vocal_only"

    class_map = load_class_map(data_dir / "classes.json")
    num_classes = len(class_map)

    val_dataset = resolve_split_dataset(data_dir, "val_data")
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_conf.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(train_conf.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(train_conf.get("num_workers", 0)) > 0),
    )

    model = build_model(model_name, num_classes=num_classes, train_conf=train_conf).to(device)
    model_dir = project_root / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # -- Define curriculum stages ------------------------------------------
    # Each stage can override training parameters to control difficulty.
    stages = []

    vocal_only_meta = vocal_only_dir / "train_data_meta.json"
    if vocal_only_meta.exists():
        stages.append({
            "name": "stage1_vocal_only",
            "data_dir": vocal_only_dir,
            "stem": "train_data",
            "epochs": max(int(train_conf.get("epochs", 120)) // 3, 20),
            "label_smoothing": 0.0,    # strict learning from clean data
            "lr_factor": 1.0,
        })
        logger.info("[Curriculum] Vocal-only split found -- Stage 1 enabled.")
    else:
        logger.warning(
            "[Curriculum] Vocal-only split not found at %s. "
            "Skipping Stage 1 and running standard training only.",
            vocal_only_dir,
        )

    stages.append({
        "name": "stage2_polyphonic",
        "data_dir": data_dir,
        "stem": "train_data",
        "epochs": int(train_conf.get("epochs", 120)),
        "label_smoothing": 0.1,    # slight smoothing for noisy polyphonic data
        "lr_factor": 0.3,          # lower LR for fine-tuning stage
    })

    best_val_acc_overall = 0.0
    checkpoint_path = model_dir / "best_model_curriculum.pth"
    history_all: dict = {"stage": [], "epoch": [], "train_loss": [],
                         "train_acc": [], "val_loss": [], "val_acc": [],
                         "val_top5_acc": []}

    for stage in stages:
        logger.info("\n%s\n[Curriculum] Starting %s (%d epochs)\n%s",
                    "=" * 60, stage["name"], stage["epochs"], "=" * 60)

        # Load this stage's training data
        stage_train_dataset = resolve_split_dataset(stage["data_dir"], stage["stem"])
        _, _, class_weights_np, sample_weights = build_balancing(stage_train_dataset, num_classes)

        sampler = WeightedRandomSampler(
            weights=np.asarray(sample_weights, dtype=np.float64).tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        _nw = int(train_conf.get("num_workers", 0))
        stage_loader = DataLoader(
            stage_train_dataset,
            batch_size=int(train_conf.get("batch_size", 8)),
            sampler=sampler,
            shuffle=False,
            num_workers=_nw,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(_nw > 0),
        )

        # Reinitialise optimiser for each stage (allows LR adjustment)
        stage_lr = float(train_conf.get("learning_rate", 3e-4)) * stage["lr_factor"]
        optimizer = optim.AdamW(
            model.parameters(),
            lr=stage_lr,
            weight_decay=float(train_conf.get("weight_decay", 1e-3)),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(train_conf.get("scheduler_patience", 5)),
            factor=float(train_conf.get("scheduler_factor", 0.5)),
            min_lr=float(train_conf.get("min_lr", 1e-6)),
        )
        class_weights_tensor = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=stage["label_smoothing"],
        )

        patience = int(train_conf.get("early_stopping_patience", 20))
        grad_clip = float(train_conf.get("gradient_clip", 1.0))
        no_improve = 0
        best_stage_val_acc = 0.0

        for epoch in range(1, stage["epochs"] + 1):
            model.train()
            running_loss = 0.0
            total_train = 0
            correct_train = 0

            for X, y in stage_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = extract_logits(model(X))
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * y.size(0)
                total_train += y.size(0)
                correct_train += (logits.argmax(dim=1) == y).sum().item()

            train_loss = running_loss / max(total_train, 1)
            train_acc = correct_train / max(total_train, 1)
            val_loss, val_acc, val_top5 = evaluate_epoch(model, val_loader, device, criterion, use_amp)
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "[%s] Epoch %03d | train_loss %.4f | train_acc %.4f "
                "| val_acc %.4f | val_top5 %.4f | lr %.2e",
                stage["name"], epoch, train_loss, train_acc, val_acc, val_top5, current_lr,
            )

            history_all["stage"].append(stage["name"])
            history_all["epoch"].append(epoch)
            history_all["train_loss"].append(train_loss)
            history_all["train_acc"].append(train_acc)
            history_all["val_loss"].append(val_loss)
            history_all["val_acc"].append(val_acc)
            history_all["val_top5_acc"].append(val_top5)

            if val_acc > best_val_acc_overall:
                best_val_acc_overall = val_acc
                best_stage_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("[Curriculum] New best model saved (val_acc=%.4f)", val_acc)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("[Curriculum] Early stopping at epoch %d", epoch)
                    break

        logger.info("[Curriculum] %s complete. Best val_acc: %.4f",
                    stage["name"], best_stage_val_acc)

    # Save curriculum training summary
    results_dir = project_root / "results" / "evaluations"
    results_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(results_dir / f"{model_name}_curriculum_summary.json", "w", encoding="utf-8") as f:
        _json.dump({
            "model": model_name,
            "stages": [s["name"] for s in stages],
            "best_val_accuracy": best_val_acc_overall,
            "history_length": len(history_all["epoch"]),
        }, f, indent=2)

    logger.info("\n[Curriculum] Training complete. Best overall val_acc: %.2f%%",
                best_val_acc_overall * 100)


# -----------------------------------------------------------------------------
#  CLI entrypoint -- exactly one __main__ block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        description="Train a Hum2Tune melody recognition model.",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=sorted(MODEL_REGISTRY),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help=(
            "Use the two-stage curriculum learning pipeline (vocal-only -> polyphonic). "
            "Addresses RQ4. Requires data/processed/vocal_only/ split to exist."
        ),
    )
    args = parser.parse_args()

    if args.curriculum:
        curriculum_train_pipeline(args.model)
    else:
        train_pipeline(args.model)
