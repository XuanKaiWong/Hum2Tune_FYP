from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fyp_title11.models.dual_encoder import (  # noqa: E402
    DualEncoder,
    MultiPositiveContrastiveLoss,
    compute_retrieval_metrics,
)
from fyp_title11.tokenization.feature_extractor import FeatureExtractor  # noqa: E402

LOGGER = logging.getLogger("train_dual_encoder")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_dual_encoder.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass(frozen=True)
class PairRecord:
    query_id: str
    song_id: str
    title: str
    query_path: Path
    reference_path: Path
    label: int
    split: str
    singer_id: str = "unknown"
    source: str = "unknown"


class FeatureCache:
    """Disk-backed cache for FeatureExtractor outputs.

    Feature extraction dominates the runtime. Caching makes repeated training,
    evaluation and hyperparameter tuning much faster while still invalidating
    entries when files change.
    """

    def __init__(self, cache_dir: Path, extractor: FeatureExtractor) -> None:
        self.cache_dir = cache_dir
        self.extractor = extractor
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: dict[Path, list[np.ndarray]] = {}

    @staticmethod
    def _signature(path: Path) -> str:
        stat = path.stat()
        payload = f"{path.resolve()}|{int(stat.st_mtime)}|{stat.st_size}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, path: Path) -> Path:
        return self.cache_dir / f"{self._signature(path)}.npz"

    def get(self, path: Path) -> list[np.ndarray]:
        path = path.resolve()
        if path in self.memory:
            return self.memory[path]

        cache_path = self._cache_path(path)
        if cache_path.exists():
            try:
                payload = np.load(cache_path, allow_pickle=False)
                arr = payload["features"].astype(np.float32)
                features = [arr[i] for i in range(arr.shape[0])]
                self.memory[path] = features
                return features
            except Exception as exc:
                LOGGER.warning("Ignoring corrupt feature cache %s: %s", cache_path.name, exc)

        features = self.extractor.process_file(path)
        features = [np.asarray(x, dtype=np.float32) for x in features if x is not None]
        if not features:
            raise ValueError(f"No features extracted from {path}")

        np.savez_compressed(cache_path, features=np.stack(features).astype(np.float32))
        self.memory[path] = features
        return features


def resolve_path(path_value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path from a manifest, accepting both relative and absolute paths."""
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else (root / path).resolve()


def load_pair_records(
    manifest_dir: Path = PROJECT_ROOT / "data" / "manifests",
    *,
    reference_column: str = "reference_path",
) -> tuple[list[PairRecord], dict[int, str]]:
    """Load and validate paired query/reference records from manifests."""
    songs_path = manifest_dir / "songs.csv"
    queries_path = manifest_dir / "queries.csv"
    splits_path = manifest_dir / "splits.csv"

    if not songs_path.exists():
        raise FileNotFoundError(f"Missing songs manifest: {songs_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries manifest: {queries_path}")

    songs = pd.read_csv(songs_path, dtype=str).fillna("")
    queries = pd.read_csv(queries_path, dtype=str).fillna("")

    required_songs = {"song_id", "title"}
    required_queries = {"query_id", "song_id", "query_path"}
    missing_songs = required_songs - set(songs.columns)
    missing_queries = required_queries - set(queries.columns)
    if missing_songs:
        raise ValueError(f"songs.csv missing columns: {sorted(missing_songs)}")
    if missing_queries:
        raise ValueError(f"queries.csv missing columns: {sorted(missing_queries)}")

    if reference_column not in songs.columns:
        fallback_columns = [c for c in ("vocals_path", "reference_path", "original_path") if c in songs.columns]
        if not fallback_columns:
            raise ValueError(
                f"songs.csv must contain '{reference_column}' or one of "
                "vocals_path/reference_path/original_path"
            )
        reference_column = fallback_columns[0]
        LOGGER.warning("Using reference column '%s'", reference_column)

    if "split" not in queries.columns:
        if splits_path.exists():
            splits = pd.read_csv(splits_path, dtype=str).fillna("")
            if not {"query_id", "split"}.issubset(splits.columns):
                raise ValueError("splits.csv must contain query_id and split columns")
            queries = queries.merge(splits[["query_id", "split"]], on="query_id", how="left")
        else:
            queries["split"] = "train"

    songs = songs.drop_duplicates("song_id").copy()
    label_map = {song_id: idx for idx, song_id in enumerate(sorted(songs["song_id"].unique()))}
    idx_to_title = {
        label_map[row.song_id]: str(row.title)
        for row in songs.itertuples(index=False)
    }

    merged = queries.merge(
        songs[["song_id", "title", reference_column]],
        on="song_id",
        how="left",
        validate="many_to_one",
    )

    records: list[PairRecord] = []
    skipped = 0
    for row in merged.itertuples(index=False):
        song_id = str(row.song_id)
        query_id = str(row.query_id)
        title = str(row.title)
        split = str(getattr(row, "split", "train") or "train").lower().strip()
        if split not in {"train", "val", "test"}:
            split = "train"

        query_path = resolve_path(getattr(row, "query_path"))
        reference_path = resolve_path(getattr(row, reference_column))

        if not query_path.exists():
            LOGGER.warning("Skipping %s because query file is missing: %s", query_id, query_path)
            skipped += 1
            continue
        if not reference_path.exists():
            LOGGER.warning("Skipping %s because reference file is missing: %s", query_id, reference_path)
            skipped += 1
            continue
        if query_path.suffix.lower() not in AUDIO_EXTENSIONS:
            LOGGER.warning("Skipping %s because query extension is unsupported: %s", query_id, query_path.suffix)
            skipped += 1
            continue

        records.append(
            PairRecord(
                query_id=query_id,
                song_id=song_id,
                title=title,
                query_path=query_path,
                reference_path=reference_path,
                label=label_map[song_id],
                split=split,
                singer_id=str(getattr(row, "singer_id", "unknown") or "unknown"),
                source=str(getattr(row, "source", "unknown") or "unknown"),
            )
        )

    if not records:
        raise RuntimeError(
            "No valid paired records were loaded. Check data/manifests/*.csv and audio paths."
        )

    LOGGER.info(
        "Loaded %d paired records from manifests (%d skipped). Split counts: %s",
        len(records),
        skipped,
        pd.Series([r.split for r in records]).value_counts().to_dict(),
    )
    return records, idx_to_title


class ManifestPairDataset(Dataset):
    """Dataset of query feature windows paired with matching reference windows."""

    def __init__(
        self,
        records: Iterable[PairRecord],
        feature_cache: FeatureCache,
        *,
        split: str,
        seed: int = 42,
    ) -> None:
        self.records = [r for r in records if r.split == split]
        self.cache = feature_cache
        self.seed = int(seed)
        self.split = split
        self.examples: list[tuple[np.ndarray, Path, int, str, str]] = []

        if not self.records:
            raise ValueError(f"No records found for split='{split}'")

        for record in self.records:
            try:
                query_features = self.cache.get(record.query_path)
            except Exception as exc:
                LOGGER.warning("Skipping query %s during dataset build: %s", record.query_path.name, exc)
                continue

            for q_feat in query_features:
                self.examples.append(
                    (q_feat, record.reference_path, record.label, record.query_id, record.title)
                )

        if not self.examples:
            raise RuntimeError(f"No usable feature windows found for split='{split}'")

        self.reference_paths_by_label: dict[int, Path] = {}
        self.title_by_label: dict[int, str] = {}
        for record in self.records:
            self.reference_paths_by_label.setdefault(record.label, record.reference_path)
            self.title_by_label.setdefault(record.label, record.title)

        LOGGER.info(
            "%s dataset: %d records -> %d query windows -> %d reference classes",
            split,
            len(self.records),
            len(self.examples),
            len(self.reference_paths_by_label),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        q_feat, reference_path, label, query_id, title = self.examples[index]
        reference_features = self.cache.get(reference_path)

        # Deterministic cycling gives reference-window diversity without making
        # the dataset non-reproducible across runs.
        ref_index = (index * 9973 + self.seed) % len(reference_features)
        r_feat = reference_features[ref_index]

        return (
            torch.as_tensor(q_feat, dtype=torch.float32),
            torch.as_tensor(r_feat, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.long),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def collate_pairs(batch):
    q, r, y = zip(*batch)
    return torch.stack(q), torch.stack(r), torch.stack(y)


@torch.no_grad()
def mean_reference_embedding(
    model: DualEncoder,
    cache: FeatureCache,
    reference_path: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features = cache.get(reference_path)
    embeddings = []
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.as_tensor(np.stack(features[start:start + batch_size]), dtype=torch.float32, device=device)
        embeddings.append(model.encode_reference(batch).cpu())
    return torch.nn.functional.normalize(torch.cat(embeddings, dim=0).mean(dim=0), dim=-1)


@torch.no_grad()
def evaluate_retrieval(
    model: DualEncoder,
    dataset: ManifestPairDataset,
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict:
    """Evaluate query-to-reference retrieval on a manifest split."""
    model.eval()

    labels_sorted = sorted(dataset.reference_paths_by_label)
    ref_embeddings = []
    for label in labels_sorted:
        emb = mean_reference_embedding(
            model,
            dataset.cache,
            dataset.reference_paths_by_label[label],
            device,
            batch_size=batch_size,
        )
        ref_embeddings.append(emb)
    ref_matrix = torch.stack(ref_embeddings).to(device)

    ranks: list[int | None] = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    label_to_column = {label: idx for idx, label in enumerate(labels_sorted)}

    for q_x, _, labels in loader:
        q_x = q_x.to(device)
        q_emb = model.encode_query(q_x)
        scores = q_emb @ ref_matrix.T
        order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()

        for row_order, label in zip(order, labels.numpy()):
            true_col = label_to_column[int(label)]
            positions = np.where(row_order == true_col)[0]
            ranks.append(int(positions[0]) + 1 if len(positions) else None)

    metrics = compute_retrieval_metrics(ranks)
    return {
        "count": metrics.count,
        "top1": metrics.top1,
        "top3": metrics.top3,
        "top5": metrics.top5,
        "mrr": metrics.mrr,
    }


def train_dual_encoder(
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 3e-4,
    embedding_dim: int = 128,
    temperature: float = 0.1,
    weight_decay: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
    manifest_dir: str | Path = PROJECT_ROOT / "data" / "manifests",
    reference_column: str = "reference_path",
) -> None:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    LOGGER.info("Device: %s | AMP: %s", device, use_amp)

    extractor = FeatureExtractor(config_path=str(PROJECT_ROOT / "config" / "audio_config.yaml"))
    cache = FeatureCache(PROJECT_ROOT / "outputs" / "dual_encoder_feature_cache", extractor)

    records, idx_to_title = load_pair_records(Path(manifest_dir), reference_column=reference_column)

    train_dataset = ManifestPairDataset(records, cache, split="train", seed=seed)
    val_split = "val" if any(r.split == "val" for r in records) else "train"
    val_dataset = ManifestPairDataset(records, cache, split=val_split, seed=seed + 1)

    input_channels = train_dataset[0][0].shape[0]
    model = DualEncoder(
        input_channels=int(input_channels),
        embedding_dim=int(embedding_dim),
        dropout=0.2,
    ).to(device)

    criterion = MultiPositiveContrastiveLoss(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pairs,
        drop_last=False,
    )

    model_dir = PROJECT_ROOT / "models" / "dual_encoder"
    result_dir = PROJECT_ROOT / "results" / "evaluations"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "best_model.pth"
    last_path = model_dir / "last_checkpoint.pth"
    class_map_path = model_dir / "classes.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_title.items()}, f, indent=2, ensure_ascii=False)

    best_mrr = -1.0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for q_x, r_x, labels in train_loader:
            q_x = q_x.to(device, non_blocking=True)
            r_x = r_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                q_emb, r_emb = model(q_x, r_x)
                loss = criterion(q_emb, r_emb, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * q_x.size(0)
            total_count += q_x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total_count, 1)
        val_metrics = evaluate_retrieval(model, val_dataset, device, batch_size=batch_size)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "val_mrr": val_metrics["mrr"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        LOGGER.info(
            "Epoch %03d/%03d | loss %.4f | val_top1 %.3f | val_top3 %.3f | "
            "val_mrr %.3f | lr %.2e",
            epoch,
            epochs,
            train_loss,
            val_metrics["top1"],
            val_metrics["top3"],
            val_metrics["mrr"],
            optimizer.param_groups[0]["lr"],
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "config": {
                    "input_channels": int(input_channels),
                    "embedding_dim": int(embedding_dim),
                    "temperature": float(temperature),
                    "reference_column": reference_column,
                },
                "history": history,
            },
            last_path,
        )

        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            LOGGER.info("Saved best dual encoder to %s (val_mrr=%.4f)", best_path, best_mrr)
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info("Early stopping after %d epochs without MRR improvement.", patience)
                break

    summary = {
        "model": "dual_encoder",
        "epochs_trained": len(history),
        "best_val_mrr": best_mrr,
        "final": history[-1] if history else None,
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "num_reference_classes": len(train_dataset.reference_paths_by_label),
        "manifest_dir": str(Path(manifest_dir)),
        "reference_column": reference_column,
    }
    with open(result_dir / "dual_encoder_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history).to_csv(result_dir / "dual_encoder_training_history.csv", index=False)
    LOGGER.info("Training complete. Best validation MRR: %.4f", best_mrr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Hum2Tune DualEncoder for humming-to-song retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", type=str, default=str(PROJECT_ROOT / "data" / "manifests"))
    parser.add_argument(
        "--reference-column",
        type=str,
        default="reference_path",
        help="Column in songs.csv used as the reference audio path.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_dual_encoder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        manifest_dir=args.manifest_dir,
        reference_column=args.reference_column,
    )
