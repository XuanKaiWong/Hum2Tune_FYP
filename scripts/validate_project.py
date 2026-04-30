from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def resolve_path(value: object) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_manifests() -> int:
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    songs_path = manifest_dir / "songs.csv"
    queries_path = manifest_dir / "queries.csv"
    splits_path = manifest_dir / "splits.csv"

    errors = 0

    if not songs_path.exists():
        fail(f"Missing {songs_path}")
        return 1
    if not queries_path.exists():
        fail(f"Missing {queries_path}")
        return 1

    songs = pd.read_csv(songs_path, dtype=str).fillna("")
    queries = pd.read_csv(queries_path, dtype=str).fillna("")

    for col in ("song_id", "title", "reference_path"):
        if col not in songs.columns:
            fail(f"songs.csv missing required column: {col}")
            errors += 1

    for col in ("query_id", "song_id", "query_path"):
        if col not in queries.columns:
            fail(f"queries.csv missing required column: {col}")
            errors += 1

    if errors:
        return errors

    if "split" not in queries.columns:
        if splits_path.exists():
            splits = pd.read_csv(splits_path, dtype=str).fillna("")
            if {"query_id", "split"}.issubset(splits.columns):
                queries = queries.merge(splits[["query_id", "split"]], on="query_id", how="left")
                ok("queries.csv joined with splits.csv")
            else:
                fail("splits.csv exists but must contain query_id and split")
                errors += 1
        else:
            warn("No split column in queries.csv and no splits.csv; auto split will be used")

    missing_song_ids = sorted(set(queries["song_id"]) - set(songs["song_id"]))
    if missing_song_ids:
        fail(f"queries.csv references song_id values missing from songs.csv: {missing_song_ids}")
        errors += 1
    else:
        ok("All query song_id values exist in songs.csv")

    missing_queries = []
    for row in queries.itertuples(index=False):
        path = resolve_path(row.query_path)
        if not path.exists():
            missing_queries.append(str(path))
        elif path.suffix.lower() not in AUDIO_EXTENSIONS:
            warn(f"Unsupported query extension: {path}")

    missing_refs = []
    for row in songs.itertuples(index=False):
        path = resolve_path(row.reference_path)
        if not path.exists():
            missing_refs.append(str(path))
        elif path.suffix.lower() not in AUDIO_EXTENSIONS:
            warn(f"Unsupported reference extension: {path}")

    if missing_queries:
        fail(f"Missing query files: {len(missing_queries)}")
        for item in missing_queries[:10]:
            print(f"       {item}")
        if len(missing_queries) > 10:
            print(f"       ... {len(missing_queries) - 10} more")
        errors += 1
    else:
        ok(f"All {len(queries)} query files exist")

    if missing_refs:
        fail(f"Missing reference files: {len(missing_refs)}")
        for item in missing_refs[:10]:
            print(f"       {item}")
        if len(missing_refs) > 10:
            print(f"       ... {len(missing_refs) - 10} more")
        errors += 1
    else:
        ok(f"All {len(songs)} reference files exist")

    if "split" in queries.columns:
        split_counts = Counter(str(s).lower() for s in queries["split"] if str(s).strip())
        ok(f"Split counts: {dict(split_counts)}")
        for required in ("train", "val"):
            if split_counts.get(required, 0) == 0:
                fail(f"No records in required split: {required}")
                errors += 1

    class_counts = queries.merge(songs[["song_id", "title"]], on="song_id", how="left")["title"].value_counts()
    weak = class_counts[class_counts < 3]
    if len(weak):
        warn(
            f"{len(weak)} songs have fewer than 3 queries. "
            "Use retrieval metrics for these; avoid treating classifier accuracy as strong evidence."
        )

    summary = {
        "num_songs": int(len(songs)),
        "num_queries": int(len(queries)),
        "queries_per_song": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        "errors": int(errors),
    }
    out_path = PROJECT_ROOT / "results" / "evaluations" / "project_validation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ok(f"Saved validation summary to {out_path}")

    return errors


def main() -> int:
    print("Hum2Tune project validation")
    print("=" * 32)
    errors = validate_manifests()
    if errors:
        fail(f"Validation completed with {errors} issue(s). Fix these before final experiments.")
        return 1
    ok("Validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
