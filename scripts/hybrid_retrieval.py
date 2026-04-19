from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
QUERY_ROOT_DEFAULT = ROOT / "data" / "Humming Audio"
DEMIX_ROOT = ROOT / "data" / "demucs_output" / "htdemucs"
ORIGINAL_ROOT = ROOT / "data" / "Original Songs"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "reference_feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")
SR = 16000


@dataclass
class ReferenceEntry:
    title_display: str
    title_key: str
    vocals_path: Optional[Path]
    no_vocals_path: Optional[Path]
    original_path: Optional[Path]


@dataclass
class ReferenceFeatures:
    title_display: str
    title_key: str
    vocals_pitch: Optional[np.ndarray]
    vocals_chroma: Optional[np.ndarray]
    original_chroma: Optional[np.ndarray]
    no_vocals_chroma: Optional[np.ndarray]


# Leave empty if your names are already consistent.
ALIASES: dict[str, str] = {}


def canonical_title(name: str) -> str:
    """
    Convert a folder/file name into a canonical matching key.

    Handles:
    - upper/lowercase
    - punctuation
    - apostrophes / symbols
    - underscores / extra spaces
    """
    name = Path(name).stem
    name = unicodedata.normalize("NFKD", name)
    name = name.replace("'", "'").replace("'", "'").replace("_", " ")
    name = re.sub(r"[^a-zA-Z0-9]+", " ", name.lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return ALIASES.get(name, name)


def sanitize_filename(name: str) -> str:
    name = canonical_title(name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return name or "unknown"


def load_audio(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y is None or len(y) == 0:
        raise ValueError(f"Empty audio file: {path}")
    y = librosa.util.normalize(y)
    return y.astype(np.float32), sr


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x.astype(np.float32)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def extract_chroma(path: Path, sr: int = SR, hop_length: int = 512) -> np.ndarray:
    y, _ = load_audio(path, sr=sr)
    y_harmonic, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harmonic)) > 1e-6:
        y = y_harmonic

    chroma = librosa.feature.chroma_cens(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )
    if chroma.shape[1] == 0:
        raise ValueError(f"No chroma frames extracted from {path}")
    return chroma.astype(np.float32)


def extract_pitch_sequence(
    path: Path,
    sr: int = SR,
    frame_length: int = 2048,
    hop_length: int = 256,
    fmin_note: str = "C2",
    fmax_note: str = "C6",
    min_voiced_frames: int = 8,
) -> np.ndarray:
    y, sr = load_audio(path, sr=sr)

    y_harmonic, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harmonic)) > 1e-6:
        y = y_harmonic

    f0, voiced_flag, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if f0 is None:
        raise ValueError(f"pYIN failed for {path}")

    mask = (~np.isnan(f0)) & voiced_flag.astype(bool)
    pitch_hz = f0[mask]

    if len(pitch_hz) < min_voiced_frames:
        raise ValueError(f"Too few voiced frames in {path}")

    pitch_midi = librosa.hz_to_midi(pitch_hz).astype(np.float32)
    pitch_midi = moving_average(pitch_midi, k=5)
    pitch_midi = pitch_midi - np.median(pitch_midi)
    return pitch_midi.astype(np.float32)


def to_interval_contour(seq: np.ndarray) -> np.ndarray:
    if len(seq) < 2:
        raise ValueError("Sequence too short for interval contour")
    contour = np.diff(seq).astype(np.float32)
    contour = np.clip(contour, -12.0, 12.0)
    contour = moving_average(contour, k=3)
    return contour.astype(np.float32)


def downsample_chroma(chroma: np.ndarray, factor: int = 4) -> np.ndarray:
    if chroma.shape[1] <= factor:
        return chroma.astype(np.float32)
    return chroma[:, ::factor].astype(np.float32)


def subseq_dtw_distance(query_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    X = query_seq[np.newaxis, :]
    Y = ref_seq[np.newaxis, :]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean", subseq=True)
    end_j = int(np.argmin(D[-1, :]))
    best_cost = float(D[-1, end_j])
    path_len = max(len(wp), 1)
    return best_cost / path_len


def dtw_chroma_distance(query_feat: np.ndarray, ref_feat: np.ndarray) -> float:
    D, wp = librosa.sequence.dtw(X=query_feat, Y=ref_feat, metric="cosine")
    path_len = max(len(wp), 1)
    return float(D[-1, -1] / path_len)


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def discover_queries(query_root: Path) -> list[dict]:
    queries = []
    for song_dir in sorted(query_root.iterdir()):
        if not song_dir.is_dir():
            continue

        song_key = canonical_title(song_dir.name)

        for path in sorted(song_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                queries.append(
                    {
                        "query_path": path,
                        "true_title_display": song_dir.name,
                        "true_title_key": song_key,
                    }
                )
    return queries


def discover_references() -> dict[str, ReferenceEntry]:
    refs: dict[str, ReferenceEntry] = {}

    if DEMIX_ROOT.exists():
        for song_dir in sorted(DEMIX_ROOT.iterdir()):
            if not song_dir.is_dir():
                continue

            key = canonical_title(song_dir.name)
            vocals_path = song_dir / "vocals.wav"
            no_vocals_path = song_dir / "no_vocals.wav"

            refs[key] = ReferenceEntry(
                title_display=song_dir.name,
                title_key=key,
                vocals_path=vocals_path if vocals_path.exists() else None,
                no_vocals_path=no_vocals_path if no_vocals_path.exists() else None,
                original_path=None,
            )

    if ORIGINAL_ROOT.exists():
        for path in sorted(ORIGINAL_ROOT.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            key = canonical_title(path.stem)

            if key in refs:
                refs[key].original_path = path
            else:
                refs[key] = ReferenceEntry(
                    title_display=path.stem,
                    title_key=key,
                    vocals_path=None,
                    no_vocals_path=None,
                    original_path=path,
                )

    return refs


def mode_weights(mode: str) -> dict[str, float]:
    if mode == "vocal_only":
        return {
            "vocal_pitch": 0.60,
            "vocal_chroma": 0.40,
            "orig_chroma": 0.00,
            "novocal_chroma": 0.00,
        }

    if mode == "vocal_plus_original":
        return {
            "vocal_pitch": 0.50,
            "vocal_chroma": 0.30,
            "orig_chroma": 0.20,
            "novocal_chroma": 0.00,
        }

    return {
        "vocal_pitch": 0.45,
        "vocal_chroma": 0.25,
        "orig_chroma": 0.20,
        "novocal_chroma": 0.10,
    }


def compute_rank_metrics(ranked_df: pd.DataFrame, true_title_key: str):
    ranked_keys = ranked_df["candidate_title_key"].tolist()
    try:
        rank = ranked_keys.index(true_title_key) + 1
    except ValueError:
        rank = None

    top1 = int(rank == 1) if rank is not None else 0
    top3 = int(rank is not None and rank <= 3)
    top5 = int(rank is not None and rank <= 5)
    mrr = 1.0 / rank if rank is not None else 0.0
    return rank, top1, top3, top5, mrr


def summarize(df: pd.DataFrame) -> dict:
    """Compute retrieval summary metrics from ranked results.

    Metrics reported:
      Top-1, Top-3, Top-5: fraction of queries where the correct song
        appears within the top K results.
      MRR: Mean Reciprocal Rank -- average of 1/rank for each query.
      MAP@10: Mean Average Precision at rank 10 -- standard IR metric
        that rewards correct answers near the top of the ranking.
      NDCG@10: Normalised Discounted Cumulative Gain at rank 10 -- applies
        a logarithmic discount to penalise correct answers at lower ranks.
    """
    if len(df) == 0:
        return {"count": 0, "top1": None, "top3": None, "top5": None,
                "mrr": None, "map_at_10": None, "ndcg_at_10": None}

    # MRR from pre-computed per-query rank
    mrr = float(df["mrr"].mean())

    # MAP@10: treat each query as a single-relevant-item retrieval task.
    # rank_of_true_song gives position in the full ranked list.
    map_at_10_vals = []
    ndcg_at_10_vals = []
    for _, row in df.iterrows():
        rank = row.get("rank_of_true_song")
        if rank is not None and not pd.isna(rank) and int(rank) <= 10:
            r = int(rank)
            map_at_10_vals.append(1.0 / r)           # AP for single-relevant item = 1/rank
            ndcg_at_10_vals.append(1.0 / np.log2(r + 1))  # DCG discount / IDCG
        else:
            map_at_10_vals.append(0.0)
            ndcg_at_10_vals.append(0.0)

    return {
        "count": int(len(df)),
        "top1": float(df["top1"].mean()),
        "top3": float(df["top3"].mean()),
        "top5": float(df["top5"].mean()),
        "mrr": mrr,
        "map_at_10": float(np.mean(map_at_10_vals)),
        "ndcg_at_10": float(np.mean(ndcg_at_10_vals)),
    }


def cache_file_for_reference(ref: ReferenceEntry) -> Path:
    safe_name = sanitize_filename(ref.title_display)
    return CACHE_DIR / f"{safe_name}.pkl"


def file_sig(path: Optional[Path]) -> Optional[tuple[str, int, int]]:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime), int(stat.st_size))


def load_reference_features_from_cache(ref: ReferenceEntry) -> Optional[ReferenceFeatures]:
    cache_path = cache_file_for_reference(ref)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    expected_meta = {
        "vocals": file_sig(ref.vocals_path),
        "original": file_sig(ref.original_path),
        "no_vocals": file_sig(ref.no_vocals_path),
    }

    if payload.get("meta") != expected_meta:
        return None

    feats = payload.get("features")
    if not isinstance(feats, ReferenceFeatures):
        return None
    return feats


def save_reference_features_to_cache(ref: ReferenceEntry, feats: ReferenceFeatures) -> None:
    cache_path = cache_file_for_reference(ref)
    payload = {
        "meta": {
            "vocals": file_sig(ref.vocals_path),
            "original": file_sig(ref.original_path),
            "no_vocals": file_sig(ref.no_vocals_path),
        },
        "features": feats,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def safe_extract_reference_features(ref: ReferenceEntry, use_cache: bool = True) -> ReferenceFeatures:
    if use_cache:
        cached = load_reference_features_from_cache(ref)
        if cached is not None:
            return cached

    vocals_pitch = None
    vocals_chroma = None
    original_chroma = None
    no_vocals_chroma = None

    try:
        if ref.vocals_path and ref.vocals_path.exists():
            vocals_pitch = to_interval_contour(extract_pitch_sequence(ref.vocals_path))
    except Exception:
        vocals_pitch = None

    try:
        if ref.vocals_path and ref.vocals_path.exists():
            vocals_chroma = extract_chroma(ref.vocals_path)
    except Exception:
        vocals_chroma = None

    try:
        if ref.original_path and ref.original_path.exists():
            original_chroma = extract_chroma(ref.original_path)
    except Exception:
        original_chroma = None

    try:
        if ref.no_vocals_path and ref.no_vocals_path.exists():
            no_vocals_chroma = extract_chroma(ref.no_vocals_path)
    except Exception:
        no_vocals_chroma = None

    feats = ReferenceFeatures(
        title_display=ref.title_display,
        title_key=ref.title_key,
        vocals_pitch=vocals_pitch,
        vocals_chroma=vocals_chroma,
        original_chroma=original_chroma,
        no_vocals_chroma=no_vocals_chroma,
    )

    if use_cache:
        save_reference_features_to_cache(ref, feats)

    return feats


def safe_extract_query_features(query_path: Path) -> tuple[np.ndarray, np.ndarray]:
    q_pitch = to_interval_contour(extract_pitch_sequence(query_path))
    q_chroma = extract_chroma(query_path)
    return q_pitch, q_chroma


def save_outputs(
    ranked_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    summary: dict,
    mode: str,
    suffix: str = "",
) -> None:
    suffix = f"_{suffix}" if suffix else ""
    ranked_out = OUTPUT_DIR / f"hybrid_retrieval_{mode}_ranked{suffix}.csv"
    candidates_out = OUTPUT_DIR / f"hybrid_retrieval_{mode}_topk{suffix}.csv"
    summary_out = OUTPUT_DIR / f"hybrid_retrieval_{mode}_summary{suffix}.json"

    ranked_df.to_csv(ranked_out, index=False)
    candidates_df.to_csv(candidates_out, index=False)

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run(
    mode: str = "hybrid",
    query_root: str = "data/Humming Audio",
    top_k: int = 5,
    shortlist: int = 20,
    max_queries: Optional[int] = None,
    save_every: int = 20,
    use_cache: bool = True,
):
    query_root_path = (ROOT / query_root).resolve() if not Path(query_root).is_absolute() else Path(query_root)
    weights = mode_weights(mode)

    queries = discover_queries(query_root_path)
    refs = discover_references()

    if max_queries is not None:
        queries = queries[:max_queries]

    if not queries:
        raise FileNotFoundError(f"No query files found under {query_root_path}")
    if not refs:
        raise FileNotFoundError("No references found under demucs_output/htdemucs or Original Songs")

    print(f"Queries found: {len(queries)}")
    print(f"Reference songs found: {len(refs)}")
    print(f"Mode: {mode}")
    print(f"Weights: {weights}")
    print(f"Shortlist size: {shortlist}")
    print(f"Cache enabled: {use_cache}")

    print("\n[1/3] Precomputing reference features...")
    ref_features: dict[str, ReferenceFeatures] = {}
    for key, ref in tqdm(refs.items(), total=len(refs), desc="References", unit="song"):
        ref_features[key] = safe_extract_reference_features(ref, use_cache=use_cache)

    print("[2/3] Running query retrieval...")
    ranked_rows = []
    candidate_rows = []

    start_time = time.perf_counter()

    for idx, q in enumerate(tqdm(queries, total=len(queries), desc="Queries", unit="query"), start=1):
        query_path: Path = q["query_path"]
        true_key: str = q["true_title_key"]

        try:
            q_pitch, q_chroma = safe_extract_query_features(query_path)
        except Exception as exc:
            print(f"Skipping query {query_path.name}: {exc}")
            continue

        # Stage 1: coarse shortlist using downsampled chroma
        coarse_rows = []
        q_chroma_small = downsample_chroma(q_chroma, factor=4)

        for ref_key, feat in ref_features.items():
            candidates = []

            if feat.vocals_chroma is not None:
                try:
                    candidates.append(
                        dtw_chroma_distance(
                            q_chroma_small,
                            downsample_chroma(feat.vocals_chroma, factor=4),
                        )
                    )
                except Exception:
                    pass

            if feat.original_chroma is not None:
                try:
                    candidates.append(
                        dtw_chroma_distance(
                            q_chroma_small,
                            downsample_chroma(feat.original_chroma, factor=4),
                        )
                    )
                except Exception:
                    pass

            if feat.no_vocals_chroma is not None:
                try:
                    candidates.append(
                        dtw_chroma_distance(
                            q_chroma_small,
                            downsample_chroma(feat.no_vocals_chroma, factor=4),
                        )
                    )
                except Exception:
                    pass

            coarse_score = min(candidates) if candidates else np.inf

            coarse_rows.append(
                {
                    "candidate_title_key": ref_key,
                    "candidate_title_display": feat.title_display,
                    "coarse_score": coarse_score,
                }
            )

        coarse_df = pd.DataFrame(coarse_rows)
        coarse_df = coarse_df.sort_values(["coarse_score", "candidate_title_display"], ascending=[True, True])
        shortlist_keys = coarse_df.head(shortlist)["candidate_title_key"].tolist()

        # Stage 2: fine ranking on shortlist only
        fine_rows = []

        for ref_key in shortlist_keys:
            feat = ref_features[ref_key]

            row = {
                "query_path": str(query_path),
                "query_name": query_path.name,
                "true_title_display": q["true_title_display"],
                "true_title_key": true_key,
                "candidate_title_display": feat.title_display,
                "candidate_title_key": ref_key,
                "vocal_pitch_dist": np.nan,
                "vocal_chroma_dist": np.nan,
                "orig_chroma_dist": np.nan,
                "novocal_chroma_dist": np.nan,
            }

            if feat.vocals_pitch is not None:
                try:
                    row["vocal_pitch_dist"] = subseq_dtw_distance(q_pitch, feat.vocals_pitch)
                except Exception:
                    pass

            if feat.vocals_chroma is not None:
                try:
                    row["vocal_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.vocals_chroma)
                except Exception:
                    pass

            if feat.original_chroma is not None:
                try:
                    row["orig_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.original_chroma)
                except Exception:
                    pass

            if feat.no_vocals_chroma is not None:
                try:
                    row["novocal_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.no_vocals_chroma)
                except Exception:
                    pass

            fine_rows.append(row)

        cand_df = pd.DataFrame(fine_rows)

        score_cols = [
            "vocal_pitch_dist",
            "vocal_chroma_dist",
            "orig_chroma_dist",
            "novocal_chroma_dist",
        ]

        for col in score_cols:
            values = cand_df[col].to_numpy(dtype=np.float32)
            valid_mask = np.isfinite(values)
            norm = np.ones_like(values, dtype=np.float32)
            if valid_mask.any():
                norm_valid = minmax_normalize(values[valid_mask])
                norm[valid_mask] = norm_valid
            cand_df[col + "_norm"] = norm

        # -- Vectorised fused score computation --------------------------
        # Replaces the previous iterrows() loop (O(N) Python overhead per row).
        # Each distance column is normalised; scores are weighted and summed.
        score_column_map = {
            "vocal_pitch":    "vocal_pitch_dist",
            "vocal_chroma":   "vocal_chroma_dist",
            "orig_chroma":    "orig_chroma_dist",
            "novocal_chroma": "novocal_chroma_dist",
        }

        fused = np.zeros(len(cand_df), dtype=np.float32)
        weight_sums = np.zeros(len(cand_df), dtype=np.float32)

        for score_name, weight in weights.items():
            if weight <= 0:
                continue
            col_norm = score_column_map[score_name] + "_norm"
            if col_norm not in cand_df.columns:
                continue
            vals = cand_df[col_norm].to_numpy(dtype=np.float32)
            valid = np.isfinite(vals)
            fused[valid] += weight * vals[valid]
            weight_sums[valid] += weight

        # Where no feature was available, assign inf so the candidate
        # sinks to the bottom of the ranking.
        with np.errstate(invalid="ignore", divide="ignore"):
            fused_scores_arr = np.where(weight_sums > 0, fused / weight_sums, np.inf)

        cand_df["fused_score"] = fused_scores_arr
        cand_df = cand_df.sort_values(
            ["fused_score", "candidate_title_display"],
            ascending=[True, True],
        ).reset_index(drop=True)

        rank, top1, top3, top5, mrr = compute_rank_metrics(cand_df, true_key)

        ranked_rows.append(
            {
                "query_path": str(query_path),
                "query_name": query_path.name,
                "true_title_display": q["true_title_display"],
                "true_title_key": true_key,
                "pred_title_display": cand_df.iloc[0]["candidate_title_display"],
                "pred_title_key": cand_df.iloc[0]["candidate_title_key"],
                "rank_of_true_song": rank,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "mrr": mrr,
                "mode": mode,
                "ranking": " | ".join(
                    f"{r.candidate_title_display}:{r.fused_score:.4f}"
                    for r in cand_df.head(max(top_k, 10)).itertuples(index=False)
                ),
            }
        )

        candidate_rows.append(cand_df.head(top_k))

        if idx % save_every == 0:
            ranked_df = pd.DataFrame(ranked_rows)
            candidates_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
            summary = summarize(ranked_df)
            save_outputs(ranked_df, candidates_df, summary, mode, suffix="partial")
            elapsed = time.perf_counter() - start_time
            print(
                f"\nProgress: {idx}/{len(queries)} queries | "
                f"elapsed {elapsed/60:.1f} min | "
                f"current Top-1 {summary['top1']:.4f} | MRR {summary['mrr']:.4f}"
            )

    print("[3/3] Saving final outputs...")
    ranked_df = pd.DataFrame(ranked_rows)
    candidates_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    summary = summarize(ranked_df)
    save_outputs(ranked_df, candidates_df, summary, mode)

    elapsed = time.perf_counter() - start_time
    print("\n=== RETRIEVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
    print(f"Saved final files under: {OUTPUT_DIR}")


def rank_query(
    query_path: "Path",
    ref_features: dict,
    weights: dict,
    shortlist_size: int = 10,
    top_k: int = 3,
) -> "pd.DataFrame":
    """Rank all reference songs against a single query audio file.

    This is the shared per-query scoring function used by both the batch
    evaluation pipeline (run()) and the Streamlit app (app.py), eliminating
    the duplicated ranking logic that previously existed in both files.

    Args:
        query_path:    Path to the query audio file (humming/recording).
        ref_features:  Dict mapping title_key -> ReferenceFeatures, pre-loaded.
        weights:       Score weight dict, e.g. {"vocal_pitch": 0.6, ...}.
        shortlist_size: Number of candidates kept after coarse chroma filtering.
        top_k:         Number of ranked results to return.

    Returns:
        DataFrame with columns [candidate_title_display, candidate_title_key,
        vocal_pitch_dist, vocal_chroma_dist, fused_score], sorted ascending
        by fused_score (lower = better match). Only top_k rows returned.

    Raises:
        ValueError: If pitch/chroma features cannot be extracted from query.
    """
    import pandas as pd

    q_pitch, q_chroma = safe_extract_query_features(query_path)
    q_chroma_small = downsample_chroma(q_chroma, factor=4)

    # -- Stage 1: coarse chroma shortlist ----------------------------------
    coarse_rows = []
    for ref_key, feat in ref_features.items():
        coarse_score = np.inf
        for chroma_attr in ("vocals_chroma", "original_chroma", "no_vocals_chroma"):
            ref_chroma = getattr(feat, chroma_attr, None)
            if ref_chroma is not None:
                try:
                    s = dtw_chroma_distance(q_chroma_small, downsample_chroma(ref_chroma, factor=4))
                    coarse_score = min(coarse_score, s)
                except Exception:
                    pass
        coarse_rows.append({
            "candidate_title_key": ref_key,
            "candidate_title_display": feat.title_display,
            "coarse_score": coarse_score,
        })

    coarse_df = pd.DataFrame(coarse_rows).sort_values(
        ["coarse_score", "candidate_title_display"], ascending=[True, True]
    )
    shortlist_keys = coarse_df.head(shortlist_size)["candidate_title_key"].tolist()

    # -- Stage 2: fine-grained ranking on shortlist ------------------------
    score_column_map = {
        "vocal_pitch":    "vocal_pitch_dist",
        "vocal_chroma":   "vocal_chroma_dist",
        "orig_chroma":    "orig_chroma_dist",
        "novocal_chroma": "novocal_chroma_dist",
    }
    fine_rows = []
    for ref_key in shortlist_keys:
        feat = ref_features[ref_key]
        row: dict = {
            "candidate_title_display": feat.title_display,
            "candidate_title_key": ref_key,
            "vocal_pitch_dist":    np.nan,
            "vocal_chroma_dist":   np.nan,
            "orig_chroma_dist":    np.nan,
            "novocal_chroma_dist": np.nan,
        }
        if feat.vocals_pitch is not None:
            try:
                row["vocal_pitch_dist"] = subseq_dtw_distance(q_pitch, feat.vocals_pitch)
            except Exception:
                pass
        if feat.vocals_chroma is not None:
            try:
                row["vocal_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.vocals_chroma)
            except Exception:
                pass
        if feat.original_chroma is not None:
            try:
                row["orig_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.original_chroma)
            except Exception:
                pass
        if feat.no_vocals_chroma is not None:
            try:
                row["novocal_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.no_vocals_chroma)
            except Exception:
                pass
        fine_rows.append(row)

    cand_df = pd.DataFrame(fine_rows)
    if cand_df.empty:
        return cand_df

    # Normalise each distance column within the shortlist
    for col in [v for v in score_column_map.values()]:
        values = cand_df[col].to_numpy(dtype=np.float32)
        valid = np.isfinite(values)
        norm = np.ones_like(values, dtype=np.float32)
        if valid.any():
            norm[valid] = minmax_normalize(values[valid])
        cand_df[col + "_norm"] = norm

    # Vectorised weighted fusion
    fused = np.zeros(len(cand_df), dtype=np.float32)
    weight_sums = np.zeros(len(cand_df), dtype=np.float32)
    for score_name, weight in weights.items():
        if weight <= 0:
            continue
        col_norm = score_column_map.get(score_name, "") + "_norm"
        if col_norm not in cand_df.columns:
            continue
        vals = cand_df[col_norm].to_numpy(dtype=np.float32)
        valid = np.isfinite(vals)
        fused[valid] += weight * vals[valid]
        weight_sums[valid] += weight

    with np.errstate(invalid="ignore", divide="ignore"):
        fused_final = np.where(weight_sums > 0, fused / weight_sums, np.inf)

    cand_df["fused_score"] = fused_final
    return (
        cand_df.sort_values(["fused_score", "candidate_title_display"], ascending=[True, True])
        .reset_index(drop=True)
        .head(top_k)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid humming-to-song retrieval")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["vocal_only", "vocal_plus_original", "hybrid"])
    parser.add_argument("--query-root", type=str, default="data/Humming Audio")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shortlist", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true", help="Disable reference feature cache")
    args = parser.parse_args()

    run(
        mode=args.mode,
        query_root=args.query_root,
        top_k=args.top_k,
        shortlist=args.shortlist,
        max_queries=args.max_queries,
        save_every=args.save_every,
        use_cache=not args.no_cache,
    )