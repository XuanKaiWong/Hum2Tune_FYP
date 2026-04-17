from pathlib import Path
import os
import numpy as np
import pandas as pd
import librosa

ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def rel_to_abs(rel_path: str) -> Path:
    return ROOT / Path(rel_path.replace("/", os.sep))


def load_audio(path: Path, sr: int = 16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y is None or len(y) == 0:
        raise ValueError(f"Empty audio file: {path}")
    y = librosa.util.normalize(y)
    return y, sr


# ---------- CHROMA FEATURE ----------

def extract_chroma(path: Path, sr: int = 16000, hop_length: int = 512) -> np.ndarray:
    y, _ = load_audio(path, sr=sr)

    y_harmonic, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harmonic)) > 1e-6:
        y = y_harmonic

    chroma = librosa.feature.chroma_cens(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    if chroma.shape[1] == 0:
        raise ValueError(f"No chroma frames extracted: {path}")

    return chroma.astype(np.float32)


def dtw_chroma_distance(query_feat: np.ndarray, ref_feat: np.ndarray) -> float:
    D, wp = librosa.sequence.dtw(X=query_feat, Y=ref_feat, metric="cosine")
    path_len = max(len(wp), 1)
    return float(D[-1, -1] / path_len)


# ---------- PITCH SUBSEQUENCE FEATURE ----------

def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x.astype(np.float32)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def extract_pitch_sequence(
    path: Path,
    sr: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 256,
    fmin_note: str = "C2",
    fmax_note: str = "C6",
    min_voiced_frames: int = 10,
):
    y, sr = load_audio(path, sr=sr)

    y_harmonic, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harmonic)) > 1e-6:
        y = y_harmonic

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if f0 is None:
        raise ValueError(f"pYIN failed for: {path}")

    mask = (~np.isnan(f0)) & voiced_flag.astype(bool)
    pitch_hz = f0[mask]

    if len(pitch_hz) < min_voiced_frames:
        raise ValueError(f"Too few voiced frames in: {path}")

    pitch_midi = librosa.hz_to_midi(pitch_hz).astype(np.float32)
    pitch_midi = moving_average(pitch_midi, k=5)
    pitch_midi = pitch_midi - np.median(pitch_midi)

    return pitch_midi


def to_interval_contour(seq: np.ndarray) -> np.ndarray:
    if len(seq) < 2:
        raise ValueError("Sequence too short for interval contour")
    contour = np.diff(seq).astype(np.float32)
    contour = np.clip(contour, -12.0, 12.0)
    contour = moving_average(contour, k=3)
    return contour


def subseq_dtw_distance(query_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    X = query_seq[np.newaxis, :]
    Y = ref_seq[np.newaxis, :]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean", subseq=True)
    end_j = int(np.argmin(D[-1, :]))
    best_cost = float(D[-1, end_j])
    path_len = max(len(wp), 1)
    return best_cost / path_len


# ---------- FUSION ----------

def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def evaluate_predictions(results_df: pd.DataFrame):
    metrics = {}
    for split_name in ["train", "val", "test", "all"]:
        if split_name == "all":
            part = results_df
        else:
            part = results_df[results_df["split"] == split_name]

        if len(part) == 0:
            metrics[split_name] = None
        else:
            metrics[split_name] = float(part["correct"].mean())
    return metrics


def main():
    songs_df = pd.read_csv(MANIFEST_DIR / "songs.csv")
    queries_df = pd.read_csv(MANIFEST_DIR / "queries.csv")
    splits_df = pd.read_csv(MANIFEST_DIR / "splits.csv")

    df = queries_df.merge(splits_df, on="query_id", how="left")

    # Validate files
    for row in songs_df.itertuples(index=False):
        p = rel_to_abs(row.reference_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing reference audio: {p}")

    for row in df.itertuples(index=False):
        p = rel_to_abs(row.query_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing query audio: {p}")

    # Precompute reference features
    ref_db = {}
    for row in songs_df.itertuples(index=False):
        song_id = int(row.song_id)
        ref_path = rel_to_abs(row.reference_path)

        chroma_feat = extract_chroma(ref_path)
        pitch_seq = extract_pitch_sequence(ref_path)
        contour_seq = to_interval_contour(pitch_seq)

        ref_db[song_id] = {
            "title": row.title,
            "chroma": chroma_feat,
            "contour": contour_seq,
        }

    # Precompute all raw distances per query-song
    rows = []
    for row in df.itertuples(index=False):
        query_id = row.query_id
        true_song_id = int(row.song_id)
        split = row.split
        query_path = rel_to_abs(row.query_path)

        q_chroma = extract_chroma(query_path)
        q_pitch = extract_pitch_sequence(query_path)
        q_contour = to_interval_contour(q_pitch)

        for song_id, ref_info in ref_db.items():
            chroma_dist = dtw_chroma_distance(q_chroma, ref_info["chroma"])
            pitch_dist = subseq_dtw_distance(q_contour, ref_info["contour"])

            rows.append({
                "query_id": query_id,
                "split": split,
                "true_song_id": true_song_id,
                "candidate_song_id": song_id,
                "candidate_title": ref_info["title"],
                "chroma_dist": chroma_dist,
                "pitch_dist": pitch_dist,
                "query_path": str(query_path),
            })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUTPUT_DIR / "fusion_raw_candidate_scores.csv", index=False)

    # Sweep alpha from 0.0 to 1.0
    # fused = alpha * chroma_norm + (1-alpha) * pitch_norm
    sweep_results = []
    best_alpha = None
    best_val = -1.0
    best_train = -1.0
    best_detail_df = None

    for alpha in np.linspace(0.0, 1.0, 11):
        pred_rows = []

        for query_id, part in raw_df.groupby("query_id"):
            part = part.copy()

            part["chroma_norm"] = minmax_normalize(part["chroma_dist"].values)
            part["pitch_norm"] = minmax_normalize(part["pitch_dist"].values)
            part["fused_score"] = alpha * part["chroma_norm"] + (1.0 - alpha) * part["pitch_norm"]

            best_idx = part["fused_score"].idxmin()
            best_row = part.loc[best_idx]

            pred_rows.append({
                "query_id": best_row["query_id"],
                "split": best_row["split"],
                "true_song_id": int(best_row["true_song_id"]),
                "pred_song_id": int(best_row["candidate_song_id"]),
                "pred_title": best_row["candidate_title"],
                "correct": int(int(best_row["true_song_id"]) == int(best_row["candidate_song_id"])),
                "alpha": float(alpha),
                "query_path": best_row["query_path"],
            })

        pred_df = pd.DataFrame(pred_rows)
        metrics = evaluate_predictions(pred_df)

        sweep_results.append({
            "alpha": float(alpha),
            "train_acc": metrics["train"],
            "val_acc": metrics["val"],
            "test_acc": metrics["test"],
            "all_acc": metrics["all"],
        })

        val_acc = metrics["val"] if metrics["val"] is not None else -1.0
        train_acc = metrics["train"] if metrics["train"] is not None else -1.0

        # Select alpha using validation only; tie-break with train
        if (val_acc > best_val) or (val_acc == best_val and train_acc > best_train):
            best_val = val_acc
            best_train = train_acc
            best_alpha = float(alpha)
            best_detail_df = pred_df.copy()

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(OUTPUT_DIR / "fusion_alpha_sweep.csv", index=False)

    best_detail_df.to_csv(OUTPUT_DIR / "fusion_best_predictions.csv", index=False)

    chosen = sweep_df[sweep_df["alpha"] == best_alpha].iloc[0]

    print("\n=== Alpha sweep ===")
    print(sweep_df.to_string(index=False))

    print(f"\nChosen alpha (by val, tie-break train): {best_alpha:.1f}")
    print(f"train: {chosen['train_acc']:.3f}")
    print(f"  val: {chosen['val_acc']:.3f}")
    print(f" test: {chosen['test_acc']:.3f}")
    print(f"  all: {chosen['all_acc']:.3f}")

    print("\n=== Best-alpha predictions ===")
    print(best_detail_df[[
        "query_id", "split", "true_song_id", "pred_song_id",
        "correct", "pred_title", "alpha"
    ]].to_string(index=False))

    print(f"\nSaved raw scores to: {OUTPUT_DIR / 'fusion_raw_candidate_scores.csv'}")
    print(f"Saved sweep summary to: {OUTPUT_DIR / 'fusion_alpha_sweep.csv'}")
    print(f"Saved best predictions to: {OUTPUT_DIR / 'fusion_best_predictions.csv'}")


if __name__ == "__main__":
    main()