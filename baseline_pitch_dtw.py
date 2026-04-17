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

    # Harmonic emphasis helps a bit on vocals
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

    # Keep only confident voiced frames
    mask = (~np.isnan(f0)) & (voiced_flag.astype(bool))
    pitch_hz = f0[mask]

    if len(pitch_hz) < min_voiced_frames:
        raise ValueError(f"Too few voiced frames in: {path}")

    pitch_midi = librosa.hz_to_midi(pitch_hz).astype(np.float32)

    # Key-invariant normalization: remove absolute pitch center
    pitch_midi = pitch_midi - np.median(pitch_midi)

    return pitch_midi


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32)
    xp = np.linspace(0, 1, num=len(x))
    fp = x
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, xp, fp).astype(np.float32)


def dtw_1d(query_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    # librosa.sequence.dtw expects 2D features: [dim, time]
    X = query_seq[np.newaxis, :]
    Y = ref_seq[np.newaxis, :]

    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    path_len = max(len(wp), 1)
    return float(D[-1, -1] / path_len)


def main():
    songs_df = pd.read_csv(MANIFEST_DIR / "songs.csv")
    queries_df = pd.read_csv(MANIFEST_DIR / "queries.csv")
    splits_df = pd.read_csv(MANIFEST_DIR / "splits.csv")

    df = queries_df.merge(splits_df, on="query_id", how="left")

    if df["split"].isna().any():
        missing = df[df["split"].isna()]["query_id"].tolist()
        raise ValueError(f"Missing split labels for queries: {missing}")

    for row in songs_df.itertuples(index=False):
        ref_path = rel_to_abs(row.reference_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference audio: {ref_path}")

    for row in df.itertuples(index=False):
        query_path = rel_to_abs(row.query_path)
        if not query_path.exists():
            raise FileNotFoundError(f"Missing query audio: {query_path}")

    # Precompute reference pitch sequences
    ref_features = {}
    for row in songs_df.itertuples(index=False):
        song_id = int(row.song_id)
        ref_path = rel_to_abs(row.reference_path)
        ref_seq = extract_pitch_sequence(ref_path)
        ref_features[song_id] = {
            "title": row.title,
            "path": str(ref_path),
            "seq": ref_seq,
        }

    results = []

    for row in df.itertuples(index=False):
        query_id = row.query_id
        true_song_id = int(row.song_id)
        query_path = rel_to_abs(row.query_path)
        split = row.split

        query_seq = extract_pitch_sequence(query_path)

        candidates = []
        for song_id, ref_info in ref_features.items():
            ref_seq = ref_info["seq"]

            # Optional length balancing to reduce extreme duration mismatch
            target_len = max(len(query_seq), 32)
            q_seq = resample_1d(query_seq, target_len)

            # Compare the query against multiple window scales of the reference
            # For this tiny pilot, a simple whole-reference resample is enough
            r_seq = resample_1d(ref_seq, max(len(ref_seq), target_len))

            dist = dtw_1d(q_seq, r_seq)
            candidates.append((song_id, ref_info["title"], dist))

        candidates.sort(key=lambda x: x[2])

        pred_song_id, pred_title, best_dist = candidates[0]

        results.append({
            "query_id": query_id,
            "split": split,
            "true_song_id": true_song_id,
            "pred_song_id": pred_song_id,
            "correct": int(pred_song_id == true_song_id),
            "query_path": str(query_path),
            "pred_title": pred_title,
            "best_distance": best_dist,
            "ranked_candidates": " | ".join(
                [f"{sid}:{title}:{dist:.4f}" for sid, title, dist in candidates]
            )
        })

    results_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "baseline_pitch_dtw_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== Per-query predictions ===")
    print(results_df[[
        "query_id", "split", "true_song_id", "pred_song_id",
        "correct", "pred_title", "best_distance"
    ]].to_string(index=False))

    print("\n=== Accuracy summary ===")
    for split_name in ["train", "val", "test"]:
        part = results_df[results_df["split"] == split_name]
        if len(part) > 0:
            acc = part["correct"].mean()
            print(f"{split_name:>5}: {acc:.3f} ({part['correct'].sum()}/{len(part)})")

    overall_acc = results_df["correct"].mean()
    print(f"{'all':>5}: {overall_acc:.3f} ({results_df['correct'].sum()}/{len(results_df)})")

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()