from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

RAW_CANDIDATE_PATH = OUTPUT_DIR / "fusion_raw_candidate_scores.csv"
SWEEP_OUTPUT_PATH = OUTPUT_DIR / "fusion_ranked_alpha_sweep.csv"
BEST_OUTPUT_PATH = OUTPUT_DIR / "fusion_ranked_best_predictions.csv"


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def compute_rank_metrics(group: pd.DataFrame):
    """
    group = one query's candidates, already sorted ascending by fused_score
    """
    true_song_id = int(group.iloc[0]["true_song_id"])
    ranked_song_ids = group["candidate_song_id"].astype(int).tolist()

    try:
        rank = ranked_song_ids.index(true_song_id) + 1
    except ValueError:
        rank = None

    top1 = int(rank == 1) if rank is not None else 0
    top3 = int(rank is not None and rank <= 3)
    mrr = 1.0 / rank if rank is not None else 0.0

    return rank, top1, top3, mrr


def summarize_split(df: pd.DataFrame):
    if len(df) == 0:
        return {
            "top1": None,
            "top3": None,
            "mrr": None,
            "count": 0,
        }

    return {
        "top1": float(df["top1"].mean()),
        "top3": float(df["top3"].mean()),
        "mrr": float(df["mrr"].mean()),
        "count": int(len(df)),
    }


def main():
    if not RAW_CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw candidate file: {RAW_CANDIDATE_PATH}\n"
            "Run baseline_fusion_sweep.py first."
        )

    raw_df = pd.read_csv(RAW_CANDIDATE_PATH)

    required_cols = {
        "query_id",
        "split",
        "true_song_id",
        "candidate_song_id",
        "candidate_title",
        "chroma_dist",
        "pitch_dist",
        "query_path",
    }
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw candidate file: {sorted(missing)}")

    sweep_rows = []

    best_alpha = None
    best_val_top1 = -1.0
    best_val_mrr = -1.0
    best_train_top1 = -1.0
    best_train_mrr = -1.0
    best_ranked_df = None

    for alpha in np.linspace(0.0, 1.0, 11):
        ranked_rows = []

        for query_id, part in raw_df.groupby("query_id"):
            part = part.copy()

            part["chroma_norm"] = minmax_normalize(part["chroma_dist"].values)
            part["pitch_norm"] = minmax_normalize(part["pitch_dist"].values)
            part["fused_score"] = alpha * part["chroma_norm"] + (1.0 - alpha) * part["pitch_norm"]

            part = part.sort_values(
                ["fused_score", "candidate_song_id"],
                ascending=[True, True]
            ).reset_index(drop=True)

            rank, top1, top3, mrr = compute_rank_metrics(part)

            pred_row = part.iloc[0]

            ranked_rows.append({
                "query_id": query_id,
                "split": pred_row["split"],
                "true_song_id": int(pred_row["true_song_id"]),
                "pred_song_id": int(pred_row["candidate_song_id"]),
                "pred_title": pred_row["candidate_title"],
                "rank_of_true_song": rank,
                "top1": top1,
                "top3": top3,
                "mrr": mrr,
                "alpha": float(alpha),
                "query_path": pred_row["query_path"],
                "ranking": " | ".join(
                    f"{int(r.candidate_song_id)}:{r.candidate_title}:{r.fused_score:.4f}"
                    for r in part.itertuples(index=False)
                )
            })

        ranked_df = pd.DataFrame(ranked_rows)

        train_metrics = summarize_split(ranked_df[ranked_df["split"] == "train"])
        val_metrics = summarize_split(ranked_df[ranked_df["split"] == "val"])
        test_metrics = summarize_split(ranked_df[ranked_df["split"] == "test"])
        all_metrics = summarize_split(ranked_df)

        sweep_rows.append({
            "alpha": float(alpha),
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_mrr": train_metrics["mrr"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_mrr": val_metrics["mrr"],
            "test_top1": test_metrics["top1"],
            "test_top3": test_metrics["top3"],
            "test_mrr": test_metrics["mrr"],
            "all_top1": all_metrics["top1"],
            "all_top3": all_metrics["top3"],
            "all_mrr": all_metrics["mrr"],
        })

        val_top1 = val_metrics["top1"] if val_metrics["top1"] is not None else -1.0
        val_mrr = val_metrics["mrr"] if val_metrics["mrr"] is not None else -1.0
        train_top1 = train_metrics["top1"] if train_metrics["top1"] is not None else -1.0
        train_mrr = train_metrics["mrr"] if train_metrics["mrr"] is not None else -1.0

        should_update = False

        if val_top1 > best_val_top1:
            should_update = True
        elif val_top1 == best_val_top1:
            if val_mrr > best_val_mrr:
                should_update = True
            elif val_mrr == best_val_mrr:
                if train_top1 > best_train_top1:
                    should_update = True
                elif train_top1 == best_train_top1:
                    if train_mrr > best_train_mrr:
                        should_update = True

        if should_update:
            best_val_top1 = val_top1
            best_val_mrr = val_mrr
            best_train_top1 = train_top1
            best_train_mrr = train_mrr
            best_alpha = float(alpha)
            best_ranked_df = ranked_df.copy()

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUTPUT_PATH, index=False)
    best_ranked_df.to_csv(BEST_OUTPUT_PATH, index=False)

    chosen = sweep_df[sweep_df["alpha"] == best_alpha].iloc[0]

    print("\n=== Ranked alpha sweep ===")
    print(sweep_df.to_string(index=False))

    print(f"\nChosen alpha (by val top1, val MRR, train top1, train MRR): {best_alpha:.1f}")
    print(
        f"train top1: {chosen['train_top1']:.3f} | "
        f"top3: {chosen['train_top3']:.3f} | "
        f"mrr: {chosen['train_mrr']:.3f}"
    )
    print(
        f"  val top1: {chosen['val_top1']:.3f} | "
        f"top3: {chosen['val_top3']:.3f} | "
        f"mrr: {chosen['val_mrr']:.3f}"
    )
    print(
        f" test top1: {chosen['test_top1']:.3f} | "
        f"top3: {chosen['test_top3']:.3f} | "
        f"mrr: {chosen['test_mrr']:.3f}"
    )
    print(
        f"  all top1: {chosen['all_top1']:.3f} | "
        f"top3: {chosen['all_top3']:.3f} | "
        f"mrr: {chosen['all_mrr']:.3f}"
    )

    print("\n=== Best-alpha per-query rankings ===")
    print(best_ranked_df[[
        "query_id",
        "split",
        "true_song_id",
        "pred_song_id",
        "rank_of_true_song",
        "top1",
        "top3",
        "mrr",
        "alpha"
    ]].to_string(index=False))

    print(f"\nSaved sweep summary to: {SWEEP_OUTPUT_PATH}")
    print(f"Saved best predictions to: {BEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()