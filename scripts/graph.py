from pathlib import Path
from shutil import copy2
import json
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_EVAL_DIR = PROJECT_ROOT / "results" / "evaluations"
RESULTS_VIS_DIR = PROJECT_ROOT / "results" / "visualizations"

RESULTS_VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_first_existing_json(paths: list[Path], fallback: dict | None = None) -> dict:
    for path in paths:
        if path.exists():
            return load_json(path)

    if fallback is not None:
        print("Warning: using fallback values because JSON file was not found.")
        print("Checked paths:")
        for path in paths:
            print(f"  - {path}")
        return fallback

    raise FileNotFoundError(
        "None of the expected JSON files were found:\n"
        + "\n".join(str(p) for p in paths)
    )


def safe_get(data: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    return float(default)


def percent(value: float) -> float:
    return float(value) * 100.0


def add_bar_labels(ax, bars, fmt: str = "{:.1f}", offset: float = 1.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_score_labels(ax, bars, fmt: str = "{:.3f}", offset: float = 0.015) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def style_axes(ax) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


# ============================================================
# Load retrieval result summaries
# ============================================================
def load_retrieval_results() -> tuple[dict, dict, dict]:
    """
    Loads retrieval summaries. If the exact files are not found,
    fallback values are used based on the final report results.
    """

    vocal_only = load_first_existing_json(
        [
            OUTPUTS_DIR / "hybrid_retrieval_vocal_only_summary.json",
            OUTPUTS_DIR / "vocal_only_summary.json",
            RESULTS_EVAL_DIR / "hybrid_retrieval_vocal_only_summary.json",
        ],
        fallback={
            "top1": 0.3756,
            "top3": 0.6195,
            "top5": 0.6976,
            "mrr": 0.505,
            "map_at_10": 0.505,
            "ndcg_at_10": 0.554,
        },
    )

    hybrid = load_first_existing_json(
        [
            OUTPUTS_DIR / "hybrid_retrieval_hybrid_summary.json",
            OUTPUTS_DIR / "hybrid_retrieval_full_hybrid_summary.json",
            OUTPUTS_DIR / "full_hybrid_summary.json",
            RESULTS_EVAL_DIR / "hybrid_retrieval_hybrid_summary.json",
        ],
        fallback={
            "top1": 0.2780,
            "top3": 0.6195,
            "top5": 0.6195,
            "mrr": 0.428,
            "map_at_10": 0.428,
            "ndcg_at_10": 0.478,
        },
    )

    vocal_plus_original = load_first_existing_json(
        [
            OUTPUTS_DIR / "hybrid_retrieval_vocal_plus_original_summary.json",
            OUTPUTS_DIR / "vocal_plus_original_summary.json",
            RESULTS_EVAL_DIR / "hybrid_retrieval_vocal_plus_original_summary.json",
        ],
        fallback={
            "top1": 0.1667,
            "top3": 0.3333,
            "top5": 0.5667,
            "mrr": 0.333,
            "map_at_10": 0.328,
            "ndcg_at_10": 0.424,
        },
    )

    return vocal_only, hybrid, vocal_plus_original


# ============================================================
# Figure 5.1
# Retrieval Top-K Accuracy Comparison
# ============================================================
def fig_5_1_retrieval_topk_accuracy() -> None:
    vocal_only, hybrid, vocal_plus_original = load_retrieval_results()

    systems = ["Vocal-only", "Hybrid", "Vocal + Original"]

    top1 = [
        percent(safe_get(vocal_only, "top1", "top_1_accuracy")),
        percent(safe_get(hybrid, "top1", "top_1_accuracy")),
        percent(safe_get(vocal_plus_original, "top1", "top_1_accuracy")),
    ]

    top3 = [
        percent(safe_get(vocal_only, "top3", "top_3_accuracy")),
        percent(safe_get(hybrid, "top3", "top_3_accuracy")),
        percent(safe_get(vocal_plus_original, "top3", "top_3_accuracy")),
    ]

    top5 = [
        percent(safe_get(vocal_only, "top5", "top_5_accuracy")),
        percent(safe_get(hybrid, "top5", "top_5_accuracy")),
        percent(safe_get(vocal_plus_original, "top5", "top_5_accuracy")),
    ]

    x = np.arange(len(systems))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, top1, width, label="Top-1")
    bars2 = ax.bar(x, top3, width, label="Top-3")
    bars3 = ax.bar(x + width, top5, width, label="Top-5")

    ax.set_title("Retrieval Top-K Accuracy Comparison", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True)
    style_axes(ax)

    add_bar_labels(ax, bars1, fmt="{:.1f}", offset=1.0)
    add_bar_labels(ax, bars2, fmt="{:.1f}", offset=1.0)
    add_bar_labels(ax, bars3, fmt="{:.1f}", offset=1.0)

    out_path = RESULTS_VIS_DIR / "fig_5_1_retrieval_topk_accuracy.png"
    save_figure(fig, out_path)


# ============================================================
# Figure 5.2
# Retrieval Ranking Quality Comparison
# ============================================================
def fig_5_2_retrieval_ranking_quality() -> None:
    vocal_only, hybrid, vocal_plus_original = load_retrieval_results()

    systems = ["Vocal-only", "Hybrid", "Vocal + Original"]

    mrr = [
        safe_get(vocal_only, "mrr"),
        safe_get(hybrid, "mrr"),
        safe_get(vocal_plus_original, "mrr"),
    ]

    map10 = [
        safe_get(vocal_only, "map_at_10", "map"),
        safe_get(hybrid, "map_at_10", "map"),
        safe_get(vocal_plus_original, "map_at_10", "map"),
    ]

    ndcg10 = [
        safe_get(vocal_only, "ndcg_at_10", "ndcg"),
        safe_get(hybrid, "ndcg_at_10", "ndcg"),
        safe_get(vocal_plus_original, "ndcg_at_10", "ndcg"),
    ]

    x = np.arange(len(systems))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, mrr, width, label="MRR")
    bars2 = ax.bar(x, map10, width, label="MAP@10")
    bars3 = ax.bar(x + width, ndcg10, width, label="NDCG@10")

    ax.set_title("Retrieval Ranking Quality Comparison", fontsize=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True)
    style_axes(ax)

    add_score_labels(ax, bars1, fmt="{:.3f}", offset=0.015)
    add_score_labels(ax, bars2, fmt="{:.3f}", offset=0.015)
    add_score_labels(ax, bars3, fmt="{:.3f}", offset=0.015)

    out_path = RESULTS_VIS_DIR / "fig_5_2_retrieval_ranking_quality.png"
    save_figure(fig, out_path)


# ============================================================
# Figure 5.3
# Neural Baseline Comparison
# ============================================================
def fig_5_3_neural_baseline_comparison() -> None:
    cnn = load_json(RESULTS_EVAL_DIR / "cnn_lstm_evaluation_results.json")
    audio = load_json(RESULTS_EVAL_DIR / "audio_transformer_evaluation_results.json")

    metric_labels = ["Top-1", "MRR", "NDCG@10", "Macro-F1"]

    cnn_values = [
        percent(safe_get(cnn, "top_1_accuracy", "accuracy")),
        percent(safe_get(cnn, "mrr")),
        percent(safe_get(cnn, "ndcg_at_10")),
        percent(safe_get(cnn, "macro_f1")),
    ]

    audio_values = [
        percent(safe_get(audio, "top_1_accuracy", "accuracy")),
        percent(safe_get(audio, "mrr")),
        percent(safe_get(audio, "ndcg_at_10")),
        percent(safe_get(audio, "macro_f1")),
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, cnn_values, width, label="CNN-LSTM")
    bars2 = ax.bar(x + width / 2, audio_values, width, label="Audio Transformer")

    ax.set_title("Neural Baseline Comparison", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True)
    style_axes(ax)

    add_bar_labels(ax, bars1, fmt="{:.1f}", offset=1.0)
    add_bar_labels(ax, bars2, fmt="{:.1f}", offset=1.0)

    out_path = RESULTS_VIS_DIR / "fig_5_3_neural_baseline_comparison.png"
    save_figure(fig, out_path)


# ============================================================
# Figure 5.4 and 5.5
# Copy existing training curve images
# ============================================================
def copy_existing_training_curve(model_name: str, output_filename: str) -> None:
    out_path = RESULTS_VIS_DIR / output_filename

    direct_candidates = [
        RESULTS_VIS_DIR / f"{model_name}_training_curves.png",
        RESULTS_VIS_DIR / f"{model_name}_training_curve.png",
    ]

    glob_candidates = sorted(
        RESULTS_VIS_DIR.glob(f"*{model_name}*training*curve*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    candidates = direct_candidates + glob_candidates

    for source_path in candidates:
        if source_path.exists() and source_path.resolve() != out_path.resolve():
            copy2(source_path, out_path)
            print(f"Saved: {out_path}")
            return

    print(f"Skipping {output_filename}: no existing training curve image found for {model_name}.")


def fig_5_4_cnn_lstm_training_curves() -> None:
    copy_existing_training_curve(
        model_name="cnn_lstm",
        output_filename="fig_5_4_cnn_lstm_training_curves.png",
    )


def fig_5_5_audio_transformer_training_curves() -> None:
    copy_existing_training_curve(
        model_name="audio_transformer",
        output_filename="fig_5_5_audio_transformer_training_curves.png",
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    fig_5_1_retrieval_topk_accuracy()
    fig_5_2_retrieval_ranking_quality()
    fig_5_3_neural_baseline_comparison()
    fig_5_4_cnn_lstm_training_curves()
    fig_5_5_audio_transformer_training_curves()

    print("All Chapter 5 report figures generated successfully.")


if __name__ == "__main__":
    main()