"""Generate the ablation study table for the dissertation.

Shows the contribution of each component by comparing variants that
differ in exactly one dimension. Handles both classifier eval JSONs
(top_1_accuracy etc.) and retrieval summary JSONs (top1 etc.)
automatically via the same key-alias lookup used in results_table.py.

Usage:
    python scripts/ablation_table.py
    python scripts/ablation_table.py --latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ABLATION_SOURCES = [
    # (label, results_file, rq_note)
    (
        "CNN-LSTM, 13-ch MFCC only (old baseline)",
        ROOT / "results" / "evaluations" / "cnn_lstm_13ch_evaluation_results.json",
        "RQ1 reference: features without deltas",
    ),
    (
        "CNN-LSTM, 39-ch MFCC+delta",
        ROOT / "results" / "evaluations" / "cnn_lstm_evaluation_results.json",
        "RQ1: impact of delta features",
    ),
    (
        "CNN-LSTM, 39-ch + curriculum",
        ROOT / "results" / "evaluations" / "cnn_lstm_curriculum_evaluation_results.json",
        "RQ4: impact of curriculum learning",
    ),
    (
        "Audio Transformer, 39-ch",
        ROOT / "results" / "evaluations" / "audio_transformer_evaluation_results.json",
        "RQ3: architecture comparison",
    ),
    (
        "Hybrid DTW (vocal pitch only)",
        ROOT / "outputs" / "hybrid_retrieval_vocal_only_summary.json",
        "RQ2: retrieval, pitch contour only",
    ),
    (
        "Hybrid DTW (vocal pitch + chroma)",
        ROOT / "outputs" / "hybrid_retrieval_vocal_plus_original_summary.json",
        "RQ2: retrieval, pitch + chroma fusion",
    ),
    (
        "Hybrid DTW (full hybrid fusion)",
        ROOT / "outputs" / "hybrid_retrieval_hybrid_summary.json",
        "RQ2: retrieval, all signals fused",
    ),
]

COLS = [
    ("Top-1",    "Top-1"),
    ("Top-5",    "Top-5"),
    ("MRR",      "MRR"),
    ("Macro-F1", "Macro-F1"),
]

KEY_ALIASES: dict[str, list[str]] = {
    "Top-1":    ["top_1_accuracy", "top1"],
    "Top-5":    ["top_5_accuracy", "top5"],
    "MRR":      ["mrr"],
    "Macro-F1": ["macro_f1"],
}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_metric(data: dict, col_name: str) -> float | None:
    for key in KEY_ALIASES.get(col_name, [col_name]):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def fmt(val: float | None) -> str:
    return f"{val * 100:6.2f}" if val is not None else "  --  "


def print_plain(rows: list) -> None:
    col_labels = [c[1] for c in COLS]
    name_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Variant':<{name_w}}" + "".join(f"{c:>8}" for c in col_labels) + "  Note"
    sep = "-" * (len(header) + 10)
    print(sep)
    print(header)
    print(sep)
    for name, data, note in rows:
        line = f"{name:<{name_w}}"
        for col_name, _ in COLS:
            v = get_metric(data, col_name) if data else None
            line += fmt(v)
        line += f"  {note}" if data else "  (not yet run)"
        print(line)
    print(sep)


def print_latex(rows: list) -> None:
    col_labels = [c[1] for c in COLS]
    cols = "l" + "c" * len(col_labels) + "l"
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Ablation study: contribution of each system component.}")
    print(r"\label{tab:ablation}")
    print(rf"\begin{{tabular}}{{{cols}}}")
    print(r"\toprule")
    print("Variant & " + " & ".join(col_labels) + r" & Research Question \\")
    print(r"\midrule")
    prev_rq = None
    for name, data, note in rows:
        rq = note.split(":")[0]
        if prev_rq and rq != prev_rq:
            print(r"\midrule")
        prev_rq = rq
        cells = [name]
        for col_name, _ in COLS:
            v = get_metric(data, col_name) if data else None
            cells.append(f"{v * 100:.2f}" if v is not None else "--")
        cells.append(note)
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation study table")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tabular")
    args = parser.parse_args()

    rows = [(label, load(path), note) for label, path, note in ABLATION_SOURCES]
    found = sum(1 for _, d, _ in rows if d is not None)
    print(f"[ablation_table] Found {found}/{len(rows)} result files.\n", file=sys.stderr)

    if args.latex:
        print_latex(rows)
    else:
        print_plain(rows)


if __name__ == "__main__":
    main()
