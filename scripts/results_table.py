from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Display columns and their labels
METRIC_COLS = [
    ("Top-1",    "Top-1"),
    ("Top-3",    "Top-3"),
    ("Top-5",    "Top-5"),
    ("MRR",      "MRR"),
    ("MAP@10",   "MAP@10"),
    ("NDCG@10",  "NDCG@10"),
    ("Macro-F1", "Macro-F1"),
]

# For each system, list the JSON file and the label separator comment.
# Retrieval rows are separated from classifier rows by a midrule in LaTeX.
SYSTEM_SOURCES = [
    # --- Classifier systems (evaluate.py output) ----------------------------
    (
        "CNN-LSTM (standard)",
        ROOT / "results" / "evaluations" / "cnn_lstm_evaluation_results.json",
        "classifier",
    ),
    (
        "CNN-LSTM + Curriculum",
        ROOT / "results" / "evaluations" / "cnn_lstm_curriculum_evaluation_results.json",
        "classifier",
    ),
    (
        "Audio Transformer",
        ROOT / "results" / "evaluations" / "audio_transformer_evaluation_results.json",
        "classifier",
    ),
    # --- Retrieval systems (hybrid_retrieval.py output) ---------------------
    (
        "Hybrid DTW (vocal only)",
        ROOT / "outputs" / "hybrid_retrieval_vocal_only_summary.json",
        "retrieval",
    ),
    (
        "Hybrid DTW (vocal + original)",
        ROOT / "outputs" / "hybrid_retrieval_vocal_plus_original_summary.json",
        "retrieval",
    ),
    (
        "Hybrid DTW (full hybrid)",
        ROOT / "outputs" / "hybrid_retrieval_hybrid_summary.json",
        "retrieval",
    ),
]


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Could not read {path.name}: {exc}", file=sys.stderr)
        return None


# Mapping from our canonical column name to the actual JSON key(s) to try.
# Classifier evals use long-form keys; retrieval summaries use short-form.
# get_metric() tries each alias in order and returns the first hit.
KEY_ALIASES: dict[str, list[str]] = {
    "Top-1":    ["top_1_accuracy", "top1"],
    "Top-3":    ["top_3_accuracy", "top3"],
    "Top-5":    ["top_5_accuracy", "top5"],
    "MRR":      ["mrr"],
    "MAP@10":   ["map_at_10"],
    "NDCG@10":  ["ndcg_at_10"],
    "Macro-F1": ["macro_f1"],
}


def get_metric(data: dict, col_name: str) -> float | None:
    """Try each alias for col_name until one is found in data."""
    for key in KEY_ALIASES.get(col_name, [col_name]):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def build_table() -> list[dict]:
    rows = []
    for name, path, group in SYSTEM_SOURCES:
        data = load_result(path)
        row = {"System": name, "_found": data is not None, "_group": group}
        for col_name, _ in METRIC_COLS:
            row[col_name] = get_metric(data, col_name) if data else None
        rows.append(row)
    return rows


def fmt(val: float | None) -> str:
    if val is None:
        return "  --  "
    return f"{val * 100:6.2f}"


def print_plain(rows: list[dict]) -> None:
    col_labels = [label for _, label in METRIC_COLS]
    name_w = max(len(r["System"]) for r in rows) + 2
    header = f"{'System':<{name_w}}" + "".join(f"{c:>8}" for c in col_labels)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    prev_group = None
    for row in rows:
        if prev_group and row["_group"] != prev_group:
            print(sep)   # visual separator between classifiers and retrieval
        prev_group = row["_group"]
        line = f"{row['System']:<{name_w}}"
        for col_name, _ in METRIC_COLS:
            line += fmt(row[col_name])
        if not row["_found"]:
            line += "  (file not found)"
        print(line)
    print(sep)
    print("Values are percentages (x 100). '--' = results file not yet generated.")


def print_latex(rows: list[dict]) -> None:
    col_labels = [label for _, label in METRIC_COLS]
    cols = "l" + "c" * len(col_labels)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comparison of melody recognition systems on the Hum2Tune dataset.}")
    print(r"\label{tab:results}")
    print(rf"\begin{{tabular}}{{{cols}}}")
    print(r"\toprule")
    print("System & " + " & ".join(col_labels) + r" \\")
    print(r"\midrule")
    prev_group = None
    for row in rows:
        if prev_group and row["_group"] != prev_group:
            print(r"\midrule")
        prev_group = row["_group"]
        cells = [row["System"]]
        for col_name, _ in METRIC_COLS:
            v = row[col_name]
            cells.append(f"{v * 100:.2f}" if v is not None else "--")
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_csv(rows: list[dict]) -> None:
    col_labels = [label for _, label in METRIC_COLS]
    print("System,Group," + ",".join(col_labels))
    for row in rows:
        cells = [row["System"], row["_group"]]
        for col_name, _ in METRIC_COLS:
            v = row[col_name]
            cells.append(f"{v * 100:.4f}" if v is not None else "")
        print(",".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dissertation results table")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--latex", action="store_true", help="Output LaTeX tabular")
    group.add_argument("--csv",   action="store_true", help="Output CSV")
    args = parser.parse_args()

    rows = build_table()
    found = sum(1 for r in rows if r["_found"])
    print(f"[results_table] Found {found}/{len(rows)} result files.\n", file=sys.stderr)

    if args.latex:
        print_latex(rows)
    elif args.csv:
        print_csv(rows)
    else:
        print_plain(rows)


if __name__ == "__main__":
    main()
