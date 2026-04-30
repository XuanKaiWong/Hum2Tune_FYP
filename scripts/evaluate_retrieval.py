from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from scripts.hybrid_retrieval import (
    discover_queries,
    discover_references,
    rank_query,
    mode_weights,
    safe_extract_reference_features,
    summarize,
    compute_rank_metrics,
)

import pandas as pd


def evaluate_retrieval(
    mode: str = "vocal_only",
    query_root: str = "data/Humming Audio",
    song_filter: list[str] | None = None,
    shortlist: int = 20,
    top_k: int = 5,
    use_cache: bool = True,
) -> dict:

    query_root_path = (project_root / query_root).resolve()
    weights = mode_weights(mode)

    queries = discover_queries(query_root_path)
    refs = discover_references()

    # Apply song filter
    if song_filter:
        filter_lower = {s.lower() for s in song_filter}
        queries = [q for q in queries if q["true_title_display"].lower() in filter_lower]
        refs = {k: v for k, v in refs.items()
                if v.title_display.lower() in filter_lower}

    if not queries:
        print(f"No queries found under {query_root_path}", file=sys.stderr)
        return {}
    if not refs:
        print("No references found.", file=sys.stderr)
        return {}

    print(f"Queries: {len(queries)} | References: {len(refs)} | Mode: {mode}")

    # Pre-compute reference features
    ref_features = {}
    for key, ref in refs.items():
        ref_features[key] = safe_extract_reference_features(ref, use_cache=use_cache)

    ranked_rows = []
    for q in queries:
        try:
            cand_df = rank_query(
                query_path=q["query_path"],
                ref_features=ref_features,
                weights=weights,
                shortlist_size=shortlist,
                top_k=len(refs),   # rank all refs for proper metric computation
            )
            if cand_df.empty:
                continue
            true_key = q["true_title_key"]
            rank, top1, top3, top5, mrr = compute_rank_metrics(cand_df, true_key)
            ranked_rows.append({
                "query_name": q["query_path"].name,
                "true_title_display": q["true_title_display"],
                "rank_of_true_song": rank,
                "top1": top1, "top3": top3, "top5": top5, "mrr": mrr,
            })
        except Exception as exc:
            print(f"Skipping {q['query_path'].name}: {exc}", file=sys.stderr)

    if not ranked_rows:
        print("No results computed.", file=sys.stderr)
        return {}

    df = pd.DataFrame(ranked_rows)
    summary = summarize(df)

    print("\n=== RETRIEVAL EVALUATION RESULTS ===")
    print(f"Queries evaluated: {summary['count']}")
    print(f"  Top-1:    {summary['top1']*100:.2f}%")
    print(f"  Top-3:    {summary['top3']*100:.2f}%")
    print(f"  Top-5:    {summary['top5']*100:.2f}%")
    print(f"  MRR:      {summary['mrr']:.4f}")
    if summary.get("map_at_10"):
        print(f"  MAP@10:   {summary['map_at_10']:.4f}")
    if summary.get("ndcg_at_10"):
        print(f"  NDCG@10:  {summary['ndcg_at_10']:.4f}")

    # Save
    out_dir = project_root / "results" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval_{mode}_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hybrid DTW retrieval")
    parser.add_argument("--mode", type=str, default="vocal_only",
                        choices=["vocal_only", "vocal_plus_original", "hybrid"])
    parser.add_argument("--query-root", type=str, default="data/Humming Audio")
    parser.add_argument("--songs", type=str, nargs="*", default=None,
                        help="Restrict evaluation to specific songs")
    parser.add_argument("--shortlist", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    evaluate_retrieval(
        mode=args.mode,
        query_root=args.query_root,
        song_filter=args.songs,
        shortlist=args.shortlist,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
