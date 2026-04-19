"""Hum2Tune - unified management console.

Usage:
    python main.py setup
    python main.py dataset --create
    python main.py train --model cnn_lstm
    python main.py train --model audio_transformer
    python main.py train --model cnn_lstm --curriculum
    python main.py evaluate --model cnn_lstm
    python main.py evaluate --model audio_transformer
    python main.py predict --audio path/to/hum.wav --model cnn_lstm
    python main.py retrieve --mode vocal_only
    python main.py web
    python main.py test
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Belt-and-suspenders: strip UTF-8 BOM from this file if Windows zip extraction
# added one. The BOM bytes (0xEF 0xBB 0xBF) are non-ASCII and cause
# UnicodeEncodeError on narrow Windows terminals when argparse prints --help.
_self = Path(__file__).resolve()
_raw = _self.read_bytes()
if _raw.startswith(b"\xef\xbb\xbf"):
    _self.write_bytes(_raw[3:])

# Ensure help text and print() output does not crash on Windows terminals
# that default to cp1252 or similar narrow encodings.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

# Keep in sync with MODEL_REGISTRY in scripts/train_model.py
# and SUPPORTED_MODELS in scripts/evaluate.py.
MODEL_CHOICES = ["cnn_lstm", "audio_transformer"]


def run_streamlit() -> None:
    web_app_path = project_root / "src" / "fyp_title11" / "app.py"
    if not web_app_path.exists():
        print(f"Web app not found at {web_app_path}")
        return
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(web_app_path)],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hum2Tune management console",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup
    subparsers.add_parser("setup", help="Set up project directories and config")

    # dataset
    parser_dataset = subparsers.add_parser("dataset", help="Dataset operations")
    parser_dataset.add_argument(
        "--create",
        action="store_true",
        help="Create processed dataset from raw humming audio",
    )

    # train
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model architecture to train",
    )
    parser_train.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help=(
            "Use two-stage curriculum learning (vocal-only -> polyphonic). "
            "Requires data/processed/vocal_only/ to exist."
        ),
    )

    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    parser_eval.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model architecture to evaluate",
    )
    parser_eval.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help=(
            "Evaluate the curriculum checkpoint (best_model_curriculum.pth). "
            "Results go to a separate file so the standard baseline is not overwritten."
        ),
    )

    # predict
    parser_predict = subparsers.add_parser(
        "predict",
        help="Predict song from a humming audio file",
    )
    parser_predict.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to the audio file (wav/mp3/m4a)",
    )
    parser_predict.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model architecture to use for prediction",
    )
    parser_predict.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display",
    )

    # retrieve
    parser_retrieve = subparsers.add_parser(
        "retrieve",
        help="Run hybrid DTW retrieval against the full song reference database",
    )
    parser_retrieve.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["vocal_only", "vocal_plus_original", "hybrid"],
        help="Retrieval fusion mode",
    )
    parser_retrieve.add_argument(
        "--query-root",
        type=str,
        default="data/Humming Audio",
        help="Folder containing per-song humming query subfolders",
    )
    parser_retrieve.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to return per query",
    )
    parser_retrieve.add_argument(
        "--shortlist",
        type=int,
        default=20,
        help="Coarse chroma shortlist size (top-N before fine re-ranking)",
    )
    parser_retrieve.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit evaluation to first N queries (useful for quick benchmark checks)",
    )
    parser_retrieve.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save partial results every N queries (for long runs)",
    )
    parser_retrieve.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable reference feature cache (forces re-extraction)",
    )

    # web
    subparsers.add_parser("web", help="Launch the Streamlit demo app")

    # test
    subparsers.add_parser("test", help="Run the full test suite via pytest")

    args = parser.parse_args()

    if args.command == "setup":
        from scripts import setup_environment
        setup_environment.setup_project()

    elif args.command == "dataset":
        if args.create:
            from scripts import prepare_dataset
            prepare_dataset.prepare_data()
        else:
            parser_dataset.print_help()

    elif args.command == "train":
        from scripts import train_model
        if args.curriculum:
            train_model.curriculum_train_pipeline(model_name=args.model)
        else:
            train_model.train_pipeline(model_name=args.model)

    elif args.command == "evaluate":
        from scripts import evaluate
        evaluate.evaluate(model_name=args.model, curriculum=args.curriculum)

    elif args.command == "predict":
        from scripts import predict
        predict.predict(
            audio_path=args.audio,
            model_name=args.model,
            top_k=args.top_k,
        )

    elif args.command == "retrieve":
        from scripts import hybrid_retrieval
        hybrid_retrieval.run(
            mode=args.mode,
            query_root=args.query_root,
            top_k=args.top_k,
            shortlist=args.shortlist,
            max_queries=args.max_queries,
            save_every=args.save_every,
            use_cache=not args.no_cache,
        )

    elif args.command == "web":
        run_streamlit()

    elif args.command == "test":
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            check=False,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
