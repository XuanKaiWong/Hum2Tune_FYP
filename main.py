import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

MODEL_CHOICES = ["cnn_lstm"]


def run_streamlit():
    web_app_path = project_root / "src" / "fyp_title11" / "app.py"
    if not web_app_path.exists():
        print(f"Web app not found at {web_app_path}")
        return
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(web_app_path)], check=False)


def main():
    parser = argparse.ArgumentParser(description="Hum2Tune management console")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("setup", help="Setup project directories and config")

    parser_dataset = subparsers.add_parser("dataset", help="Dataset operations")
    parser_dataset.add_argument(
        "--create",
        action="store_true",
        help="Create processed dataset from raw humming audio",
    )

    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model to train",
    )

    parser_predict = subparsers.add_parser("predict", help="Predict song from audio")
    parser_predict.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser_predict.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model to use for prediction",
    )
    parser_predict.add_argument("--top-k", type=int, default=3)

    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    parser_eval.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=MODEL_CHOICES,
        help="Model to evaluate",
    )

    parser_retrieve = subparsers.add_parser(
        "retrieve",
        help="Run hybrid DTW retrieval using humming queries and song references",
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
        help="Folder containing humming query subfolders",
    )
    parser_retrieve.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top candidates to keep per query",
    )

    subparsers.add_parser("web", help="Launch web interface")
    subparsers.add_parser("test", help="Run unit tests")

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
        train_model.train_pipeline(model_name=args.model)

    elif args.command == "predict":
        from scripts import predict
        predict.predict(args.audio, model_name=args.model, top_k=args.top_k)

    elif args.command == "evaluate":
        from scripts import evaluate
        try:
            evaluate.evaluate(model_name=args.model)
        except TypeError:
            evaluate.evaluate()

    elif args.command == "retrieve":
        from scripts import hybrid_retrieval
        hybrid_retrieval.run(
            mode=args.mode,
            query_root=args.query_root,
            top_k=args.top_k,
        )

    elif args.command == "web":
        run_streamlit()

    elif args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=False)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()