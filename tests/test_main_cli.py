import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(project_root / "main.py")] + args,
        capture_output=True,
        text=True,
    )


def test_main_help_exits_cleanly():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "retrieve" in result.stdout


def test_train_help_shows_audio_transformer():
    result = _run(["train", "--help"])
    assert result.returncode == 0
    assert "audio_transformer" in result.stdout


def test_train_help_shows_curriculum_flag():
    result = _run(["train", "--help"])
    assert result.returncode == 0
    assert "--curriculum" in result.stdout


def test_evaluate_help_shows_curriculum_flag():
    result = _run(["evaluate", "--help"])
    assert result.returncode == 0
    assert "--curriculum" in result.stdout


def test_evaluate_help_shows_audio_transformer():
    result = _run(["evaluate", "--help"])
    assert result.returncode == 0
    assert "audio_transformer" in result.stdout


def test_retrieve_help_shows_all_benchmark_flags():
    """All flags needed for the dissertation benchmark must be exposed."""
    result = _run(["retrieve", "--help"])
    assert result.returncode == 0
    for flag in ("--shortlist", "--max-queries", "--save-every", "--no-cache"):
        assert flag in result.stdout, f"Missing retrieve flag: {flag}"


def test_predict_help_shows_audio_transformer():
    result = _run(["predict", "--help"])
    assert result.returncode == 0
    assert "audio_transformer" in result.stdout


def test_model_choices_in_main_module():
    """MODEL_CHOICES must include both architectures."""
    sys.path.insert(0, str(project_root))
    import importlib
    main_mod = importlib.import_module("main")
    assert "cnn_lstm" in main_mod.MODEL_CHOICES
    assert "audio_transformer" in main_mod.MODEL_CHOICES


def test_no_command_prints_help():
    result = _run([])
    # Should print help and exit 0 (or 2 for missing subcommand -- both OK)
    assert "train" in result.stdout or "train" in result.stderr
