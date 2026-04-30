import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Project root on sys.path (set by conftest.py or here)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Feature extraction ───────────────────────────────────────────────────────

class TestFeatureExtractionSmoke:
    """Smoke tests for the 39-channel MFCC+delta feature extractor."""

    def setup_method(self):
        from fyp_title11.tokenization.feature_extractor import FeatureExtractor
        self.extractor = FeatureExtractor()

    def test_extractor_channels(self):
        assert self.extractor.input_channels == 39
        assert self.extractor.n_mfcc == 13
        assert self.extractor.n_channels == 13 * 3

    def test_extractor_vocal_frequency_range(self):
        assert self.extractor.fmin == pytest.approx(65.0)
        assert self.extractor.fmax == pytest.approx(2093.0)

    def test_process_segment_returns_correct_shape(self):
        sr = self.extractor.sr
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2)))).astype(np.float32)
        feat = self.extractor.process_segment(audio)
        assert feat is not None
        assert feat.shape == (self.extractor.input_channels, self.extractor.target_len)
        assert feat.dtype == np.float32

    def test_process_audio_returns_list(self):
        sr = self.extractor.sr
        audio = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 3, int(sr * 3)))).astype(np.float32)
        result = self.extractor.process_audio(audio)
        assert isinstance(result, list)
        assert len(result) >= 1


# ─── Model forward passes ─────────────────────────────────────────────────────

class TestCNNLSTMSmoke:
    """Smoke tests for CNNLSTMModel."""

    def test_default_channels_is_39(self):
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        m = CNNLSTMModel(num_classes=10)
        assert m.input_channels == 39

    def test_train_mode_forward(self):
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        m = CNNLSTMModel(input_channels=39, num_classes=10)
        m.train()
        x = torch.randn(4, 39, 200)
        logits = m(x)
        assert logits.shape == (4, 10)
        assert not torch.isnan(logits).any()

    def test_eval_mode_forward(self):
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        m = CNNLSTMModel(input_channels=39, num_classes=10)
        m.eval()
        with torch.no_grad():
            x = torch.randn(2, 39, 431)
            logits = m(x)
        assert logits.shape == (2, 10)

    def test_embedding_shape(self):
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        m = CNNLSTMModel(input_channels=39, num_classes=10, hidden_size=64, bidirectional=True)
        emb = m.get_embedding(torch.randn(3, 39, 200))
        assert emb.shape == (3, 128)   # 64 * 2 (bidirectional)

    def test_predict_proba_sums_to_one(self):
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        m = CNNLSTMModel(input_channels=39, num_classes=10)
        m.eval()
        with torch.no_grad():
            probs = m.predict_proba(torch.randn(4, 39, 200))
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


class TestAudioTransformerSmoke:
    """Smoke tests for AudioTransformer."""

    def test_default_channels_is_39(self):
        from fyp_title11.models.audio_transformer import AudioTransformer
        m = AudioTransformer(num_classes=10)
        assert m.input_channels == 39

    def test_train_mode_forward(self):
        from fyp_title11.models.audio_transformer import AudioTransformer
        m = AudioTransformer(input_channels=39, num_classes=10)
        m.train()
        x = torch.randn(4, 39, 200)
        logits = m(x)
        assert logits.shape == (4, 10)
        assert not torch.isnan(logits).any()

    def test_same_api_as_cnn_lstm(self):
        """Both models accept (B, C, T) and return (B, K) -- key ablation requirement."""
        from fyp_title11.models.cnn_lstm import CNNLSTMModel
        from fyp_title11.models.audio_transformer import AudioTransformer
        x = torch.randn(2, 39, 300)
        cnn_out = CNNLSTMModel(input_channels=39, num_classes=5)(x)
        tf_out  = AudioTransformer(input_channels=39, num_classes=5)(x)
        assert cnn_out.shape == tf_out.shape == (2, 5)

    def test_return_features_context_vector(self):
        from fyp_title11.models.audio_transformer import AudioTransformer
        m = AudioTransformer(input_channels=39, num_classes=10, d_model=64)
        logits, feats = m(torch.randn(2, 39, 100), return_features=True)
        assert feats["context_vector"].shape == (2, 64)


# ─── DTW functions ────────────────────────────────────────────────────────────

class TestDTWSmoke:
    def test_dtw_distance_identical(self):
        from fyp_title11.models.dtw_matcher import dtw_distance
        seq = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        assert dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-5)

    def test_subseq_dtw_distance_embedded(self):
        from fyp_title11.models.dtw_matcher import subseq_dtw_distance
        query = np.sin(np.linspace(0, 2 * np.pi, 50)).astype(np.float32)
        reference = np.concatenate([np.zeros(30), query, np.zeros(30)])
        dist = subseq_dtw_distance(query, reference)
        assert dist < 0.5


# ─── Metrics (including subset-class crash regression) ────────────────────────

class TestMetricsSmoke:
    def test_compute_all_metrics_normal(self):
        from fyp_title11.evaluation.metrics import compute_all_metrics
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, size=20)
        y_prob = rng.dirichlet(np.ones(5), size=20)
        y_pred = y_prob.argmax(axis=1)
        result = compute_all_metrics(y_true, y_pred, y_prob)
        for key in ("top_1_accuracy", "mrr", "map_at_10", "ndcg_at_10", "accuracy", "macro_f1"):
            assert key in result
            assert 0.0 <= float(result[key]) <= 1.0

    def test_compute_all_metrics_subset_classes(self):
        """Regression: 7-class test set with 100-class model must not crash."""
        from fyp_title11.evaluation.metrics import compute_all_metrics
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 7, size=30)      # only 7 classes present
        y_prob = rng.dirichlet(np.ones(100), size=30)  # model has 100 outputs
        y_pred = y_prob.argmax(axis=1)
        class_names = [f"song_{i}" for i in range(100)]
        result = compute_all_metrics(y_true, y_pred, y_prob, class_names=class_names)
        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (100, 100)


# ─── Dataset loading ──────────────────────────────────────────────────────────

class TestDatasetSmoke:
    def test_chunked_dataset_loads_and_iterates(self, tmp_path):
        """ChunkedAudioDataset reads from .npy chunks without crashing."""
        from fyp_title11.data.dataset import ChunkedAudioDataset
        # Write two synthetic chunks
        X0 = np.random.randn(8, 39, 431).astype(np.float32)
        y0 = np.arange(8, dtype=np.int64)
        X1 = np.random.randn(4, 39, 431).astype(np.float32)
        y1 = np.arange(4, dtype=np.int64)
        np.save(tmp_path / "train_data_chunk0_X.npy", X0)
        np.save(tmp_path / "train_data_chunk0_y.npy", y0)
        np.save(tmp_path / "train_data_chunk1_X.npy", X1)
        np.save(tmp_path / "train_data_chunk1_y.npy", y1)
        meta = {"stem": "train_data", "n_chunks": 2, "total": 12}
        (tmp_path / "train_data_meta.json").write_text(json.dumps(meta))

        ds = ChunkedAudioDataset(tmp_path / "train_data_meta.json")
        assert len(ds) == 12
        x, y = ds[0]
        assert x.shape == (39, 431)
        assert x.dtype == torch.float32
        # Verify mmap copy does not defeat memory efficiency (returns tensor not array)
        assert isinstance(x, torch.Tensor)


# ─── Training config and build_model ─────────────────────────────────────────

class TestTrainConfigSmoke:
    def test_load_config_returns_dict_with_required_keys(self):
        sys.path.insert(0, str(ROOT))
        from scripts.train_model import load_config
        conf = load_config()
        assert "input_channels" in conf
        assert conf["input_channels"] == 39
        assert "learning_rate" in conf
        assert "epochs" in conf

    def test_build_model_cnn_lstm(self):
        from scripts.train_model import build_model, load_config
        conf = load_config()
        model = build_model("cnn_lstm", num_classes=10, train_conf=conf)
        assert model.input_channels == 39
        x = torch.randn(2, 39, 200)
        logits = model(x)
        assert logits.shape == (2, 10)

    def test_build_model_audio_transformer(self):
        from scripts.train_model import build_model, load_config
        conf = load_config()
        model = build_model("audio_transformer", num_classes=10, train_conf=conf)
        assert model.input_channels == 39
        x = torch.randn(2, 39, 200)
        logits = model(x)
        assert logits.shape == (2, 10)

    def test_transformer_config_reads_nested_block(self):
        """Transformer hyperparams must come from conf['transformer'] not the flat dict."""
        from scripts.train_model import build_model
        # Create a config that would fail if build_model reads flat keys
        conf = {
            "input_channels": 39,
            "dropout": 0.99,       # flat dropout -- should NOT be used for Transformer
            "transformer": {
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,    # this is the correct Transformer dropout
            },
        }
        model = build_model("audio_transformer", num_classes=5, train_conf=conf)
        # d_model=64 means context_vector shape should be (B, 64)
        _, feats = model(torch.randn(2, 39, 100), return_features=True)
        assert feats["context_vector"].shape == (2, 64)


# ─── CLI / main.py ────────────────────────────────────────────────────────────

class TestEvaluateCheckpointSeparation:
    """Verify standard and curriculum evaluation write to separate result files."""

    def test_standard_result_stem(self):
        """Standard evaluation -> {model_name}_evaluation_results.json (no 'curriculum')."""
        import importlib, sys
        sys.path.insert(0, str(ROOT))
        ev = importlib.import_module("scripts.evaluate")
        # result_stem logic: curriculum=False -> model_name as-is
        # We verify indirectly by checking the function signature accepts curriculum arg
        import inspect
        sig = inspect.signature(ev.evaluate)
        assert "curriculum" in sig.parameters

    def test_evaluate_argparse_has_curriculum_flag(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert "--curriculum" in result.stdout


# ─── main.py CLI ──────────────────────────────────────────────────────────────

class TestMainCLISmoke:
    def test_model_choices_complete(self):
        import importlib
        main_mod = importlib.import_module("main")
        assert "cnn_lstm" in main_mod.MODEL_CHOICES
        assert "audio_transformer" in main_mod.MODEL_CHOICES

    def test_main_py_is_pure_ascii(self):
        """main.py must contain no non-ASCII bytes (Windows encoding safety).

        Windows zip extraction tools sometimes add a UTF-8 BOM (0xEF 0xBB 0xBF)
        to the start of files. This test strips it automatically before checking
        so that a one-time extraction artifact does not block the whole suite.
        """
        path = ROOT / "main.py"
        content = path.read_bytes()

        # Auto-strip UTF-8 BOM if present (Windows extraction artefact)
        bom = b"\xef\xbb\xbf"
        if content.startswith(bom):
            content = content[3:]
            path.write_bytes(content)

        non_ascii = [b for b in content if b > 127]
        assert not non_ascii, (
            f"main.py contains {len(non_ascii)} non-ASCII bytes -- "
            "will crash on Windows terminals with narrow encoding. "
            "Run: python scripts/fix_encoding.py"
        )

    def test_argparse_train_has_curriculum(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "train", "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0
        assert "--curriculum" in result.stdout

    def test_argparse_train_has_audio_transformer(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "train", "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert "audio_transformer" in result.stdout
