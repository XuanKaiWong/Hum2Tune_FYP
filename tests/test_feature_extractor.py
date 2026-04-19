"""Feature extractor tests for Hum2Tune.

Tests verify:
- Output shape is always (input_channels, target_len) -- consistent with
  the model's expected input format.
- input_channels == n_mfcc * 3 (MFCC + delta + delta-delta).
- process_file and process_audio produce lists of arrays.
- Normalization produces zero-mean channels.
"""

import numpy as np
import pytest

from src.fyp_title11.tokenization.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def sine_audio(extractor):
    """440 Hz sine wave long enough for one full window."""
    sr = extractor.sr
    duration = extractor.duration + 0.5  # slightly longer than one window
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_input_channels_equals_n_mfcc_times_three(extractor):
    """input_channels should always be n_mfcc * 3."""
    assert extractor.input_channels == extractor.n_mfcc * 3, (
        f"Expected {extractor.n_mfcc * 3} channels, got {extractor.input_channels}"
    )


def test_process_segment_output_shape(extractor, sine_audio):
    """process_segment returns a (input_channels, target_len) array."""
    segment = sine_audio[: int(extractor.sr * 2.0)]
    features = extractor.process_segment(segment)

    assert features is not None
    # Shape must match (input_channels, target_len) -- NOT hardcoded integers.
    assert features.shape[0] == extractor.input_channels, (
        f"Channel dim: expected {extractor.input_channels}, got {features.shape[0]}"
    )
    assert features.shape[1] == extractor.target_len, (
        f"Time dim: expected {extractor.target_len}, got {features.shape[1]}"
    )


def test_process_audio_returns_list(extractor, sine_audio):
    """process_audio returns a non-empty list of feature arrays."""
    result = extractor.process_audio(sine_audio)
    assert isinstance(result, list)
    assert len(result) > 0


def test_process_audio_each_element_correct_shape(extractor, sine_audio):
    """Every element in process_audio output has the correct shape."""
    result = extractor.process_audio(sine_audio)
    for feat in result:
        assert feat.shape == (extractor.input_channels, extractor.target_len)


def test_normalize_makes_channels_near_zero_mean(extractor):
    """With normalize_features=True, each channel should have near-zero mean."""
    extractor.normalize_features = True
    sr = extractor.sr
    t = np.linspace(0, extractor.duration, int(sr * extractor.duration), endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    feat = extractor.process_segment(audio)
    assert feat is not None

    for ch in range(feat.shape[0]):
        channel_mean = feat[ch].mean()
        assert abs(channel_mean) < 0.5, (
            f"Channel {ch} mean {channel_mean:.4f} is not near zero after normalisation"
        )


def test_empty_audio_returns_none_or_empty(extractor):
    """Empty audio should not raise; it should return None or empty list."""
    result_segment = extractor.process_segment(np.array([], dtype=np.float32))
    result_list = extractor.process_audio(np.array([], dtype=np.float32))
    # Either is acceptable -- the key point is no crash.
    assert result_segment is None or isinstance(result_segment, np.ndarray)
    assert isinstance(result_list, list)


@pytest.mark.parametrize("freq_hz", [220, 440, 880])
def test_process_segment_dtype_is_float32(extractor, freq_hz):
    """Output dtype must be float32 to match PyTorch model expectations."""
    sr = extractor.sr
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    feat = extractor.process_segment(audio)
    assert feat is not None
    assert feat.dtype == np.float32
