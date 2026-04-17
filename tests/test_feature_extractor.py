import numpy as np

from src.fyp_title11.tokenization.feature_extractor import FeatureExtractor


def test_process_segment_shape():
    extractor = FeatureExtractor()
    sr = extractor.sr
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    features = extractor.process_segment(audio)
    assert features is not None
    assert features.shape[0] == 40
    assert features.shape[1] == extractor.target_len
