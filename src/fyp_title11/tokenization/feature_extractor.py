"""Feature extractor for Hum2Tune (13-channel small-data version).

This cleaned version outputs a 13-channel feature tensor:
- 13 MFCC only

It is aligned with the current final FYP setup:
- CNN-LSTM input_channels = 13
- small-data training pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import yaml


class FeatureExtractor:
    """Extract fixed-size 13-channel MFCC features for humming recognition."""

    def __init__(self, config_path: str = "config/audio_config.yaml"):
        root = Path(__file__).parent.parent.parent.parent
        full_conf_path = root / config_path

        if full_conf_path.exists():
            with open(full_conf_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            self.conf = loaded.get("audio", loaded) if isinstance(loaded, dict) else {}
        else:
            self.conf = {}

        self.sr = int(self.conf.get("sample_rate", 22050))
        self.duration = float(
            self.conf.get("duration", self.conf.get("max_duration", 10.0))
        )
        self.hop_length = int(self.conf.get("hop_length", 512))
        self.n_mfcc = int(self.conf.get("n_mfcc", 13))
        self.n_fft = int(self.conf.get("n_fft", 2048))
        self.fmin = float(self.conf.get("fmin", self.conf.get("min_freq", 65.0)))
        self.fmax = float(self.conf.get("fmax", self.conf.get("max_freq", 2093.0)))
        self.trim_silence = bool(self.conf.get("trim_silence", True))
        self.trim_top_db = float(self.conf.get("trim_top_db", 20))
        self.window_overlap_ratio = float(self.conf.get("window_overlap_ratio", 0.0))
        self.normalize_features = bool(self.conf.get("normalize_features", True))

        self.n_channels = self.n_mfcc
        self.target_len = int(np.ceil((self.sr * self.duration) / self.hop_length))

    @property
    def input_channels(self) -> int:
        return self.n_channels

    def process_file(self, audio_path) -> List[np.ndarray]:
        try:
            audio, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
            return self.process_audio(audio)
        except Exception as exc:
            print(f"Error processing {audio_path}: {exc}")
            return []

    def process_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        audio = self._prepare_audio(audio)
        return self._extract_windows(audio)

    def process_segment(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        try:
            segment = self._prepare_audio(audio_segment)
            return self._build_features(segment)
        except Exception as exc:
            print(f"Error in process_segment: {exc}")
            return None

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            return audio

        if self.trim_silence:
            trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_top_db)
            if trimmed.size > 0:
                audio = trimmed

        return audio

    def _extract_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        features_list: List[np.ndarray] = []

        samples_per_window = int(self.sr * self.duration)
        overlap_ratio = min(max(self.window_overlap_ratio, 0.0), 0.95)
        stride_samples = max(1, int(samples_per_window * (1.0 - overlap_ratio)))

        if len(audio) <= samples_per_window:
            feat = self._build_features(audio)
            return [feat] if feat is not None else []

        start_s = 0
        while start_s < len(audio):
            end_s = start_s + samples_per_window
            segment = audio[start_s:end_s]

            if segment.size == 0:
                break

            feat = self._build_features(segment)
            if feat is not None:
                features_list.append(feat)

            if end_s >= len(audio):
                break

            start_s += stride_samples

        return features_list

    def _build_features(self, segment: np.ndarray) -> Optional[np.ndarray]:
        try:
            target_samples = int(self.sr * self.duration)

            if len(segment) < target_samples:
                segment = np.pad(segment, (0, target_samples - len(segment)))
            else:
                segment = segment[:target_samples]

            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )

            features = mfcc

            if self.normalize_features:
                features = self._normalize_channels(features)

            if features.shape[1] < self.target_len:
                features = np.pad(
                    features,
                    ((0, 0), (0, self.target_len - features.shape[1])),
                )
            else:
                features = features[:, : self.target_len]

            return features.astype(np.float32)

        except Exception as exc:
            print(f"Error in _build_features: {exc}")
            return None

    def _normalize_channels(self, features: np.ndarray) -> np.ndarray:
        normalised = np.zeros_like(features, dtype=np.float32)
        for i in range(features.shape[0]):
            ch = features[i]
            normalised[i] = (ch - ch.mean()) / (ch.std() + 1e-6)
        return normalised