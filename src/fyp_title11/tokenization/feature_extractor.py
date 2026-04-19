"""Feature extractor for Hum2Tune (39-channel version).

Outputs a 39-channel feature tensor per audio segment:
  - 13 MFCC coefficients
  - 13 delta-MFCC  (first-order velocity: captures pitch direction)
  - 13 delta-delta-MFCC (second-order acceleration: captures pitch rate-of-change)

Delta features encode how the melody moves over time -- the core signal for
query-by-humming (Choi et al., 2017; Muller, 2007).

The Mel filterbank is constrained to [fmin, fmax] (65-2093 Hz by default)
to focus on the vocal frequency range and exclude sub-bass / high-frequency
noise that is irrelevant for humming recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import yaml


class FeatureExtractor:
    """Extract fixed-size 39-channel MFCC+delta features for humming recognition.

    Output shape per window: (input_channels, target_len)
    where input_channels = n_mfcc * 3  (MFCC + delta + delta-delta).
    """

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

        # Constrain to vocal frequency range (65 Hz - 2093 Hz) so the
        # Mel filterbank focuses on melodic content rather than broadband audio.
        self.fmin = float(self.conf.get("fmin", self.conf.get("min_freq", 65.0)))
        self.fmax = float(self.conf.get("fmax", self.conf.get("max_freq", 2093.0)))

        self.trim_silence = bool(self.conf.get("trim_silence", True))
        self.trim_top_db = float(self.conf.get("trim_top_db", 20))
        self.window_overlap_ratio = float(self.conf.get("window_overlap_ratio", 0.0))
        self.normalize_features = bool(self.conf.get("normalize_features", True))

        # 3 stacked coefficient sets: MFCC + delta + delta-delta
        self.n_channels = self.n_mfcc * 3
        self.target_len = int(np.ceil((self.sr * self.duration) / self.hop_length))

    @property
    def input_channels(self) -> int:
        """Number of feature channels output per frame (n_mfcc x 3)."""
        return self.n_channels

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_file(self, audio_path) -> List[np.ndarray]:
        """Load an audio file and return a list of feature windows."""
        try:
            audio, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
            return self.process_audio(audio)
        except Exception as exc:
            print(f"Error processing {audio_path}: {exc}")
            return []

    def process_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Process a raw waveform into a list of feature windows."""
        audio = self._prepare_audio(audio)
        return self._extract_windows(audio)

    def process_segment(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio segment directly into one feature array."""
        try:
            segment = self._prepare_audio(audio_segment)
            return self._build_features(segment)
        except Exception as exc:
            print(f"Error in process_segment: {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

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
        """Build a (39, target_len) feature matrix for one audio segment.

        Stacks MFCC + delta + delta-delta, each normalised per-channel.
        The Mel filterbank is restricted to [fmin, fmax] to target vocal
        frequencies and improve discriminability for humming queries.
        """
        try:
            target_samples = int(self.sr * self.duration)

            if len(segment) < target_samples:
                segment = np.pad(segment, (0, target_samples - len(segment)))
            else:
                segment = segment[:target_samples]

            # --- 13 MFCC coefficients ---
            # fmin/fmax restrict the filterbank to the vocal range.
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                fmin=self.fmin,
                fmax=self.fmax,
            )

            # --- 13 delta-MFCC (pitch velocity) ---
            # First-order difference: encodes melodic contour direction
            # (whether pitch is rising, falling, or staying the same).
            mfcc_delta = librosa.feature.delta(mfcc, order=1)

            # --- 13 delta-delta-MFCC (pitch acceleration) ---
            # Second-order difference: encodes rate-of-change of the melody,
            # distinguishing ornaments and vibrato from sustained notes.
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # Stack all three to get a (39, T) feature matrix.
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

            if self.normalize_features:
                features = self._normalize_channels(features)

            # Pad or truncate time axis to a fixed target length.
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
        """Per-channel z-score normalization."""
        normalized = np.zeros_like(features, dtype=np.float32)
        for i in range(features.shape[0]):
            ch = features[i]
            normalized[i] = (ch - ch.mean()) / (ch.std() + 1e-6)
        return normalized
