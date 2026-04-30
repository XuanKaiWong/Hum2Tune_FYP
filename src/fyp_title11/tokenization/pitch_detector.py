import numpy as np
import librosa
import torch
from typing import Tuple, Optional, Dict, List
import warnings

# Suppress only TensorFlow/CREPE verbosity -- not all warnings globally.
# A blanket warnings.filterwarnings('ignore') would hide real bugs elsewhere.
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*TF.*')
warnings.filterwarnings('ignore', message='.*CUDA.*', module='tensorflow')

# Robust Import for CREPE
try:
    import crepe  # type: ignore[import-not-found]
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False
    print("Warning: CREPE not found. Install 'crepe' and 'tensorflow' for best results.")

class PitchDetector:
    """Pitch detection using multiple algorithms"""
    
    def __init__(self,
                 sample_rate: int = 22050,
                 hop_length: int = 512,
                 fmin: float = 65.0,
                 fmax: float = 2093.0):
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def detect_yin(self, audio: np.ndarray, sr: Optional[int] = None) -> Dict:
        """YIN algorithm for pitch detection"""
        if sr is None: sr = self.sample_rate
        
        pitches = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=2048,
            hop_length=self.hop_length,
            trough_threshold=0.1
        )
        return {'pitches': pitches, 'method': 'yin'}
    
    def detect_pyin(self, audio: np.ndarray, sr: Optional[int] = None) -> Dict:
        """Probabilistic YIN (pYIN) pitch detection.

        Returns voiced_probs alongside the pitch array so callers can use
        frame-level confidence for weighted averaging in the multi-method ensemble.
        """
        if sr is None: sr = self.sample_rate

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=2048,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        return {
            'pitches': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,   # frame-level confidence [0, 1]
            'method': 'pyin',
        }
    
    def detect_crepe(self, audio: np.ndarray, sr: Optional[int] = None) -> Dict:
        """CREPE: Convolutional Representation for Pitch Estimation"""
        if not HAS_CREPE:
            return self.detect_pyin(audio, sr)

        if sr is None: sr = self.sample_rate
        
        try:
            # step_size in ms
            step_size = (self.hop_length / sr) * 1000
            
            time, frequency, confidence, activation = crepe.predict(
                audio,
                sr,
                viterbi=True,
                step_size=step_size,
                verbose=0
            )
            
            # Resample CREPE output to match hop_length frames if necessary
            # (CREPE output length might differ slightly from librosa)
            target_len = int(len(audio) / self.hop_length) + 1
            if len(frequency) != target_len:
                frequency = np.interp(
                    np.linspace(0, len(frequency), target_len),
                    np.arange(len(frequency)),
                    frequency
                )
            
            return {'pitches': frequency, 'confidence': confidence, 'method': 'crepe'}
        except Exception as e:
            print(f"CREPE error: {str(e)}, falling back to pYIN")
            return self.detect_pyin(audio, sr)
    
    def detect_multi_method(self, audio: np.ndarray, sr: Optional[int] = None) -> Dict:
        """
        Detect pitch using multiple methods and combine results using MEDIAN.
        """
        if sr is None: sr = self.sample_rate
        
        # Priority list
        methods = ['pyin', 'yin']
        if HAS_CREPE:
            methods.insert(0, 'crepe')
            
        results = {}
        
        for method in methods:
            try:
                func = getattr(self, f'detect_{method}')
                res = func(audio, sr)
                results[method] = res
            except Exception as e:
                continue
        
        if not results:
            # Fallback to zero
            return {'pitches': np.zeros(int(len(audio)/self.hop_length) + 1)}
        
        # Combine results
        all_pitches = []
        for m in results:
            p = results[m]['pitches']
            # Replace NaNs with 0
            p = np.nan_to_num(p, nan=0.0)
            all_pitches.append(p)
            
        # Ensure equal lengths for averaging
        min_len = min(len(p) for p in all_pitches)
        all_pitches = [p[:min_len] for p in all_pitches]
        
        # CRITICAL FIX: Use MEDIAN instead of MEAN to avoid octave errors
        # (e.g., avg of 220Hz and 440Hz is 330Hz, which is wrong. Median picks one.)
        final_pitch = np.median(np.array(all_pitches), axis=0)
        
        # Convert to MIDI for easier matching
        midi_notes = librosa.hz_to_midi(final_pitch)
        midi_notes = np.nan_to_num(midi_notes, nan=0.0)
        
        return {
            'pitches': final_pitch,
            'midi_notes': midi_notes,
            'method': 'multi_method_ensemble'
        }

    def extract_pitch_contour(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Helper to get clean pitch contour for DTW"""
        res = self.detect_multi_method(audio, sr)
        return res['pitches']