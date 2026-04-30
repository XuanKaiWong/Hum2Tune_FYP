import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AudioVisualizer:
    """Visualize audio and extracted features"""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def plot_waveform(self, 
                     audio: np.ndarray,
                     title: str = "Audio Waveform",
                     save_path: Optional[str] = None) -> Figure:
        """Plot audio waveform"""
        fig, ax = plt.subplots(figsize=(12, 3))
        
        librosa.display.waveshow(audio, sr=self.sample_rate, ax=ax)
        ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_spectrogram(self,
                        audio: np.ndarray,
                        title: str = "Spectrogram",
                        n_fft: int = 2048,
                        n_mels: int = 128,
                        save_path: Optional[str] = None) -> Figure:
        """Plot mel spectrogram"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_fft=n_fft, hop_length=self.hop_length, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Display
        img = librosa.display.specshow(
            mel_spec_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        
        ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_mfcc(self,
                 audio: np.ndarray,
                 title: str = "MFCC Features",
                 n_mfcc: int = 13,
                 save_path: Optional[str] = None) -> Figure:
        """Plot MFCC features"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            n_mfcc=n_mfcc, hop_length=self.hop_length
        )
        
        # Display
        img = librosa.display.specshow(
            mfcc,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            ax=ax
        )
        
        ax.set(title=title, xlabel="Time (s)", ylabel="MFCC Coefficients")
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_chroma(self,
                   audio: np.ndarray,
                   title: str = "Chroma Features",
                   save_path: Optional[str] = None) -> Figure:
        """Plot chroma features"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Extract chroma
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Display
        img = librosa.display.specshow(
            chroma,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='chroma',
            ax=ax
        )
        
        ax.set(title=title, xlabel="Time (s)", ylabel="Pitch Class")
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_pitch_contour(self,
                          pitch: np.ndarray,
                          times: Optional[np.ndarray] = None,
                          audio: Optional[np.ndarray] = None,
                          title: str = "Pitch Contour",
                          save_path: Optional[str] = None) -> Figure:
        """Plot pitch contour"""
        if times is None:
            times = librosa.times_like(pitch, sr=self.sample_rate, hop_length=self.hop_length)
        
        if audio is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            
            # Plot waveform
            librosa.display.waveshow(audio, sr=self.sample_rate, ax=ax1)
            ax1.set(title="Audio Waveform", xlabel="", ylabel="Amplitude")
            ax1.grid(True, alpha=0.3)
            
            # Plot pitch contour
            ax2.plot(times, pitch, label='Pitch (Hz)', color='red', linewidth=2)
            ax2.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times, pitch, label='Pitch (Hz)', color='red', linewidth=2)
            ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_feature_comparison(self,
                               features1: Dict,
                               features2: Dict,
                               feature_name: str,
                               title: str = "Feature Comparison",
                               save_path: Optional[str] = None) -> Figure:
        """Compare two feature sets"""
        if feature_name not in features1 or feature_name not in features2:
            raise ValueError(f"Feature {feature_name} not found in both feature sets")
        
        feat1 = features1[feature_name]
        feat2 = features2[feature_name]
        
        # Ensure same shape for comparison
        min_len = min(feat1.shape[1], feat2.shape[1])
        feat1 = feat1[:, :min_len]
        feat2 = feat2[:, :min_len]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot first feature set
        img1 = librosa.display.specshow(
            feat1,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            ax=ax1
        )
        ax1.set(title=f"Feature Set 1: {feature_name}", xlabel="", ylabel="Coefficients")
        fig.colorbar(img1, ax=ax1)
        
        # Plot second feature set
        img2 = librosa.display.specshow(
            feat2,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            ax=ax2
        )
        ax2.set(title=f"Feature Set 2: {feature_name}", xlabel="", ylabel="Coefficients")
        fig.colorbar(img2, ax=ax2)
        
        # Plot difference
        diff = np.abs(feat1 - feat2)
        img3 = librosa.display.specshow(
            diff,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            ax=ax3
        )
        ax3.set(title=f"Absolute Difference: {feature_name}", xlabel="Time (s)", ylabel="Coefficients")
        fig.colorbar(img3, ax=ax3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_model_predictions(self,
                              predictions: List[Dict],
                              title: str = "Model Predictions",
                              save_path: Optional[str] = None) -> Figure:
        """Visualize model prediction results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        song_names = [p['song_name'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # Create bar plot
        bars = ax.barh(range(len(song_names)), confidences, color='skyblue')
        ax.set_yticks(range(len(song_names)))
        ax.set_yticklabels(song_names)
        ax.set_xlabel('Confidence')
        ax.set_title(title)
        ax.invert_yaxis()  # Highest confidence at top
        
        # Add confidence values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{confidences[i]:.2%}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def create_analysis_dashboard(self,
                                 audio: np.ndarray,
                                 features: Dict,
                                 pitch_contour: np.ndarray,
                                 save_path: Optional[str] = None) -> Figure:
        """Create comprehensive analysis dashboard"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = fig.add_gridspec(3, 3)
        
        # 1. Waveform (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        librosa.display.waveshow(audio, sr=self.sample_rate, ax=ax1)
        ax1.set(title="Waveform", xlabel="", ylabel="Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'mel' in features:
            mel_db = librosa.power_to_db(features['mel'], ref=np.max)
            img = librosa.display.specshow(
                mel_db,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis='time',
                y_axis='mel',
                ax=ax2
            )
            ax2.set(title="Mel Spectrogram", xlabel="", ylabel="Frequency (Hz)")
            fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        
        # 3. Pitch contour (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        times = librosa.times_like(pitch_contour, sr=self.sample_rate, hop_length=self.hop_length)
        ax3.plot(times, pitch_contour, color='red', linewidth=2)
        ax3.set(title="Pitch Contour", xlabel="Time (s)", ylabel="Frequency (Hz)")
        ax3.grid(True, alpha=0.3)
        
        # 4. MFCC (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        if 'mfcc' in features:
            img = librosa.display.specshow(
                features['mfcc'],
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis='time',
                ax=ax4
            )
            ax4.set(title="MFCC Features", xlabel="", ylabel="MFCC Coefficients")
            fig.colorbar(img, ax=ax4)
        
        # 5. Chroma (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        if 'chroma_cens' in features:
            img = librosa.display.specshow(
                features['chroma_cens'],
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis='time',
                y_axis='chroma',
                ax=ax5
            )
            ax5.set(title="Chroma Features", xlabel="Time (s)", ylabel="Pitch Class")
            fig.colorbar(img, ax=ax5)
        
        # 6. Spectral features (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid'][0]
            times_centroid = librosa.times_like(centroid, sr=self.sample_rate, hop_length=self.hop_length)
            ax6.plot(times_centroid, centroid, label='Spectral Centroid', color='green')
            ax6.set(title="Spectral Features", xlabel="Time (s)", ylabel="Frequency (Hz)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Rhythm features (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        if 'onset_strength' in features:
            onset = features['onset_strength']
            times_onset = librosa.times_like(onset, sr=self.sample_rate, hop_length=self.hop_length)
            ax7.plot(times_onset, onset, label='Onset Strength', color='purple')
            ax7.set(title="Rhythm Analysis", xlabel="Time (s)", ylabel="Strength")
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle("Audio Analysis Dashboard", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def save_all_visualizations(self,
                               audio: np.ndarray,
                               features: Dict,
                               pitch_contour: np.ndarray,
                               output_dir: str = "results/visualizations"):
        """Save all visualizations to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual plots
        self.plot_waveform(audio, save_path=str(output_path / "waveform.png"))
        self.plot_spectrogram(audio, save_path=str(output_path / "spectrogram.png"))
        self.plot_mfcc(audio, save_path=str(output_path / "mfcc.png"))
        self.plot_chroma(audio, save_path=str(output_path / "chroma.png"))
        self.plot_pitch_contour(pitch_contour, audio=audio, 
                               save_path=str(output_path / "pitch_contour.png"))
        
        # Save dashboard
        self.create_analysis_dashboard(audio, features, pitch_contour,
                                      save_path=str(output_path / "analysis_dashboard.png"))
        
        print(f"[OK] Visualizations saved to: {output_path}")