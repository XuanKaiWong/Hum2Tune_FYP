import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================
# Utilities
# ============================================================

def _safe_group_count(num_channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def make_norm_1d(num_channels: int, norm_type: str = "group") -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    if norm_type == "group":
        return nn.GroupNorm(_safe_group_count(num_channels), num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock1D(nn.Module):
    """
    Small-data friendly 1D convolution block.
    Kept lighter than the large accuracy-focused version.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        norm_type: str = "group",
    ):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            make_norm_1d(out_channels, norm_type=norm_type),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        scores = self.attn(x)                # (B, T, 1)
        weights = F.softmax(scores, dim=1)   # (B, T, 1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


# ============================================================
# Main small-data CNN-LSTM model
# ============================================================

class CNNLSTMModel(nn.Module):
    """CNN-LSTM baseline for humming classification.

    Architecture overview:
    - 3 x ConvBlock1D (32 -> 64 -> 128 channels) extract local spectral patterns.
    - Bidirectional LSTM models temporal dependencies between frames.
    - TemporalAttention pools the LSTM output into a fixed-size embedding
      by weighting frames according to their melodic salience.
    - Linear classifier maps the embedding to class logits.

    input_channels defaults to 39 to match the FeatureExtractor output
    (13 MFCC + 13 delta + 13 delta-delta).
    """

    def __init__(
        self,
        input_channels: int = 39,   # n_mfcc * 3 -- updated from 13
        num_classes: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.4,
        bidirectional: bool = True,
        use_attention: bool = True,
        attn_hidden_dim: int = 32,  # exposed for ablation studies
        norm_type: str = "group",
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.norm_type = norm_type
        self.attn_hidden_dim = attn_hidden_dim

        # -- CNN backbone: local spectral pattern extraction --------------
        # Three stacked 1D conv blocks progressively downsample the time axis
        # (T -> T/2 -> T/4 -> T/8) while expanding the channel dimension.
        self.conv_block1 = ConvBlock1D(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            pool_size=2,
            dropout=dropout * 0.4,
            norm_type=norm_type,
        )
        self.conv_block2 = ConvBlock1D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            pool_size=2,
            dropout=dropout * 0.7,
            norm_type=norm_type,
        )
        self.conv_block3 = ConvBlock1D(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            pool_size=2,
            dropout=dropout,
            norm_type=norm_type,
        )

        # -- Bidirectional LSTM: temporal context modelling ---------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.lstm_out_size = hidden_size * (2 if bidirectional else 1)

        # -- Temporal attention pooling ------------------------------------
        # attn_hidden_dim is exposed as a hyperparameter so ablation studies
        # can test the effect of attention capacity on performance.
        if use_attention:
            self.attention = TemporalAttention(
                input_dim=self.lstm_out_size,
                hidden_dim=attn_hidden_dim,
            )

        self.embedding_dim = self.lstm_out_size

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            else:
                if "weight" in name:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.05, 0.05)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

        # Better LSTM forget gate init
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        # x: (B, C, T)

        cnn_features1 = self.conv_block1(x)               # (B, 32, T/2)
        cnn_features2 = self.conv_block2(cnn_features1)   # (B, 64, T/4)
        cnn_features3 = self.conv_block3(cnn_features2)   # (B, 128, T/8)

        lstm_input = cnn_features3.permute(0, 2, 1)       # (B, T/8, 128)
        lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input)

        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_output)
        else:
            attention_weights = None
            if self.bidirectional:
                forward_final = hidden_state[-2]
                backward_final = hidden_state[-1]
                context_vector = torch.cat([forward_final, backward_final], dim=1)
            else:
                context_vector = hidden_state[-1]

        logits = self.classifier(context_vector)

        if not return_features:
            return logits

        features = {
            "cnn_features1": cnn_features1,
            "cnn_features2": cnn_features2,
            "cnn_features3": cnn_features3,
            "lstm_output": lstm_output,
            "hidden_state": hidden_state,
            "cell_state": cell_state,
            "context_vector": context_vector,
        }
        if attention_weights is not None:
            features["attention_weights"] = attention_weights

        return logits, features

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, features = self.forward(x, return_features=True)
        return features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, return_features=False)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, return_features=False)
        return F.softmax(logits, dim=-1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x, return_features=True)
        return features["context_vector"]


# ============================================================
# Pitch CNN
# ============================================================

class PitchCNN(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        num_classes: int = 10,
        dropout: float = 0.4,
        norm_type: str = "group",
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2, bias=False),
            make_norm_1d(16, norm_type=norm_type),
            nn.SiLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            make_norm_1d(32, norm_type=norm_type),
            nn.SiLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            make_norm_1d(64, norm_type=norm_type),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.05, 0.05)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return x.squeeze(-1)


# ============================================================
# Optional fusion model
# ============================================================

class FusionModel(nn.Module):
    """
    Optional fusion model.
    Keep this only as an experiment, not your main path.
    """

    def __init__(
        self,
        num_classes: int = 10,
        acoustic_channels: int = 39,   # must match FeatureExtractor.input_channels (n_mfcc * 3)
        hidden_size: int = 64,
        dropout: float = 0.4,
        norm_type: str = "group",
    ):
        super().__init__()

        self.acoustic_branch = CNNLSTMModel(
            input_channels=acoustic_channels,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=True,
            use_attention=True,
            norm_type=norm_type,
        )

        self.pitch_branch = PitchCNN(
            input_size=1,
            num_classes=num_classes,
            dropout=dropout,
            norm_type=norm_type,
        )

        acoustic_dim = self.acoustic_branch.embedding_dim
        pitch_dim = 64

        self.fusion_head = nn.Sequential(
            nn.Linear(acoustic_dim + pitch_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(
        self,
        x_acoustic: torch.Tensor,
        x_pitch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, acoustic_features = self.acoustic_branch(x_acoustic, return_features=True)
        acoustic_emb = acoustic_features["context_vector"]

        pitch_emb = self.pitch_branch.get_embedding(x_pitch)

        fused = torch.cat([acoustic_emb, pitch_emb], dim=1)
        logits = self.fusion_head(fused)

        features = {
            "acoustic_embedding": acoustic_emb,
            "pitch_embedding": pitch_emb,
            "fusion_embedding": fused,
        }
        return logits, features

    def predict(self, x_acoustic: torch.Tensor, x_pitch: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x_acoustic, x_pitch)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x_acoustic: torch.Tensor, x_pitch: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x_acoustic, x_pitch)
        return F.softmax(logits, dim=-1)

# NOTE: LabelSmoothingCrossEntropy was removed because PyTorch >= 1.10 provides
# nn.CrossEntropyLoss(label_smoothing=0.1) natively. Use that instead.
# Keeping a custom implementation alongside the native one created two
# conflicting sources of truth and dead code.


if __name__ == "__main__":
    B = 4
    T = 431
    num_classes = 10

    # Use 39 channels to match FeatureExtractor output (13 MFCC x 3)
    x_acoustic = torch.randn(B, 39, T)
    x_pitch = torch.randn(B, 1, T)

    print("Testing CNNLSTMModel (39-channel input)...")
    m1 = CNNLSTMModel(input_channels=39, num_classes=num_classes)
    y1 = m1(x_acoustic)
    print("CNNLSTM logits:", y1.shape)

    print("\nTesting with return_features=True ...")
    y1_logits, y1_feats = m1(x_acoustic, return_features=True)
    print("CNNLSTM logits:", y1_logits.shape)
    print("Context vector:", y1_feats["context_vector"].shape)

    print("\nTesting PitchCNN...")
    m2 = PitchCNN(input_size=1, num_classes=num_classes)
    y2 = m2(x_pitch)
    print("PitchCNN logits:", y2.shape)

    print("\nTesting FusionModel...")
    m3 = FusionModel(num_classes=num_classes, acoustic_channels=39)
    y3, f3 = m3(x_acoustic, x_pitch)
    print("Fusion logits:", y3.shape)
    print("Fusion embedding:", f3["fusion_embedding"].shape)
