"""Audio Transformer model for Hum2Tune.

Implements a Transformer-encoder-based melody classifier to compare against
the CNN-LSTM baseline, directly addressing RQ3:
    "How does the choice of model architecture influence performance
     in melody-based music recognition?"

Architecture overview:
    1. Linear input projection: maps each frame's feature vector to d_model dims.
    2. Sinusoidal positional encoding: injects temporal order information.
       Unlike learned embeddings, sinusoidal encoding generalises to sequences
       longer than those seen during training.
    3. Transformer encoder (N x TransformerEncoderLayer): models long-range
       dependencies between melody frames via multi-head self-attention.
       This allows the model to focus on salient hooks regardless of their
       position in the sequence -- a key advantage over LSTM's sequential bias.
    4. Global average pooling: aggregates the variable-length sequence into a
       fixed-size embedding without introducing learnable parameters.
    5. Classification head: two-layer MLP maps the embedding to class logits.

References:
    Vaswani et al. (2017) "Attention Is All You Need". NeurIPS.
    Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers".
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position information to token embeddings without any learned
    parameters. The encoding uses alternating sine and cosine functions
    at geometrically spaced frequencies so that relative positions can be
    inferred by the attention mechanism.

    Args:
        d_model: Embedding dimension (must match input projection output).
        dropout: Dropout applied after adding positional encoding.
        max_len: Maximum sequence length supported.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the (max_len, d_model) encoding table once and register as buffer
        # so it moves to the correct device with the model automatically.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims: cosine
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -- broadcast over batch

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    """Transformer-encoder melody classifier for humming recognition.

    Designed as a direct architectural comparison to the CNN-LSTM baseline.
    Both models receive the same 39-channel MFCC+delta feature input and
    produce logits over the same class set, enabling a fair ablation study.

    Key differences from CNN-LSTM:
    - Self-attention has global receptive field (vs LSTM's sequential context).
    - No inductive bias for local patterns (the CNN-LSTM's conv layers provide).
    - Scales better with more data but may underfit on small datasets.

    Args:
        input_channels: Feature channels per frame (39 for MFCC+Delta+DeltaDelta).
        num_classes: Number of song classes.
        d_model: Transformer hidden dimension.
        nhead: Number of self-attention heads. Must divide d_model evenly.
        num_layers: Number of TransformerEncoderLayer stacks.
        dim_feedforward: Inner dimension of the FFN sublayer.
        dropout: Dropout applied throughout.
        max_seq_len: Maximum supported time-frame count.
    """

    def __init__(
        self,
        input_channels: int = 39,
        num_classes: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.embedding_dim = d_model

        # -- Input projection -----------------------------------------------
        # Maps each frame's feature vector from input_channels -> d_model.
        # A LayerNorm after projection stabilises early training.
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # -- Positional encoding --------------------------------------------
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)

        # -- Transformer encoder --------------------------------------------
        # batch_first=True so input shape is (B, T, d_model), consistent with
        # the rest of the codebase which uses (batch, time, features) order.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",   # GELU outperforms ReLU on audio tasks
            batch_first=True,
            norm_first=True,     # Pre-LN: more stable training (Xiong et al., 2020)
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        # -- Classification head --------------------------------------------
        # Two-layer MLP with dropout for regularisation.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T) -- same as CNN-LSTM API.
               C = input_channels (39), T = number of frames.
            return_features: If True, return (logits, feature_dict).
                             If False, return logits only (training default).

        Returns:
            logits: (B, num_classes)
            features (optional): dict with 'context_vector' and 'encoder_output'.
        """
        # Reorder from (B, C, T) -> (B, T, C) for the Transformer.
        x = x.permute(0, 2, 1)          # (B, T, input_channels)

        # Project each frame to d_model dimensions.
        x = self.input_proj(x)           # (B, T, d_model)

        # Add positional encoding.
        x = self.pos_enc(x)              # (B, T, d_model)

        # Run through Transformer encoder layers.
        encoder_output = self.encoder(x) # (B, T, d_model)

        # Global average pooling over the time axis to get a fixed-size embedding.
        # This is permutation-equivariant but works well in practice because the
        # positional encoding already provides ordering information.
        context_vector = encoder_output.mean(dim=1)  # (B, d_model)

        logits = self.classifier(context_vector)     # (B, num_classes)

        if not return_features:
            return logits

        features: Dict[str, torch.Tensor] = {
            "encoder_output": encoder_output,
            "context_vector": context_vector,
        }
        return logits, features

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled embedding vector (used in retrieval mode)."""
        _, features = self.forward(x, return_features=True)
        return features["context_vector"]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, return_features=False)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, return_features=False)
        return F.softmax(logits, dim=-1)


# -----------------------------------------------------------------------------
#  Quick smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, T = 4, 39, 431
    num_classes = 10
    x = torch.randn(B, C, T)

    print("Testing AudioTransformer (default config)...")
    model = AudioTransformer(input_channels=C, num_classes=num_classes)
    logits = model(x)
    print(f"  logits shape:         {logits.shape}")   # (4, 10)

    logits2, feats = model(x, return_features=True)
    print(f"  context_vector shape: {feats['context_vector'].shape}")  # (4, 128)
    print(f"  encoder_output shape: {feats['encoder_output'].shape}")  # (4, 431, 128)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print("\nAll checks passed.")
