from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Return the largest valid group count for GroupNorm."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """Conv1d -> GroupNorm -> SiLU -> Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_safe_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Small residual temporal convolution block."""

    def __init__(self, channels: int, *, kernel_size: int = 5, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_safe_group_count(channels), channels),
            nn.Dropout(float(dropout)),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.conv2(self.conv1(x)))


class AttentiveStatsPooling(nn.Module):
    """Attention-weighted mean and standard deviation pooling.

    Input:
        x: (batch, time, channels)

    Output:
        pooled: (batch, channels * 2)
    """

    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=1)
        mean = torch.sum(weights * x, dim=1)
        second_moment = torch.sum(weights * x.pow(2), dim=1)
        var = torch.clamp(second_moment - mean.pow(2), min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1)


class ProjectionHead(nn.Module):
    """MLP projection head with final L2 normalisation."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MelodyEncoder(nn.Module):
    """Domain-specific melody encoder for either query or reference audio.

    Expected input shape is (batch, channels, frames), where channels are usually
    39 for MFCC + delta + delta-delta features.
    """

    def __init__(
        self,
        input_channels: int = 39,
        embedding_dim: int = 128,
        hidden_channels: Iterable[int] = (48, 96, 160),
        recurrent_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_channels = tuple(hidden_channels)
        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one value")

        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for out_channels in hidden_channels:
            layers.append(
                ConvNormAct(
                    in_channels,
                    int(out_channels),
                    kernel_size=5,
                    stride=2,
                    dropout=dropout,
                )
            )
            layers.append(ResidualConvBlock(int(out_channels), kernel_size=5, dropout=dropout))
            in_channels = int(out_channels)

        self.cnn = nn.Sequential(*layers)
        self.recurrent = nn.GRU(
            input_size=in_channels,
            hidden_size=int(recurrent_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        recurrent_out = int(recurrent_dim) * 2
        self.pool = AttentiveStatsPooling(recurrent_out, attention_dim=max(64, recurrent_out // 2))
        self.projection = ProjectionHead(
            input_dim=recurrent_out * 2,
            embedding_dim=int(embedding_dim),
            hidden_dim=max(128, recurrent_out * 2),
            dropout=dropout,
        )

        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {x.size(1)}. "
                "Check audio_config.yaml and FeatureExtractor output."
            )

        h = self.cnn(x)
        h = h.transpose(1, 2)  # (B, T, C)
        h, _ = self.recurrent(h)
        pooled = self.pool(h)
        return self.projection(pooled)


class DualEncoder(nn.Module):
    """Dual encoder for humming-to-song retrieval."""

    def __init__(
        self,
        input_channels: int = 39,
        embedding_dim: int = 128,
        hidden_channels: Iterable[int] = (48, 96, 160),
        recurrent_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)

        self.query_encoder = MelodyEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            recurrent_dim=recurrent_dim,
            dropout=dropout,
        )
        self.reference_encoder = MelodyEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            recurrent_dim=recurrent_dim,
            dropout=dropout,
        )

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)

    def encode_reference(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference_encoder(x)

    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_query(query), self.encode_reference(reference)


class MultiPositiveContrastiveLoss(nn.Module):
    """Supervised contrastive retrieval loss for paired query/reference batches.

    If labels are provided, every query and reference with the same label is
    considered a positive. This is important for Hum2Tune because a batch can
    contain several humming windows from the same song. Treating them as
    negatives would corrupt the training signal.

    If labels are omitted, the loss falls back to diagonal-only NT-Xent.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    @staticmethod
    def _directional_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_mask = positive_mask.to(dtype=log_prob.dtype)
        positive_count = positive_mask.sum(dim=1).clamp_min(1.0)
        loss = -(positive_mask * log_prob).sum(dim=1) / positive_count
        return loss.mean()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_embeddings.shape != reference_embeddings.shape:
            raise ValueError(
                "query_embeddings and reference_embeddings must have the same shape, "
                f"got {tuple(query_embeddings.shape)} and {tuple(reference_embeddings.shape)}"
            )

        q = F.normalize(query_embeddings, dim=-1)
        r = F.normalize(reference_embeddings, dim=-1)
        logits = (q @ r.T) / self.temperature

        batch_size = logits.size(0)
        if labels is None:
            positive_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        else:
            labels = labels.to(logits.device).view(-1)
            if labels.numel() != batch_size:
                raise ValueError(f"labels length must equal batch size {batch_size}, got {labels.numel()}")
            positive_mask = labels[:, None].eq(labels[None, :])

        loss_q_to_r = self._directional_loss(logits, positive_mask)
        loss_r_to_q = self._directional_loss(logits.T, positive_mask.T)
        return 0.5 * (loss_q_to_r + loss_r_to_q)


# Backwards-compatible name used by the previous training script.
NTXentLoss = MultiPositiveContrastiveLoss


@torch.no_grad()
def cosine_retrieval(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k cosine scores and indices.

    Both tensors are normalised inside the function for safety.
    """
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    q = F.normalize(query_embeddings, dim=-1)
    r = F.normalize(reference_embeddings, dim=-1)
    scores = q @ r.T

    k = min(int(top_k), reference_embeddings.size(0))
    return scores.topk(k, dim=-1)


@dataclass(frozen=True)
class RetrievalMetrics:
    top1: float
    top3: float
    top5: float
    mrr: float
    count: int


def compute_retrieval_metrics(ranks: Iterable[int | None]) -> RetrievalMetrics:
    """Compute standard single-relevant-item retrieval metrics from ranks."""
    ranks = list(ranks)
    if not ranks:
        return RetrievalMetrics(top1=0.0, top3=0.0, top5=0.0, mrr=0.0, count=0)

    def hit_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)
    return RetrievalMetrics(
        top1=hit_at(1),
        top3=hit_at(3),
        top5=hit_at(5),
        mrr=float(mrr),
        count=len(ranks),
    )


if __name__ == "__main__":
    batch, channels, frames = 4, 39, 431
    model = DualEncoder(input_channels=channels, embedding_dim=128)
    loss_fn = MultiPositiveContrastiveLoss(temperature=0.1)

    q = torch.randn(batch, channels, frames)
    r = torch.randn(batch, channels, frames)
    labels = torch.tensor([0, 1, 1, 2])

    q_emb, r_emb = model(q, r)
    loss = loss_fn(q_emb, r_emb, labels)

    print("query embedding:", tuple(q_emb.shape))
    print("reference embedding:", tuple(r_emb.shape))
    print("loss:", float(loss))
