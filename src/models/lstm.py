"""
Sentiment classification models:
  - BiLSTMWithAttention  (main model — best accuracy)
  - BaselineLSTM         (ablation baseline)
  - StackedBiGRU         (GRU variant for comparison)

All models accept an optional `pretrained_embeddings` tensor (from GloVe)
to replace random initialisation.  Pass the tensor returned by
src.utils.embeddings.load_glove() to enable this.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _make_embedding(
    vocab_size: int,
    embedding_dim: int,
    pad_idx: int,
    pretrained: Optional[torch.Tensor],
) -> nn.Embedding:
    """
    Return an Embedding layer, optionally pre-loaded with GloVe/FastText weights.
    Pre-trained weights are *fine-tuned* during training (freeze=False).
    """
    if pretrained is not None:
        emb = nn.Embedding.from_pretrained(
            pretrained, freeze=False, padding_idx=pad_idx
        )
    else:
        emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(emb.weight)
        with torch.no_grad():
            emb.weight[pad_idx].fill_(0.0)
    return emb


class SelfAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention pooling over LSTM hidden states.
    Produces a fixed-size context vector and interpretable attention weights.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, lstm_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        energy  = torch.tanh(self.W(lstm_output))      # (B, T, H)
        scores  = self.v(energy).squeeze(-1)           # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        context = (weights.unsqueeze(-1) * lstm_output).sum(dim=1)  # (B, H)
        return context, weights


class BiLSTMWithAttention(nn.Module):
    """
    Production model: Bidirectional LSTM + Self-Attention + Layer Norm.

    Architecture:
        Embedding (random or GloVe/FastText)
        → BiLSTM (n_layers)
        → Additive Self-Attention  → context vector + attention weights
        → LayerNorm
        → FC(hidden*2, hidden) → GELU → Dropout
        → FC(hidden, num_classes)

    Parameters
    ----------
    pretrained_embeddings : optional FloatTensor (vocab_size+1, embedding_dim)
        from src.utils.embeddings.load_glove().
        When supplied, embedding_dim is inferred from the tensor shape.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        n_layers: int = 3,
        dropout: float = 0.4,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            embedding_dim = pretrained_embeddings.shape[1]

        self.embedding  = _make_embedding(vocab_size, embedding_dim, pad_idx, pretrained_embeddings)
        self.lstm       = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention  = SelfAttention(hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, num_classes)

        # Orthogonal init for LSTM weights (helps with gradient flow)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embedded           = self.dropout(self.embedding(x))   # (B, T, E)
        lstm_out, _        = self.lstm(embedded)               # (B, T, H*2)
        context, attn      = self.attention(lstm_out)
        context            = self.layer_norm(context)
        out                = F.gelu(self.fc1(self.dropout(context)))
        return self.fc2(out), attn


class BaselineLSTM(nn.Module):
    """Unidirectional stacked LSTM — ablation baseline (no attention)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        n_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            embedding_dim = pretrained_embeddings.shape[1]

        self.embedding = _make_embedding(vocab_size, embedding_dim, pad_idx, pretrained_embeddings)
        self.lstm      = nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        embedded   = self.dropout(self.embedding(x))
        lstm_out, _= self.lstm(embedded)
        return self.fc(self.dropout(lstm_out[:, -1, :])), None


class StackedBiGRU(nn.Module):
    """Stacked Bidirectional GRU with max-over-time pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        n_layers: int = 3,
        dropout: float = 0.4,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            embedding_dim = pretrained_embeddings.shape[1]

        self.embedding = _make_embedding(vocab_size, embedding_dim, pad_idx, pretrained_embeddings)
        self.gru       = nn.GRU(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        embedded      = self.dropout(self.embedding(x))
        gru_out, _    = self.gru(embedded)
        pooled, _     = gru_out.max(dim=1)
        return self.fc(self.dropout(pooled)), None
