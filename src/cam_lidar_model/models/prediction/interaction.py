"""Interaction modeling between actors."""

from __future__ import annotations

import torch
import torch.nn as nn


class InteractionModule(nn.Module):
    """Lightweight transformer-style interaction layer."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, actor_embeddings: torch.Tensor) -> torch.Tensor:
        """Model interactions.

        Args:
            actor_embeddings: ``[B, N, D]`` tensor from :class:`ActorEncoder`.

        Returns:
            ``torch.Tensor`` of the same shape.
        """

        return self.encoder(actor_embeddings)
