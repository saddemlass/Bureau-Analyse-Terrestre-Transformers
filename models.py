from __future__ import annotations

import torch


class TextClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        emb_dim: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.25,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(input_ids)
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1)
        return self.net(summed / lengths)
