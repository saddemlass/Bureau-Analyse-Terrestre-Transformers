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


class ConvBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float, norm: str) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=kernel, padding=padding, dilation=dilation)
        if norm == "batch":
            self.norm: torch.nn.Module = torch.nn.BatchNorm1d(channels)
        elif norm == "layer":
            self.norm = torch.nn.LayerNorm(channels)
        elif norm == "group":
            self.norm = torch.nn.GroupNorm(8, channels)
        elif norm == "sample":
            self.norm = torch.nn.Identity()
        else:
            self.norm = torch.nn.Identity()
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm_kind = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        if self.norm_kind == "sample":
            x = (x - x.mean(dim=(1, 2), keepdim=True)) / x.var(dim=(1, 2), keepdim=True, unbiased=False).add(1e-5).sqrt()
        elif self.norm_kind == "layer":
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)
        x = self.dropout(self.activation(x))
        return x + residual


class ConvTextClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        emb_dim: int = 48,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        kernel: int = 3,
        dropout: float = 0.20,
        norm: str = "batch",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.proj = torch.nn.Linear(emb_dim, channels)
        self.blocks = torch.nn.ModuleList(
            [ConvBlock(channels, kernel, dilation, dropout, norm) for dilation in dilations]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(channels, n_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(-1)
        x = self.proj(self.embedding(input_ids)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2) * mask
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = x.sum(dim=1) / lengths
        return self.classifier(pooled)


def receptive_field_table(kernels: list[int], dilations: list[int], strides: list[int]) -> list[dict[str, int]]:
    receptive_field = 1
    jump = 1
    rows = []
    for idx, (kernel, dilation, stride) in enumerate(zip(kernels, dilations, strides), start=1):
        added = (kernel - 1) * dilation * jump
        receptive_field += added
        rows.append(
            {
                "couche": idx,
                "kernel": kernel,
                "dilation": dilation,
                "stride": stride,
                "champ_ajoute": added,
                "champ_cumule": receptive_field,
            }
        )
        jump *= stride
    return rows
