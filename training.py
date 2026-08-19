from __future__ import annotations

import copy
import random
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    elapsed_times: list[float]
    best_state: dict[str, torch.Tensor]
    best_val_loss: float
    train_time: float


def set_seeds(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts: list[str], max_vocab: int = 12000, min_freq: int = 2) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int = 80) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = [vocab.get(token, 1) for token in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 25,
    patience: int = 4,
) -> TrainResult:
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    elapsed_times: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale = 0
    start = time.perf_counter()

    for _ in range(max_epochs):
        model.train()
        total = count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(yb)
            count += len(yb)
        train_losses.append(total / count)

        val_loss, _ = evaluate_loss(model, val_loader)
        val_losses.append(val_loss)
        elapsed_times.append(time.perf_counter() - start)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return TrainResult(train_losses, val_losses, elapsed_times, best_state, best_val_loss, train_time)


def evaluate_loss(model: torch.nn.Module, loader: DataLoader) -> tuple[float, torch.Tensor]:
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    total = count = 0
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total += float(loss_fn(logits, yb).item()) * len(yb)
            count += len(yb)
            preds.append(logits.argmax(dim=1))
    return total / count, torch.cat(preds)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, y_true: np.ndarray) -> dict[str, float]:
    _, preds = evaluate_loss(model, loader)
    y_pred = preds.numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
