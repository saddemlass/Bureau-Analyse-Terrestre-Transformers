from __future__ import annotations

import copy
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, BertTokenizer, DataCollatorWithPadding

try:
    import psutil
except ImportError:  # pragma: no cover - documented fallback.
    psutil = None


MODEL_NAME = "prajjwal1/bert-tiny"


@dataclass
class RegimeResult:
    regime: str
    model_name: str
    accuracy: float
    macro_f1: float
    total_parameters: int
    trainable_parameters: int
    trainable_percent: float
    training_step_ms: float
    training_total_s: float
    peak_memory_mib: float
    saved_artifact_mib: float
    details: dict[str, Any]


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        item["labels"] = self.labels[idx]
        return item


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name: str, n_classes: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenizer_audit(texts: list[str], tokenizer: Any) -> dict[str, float | int]:
    lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]
    return {
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(max(lengths)),
    }


def choose_max_length(stats: dict[str, float | int]) -> int:
    p99 = float(stats["p99"])
    if p99 <= 64:
        return 64
    if p99 <= 96:
        return 96
    if p99 <= 128:
        return 128
    return min(256, int(((p99 + 31) // 32) * 32))


def make_loaders(
    splits: dict[str, tuple[list[str], list[int]]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loaders = {}
    for split, (texts, labels) in splits.items():
        dataset = TextLabelDataset(texts, labels, tokenizer, max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
        )
    return loaders


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def encoder_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)


def freeze_all_encoder(model: BertClassifier) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def freeze_for_partial(model: BertClassifier) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    layers = list(model.encoder.encoder.layer)
    for param in layers[-1].parameters():
        param.requires_grad = True
    return [f"encoder.encoder.layer.{len(layers) - 1}", "classifier"]


def freeze_base_for_lora(model: BertClassifier) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def make_optimizer(model: BertClassifier, regime: str) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    if regime == "partial":
        head_params = list(model.classifier.parameters())
        encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
        lrs = {"last_layer_lr": 2e-5, "head_lr": 3e-4}
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": lrs["last_layer_lr"]},
                {"params": head_params, "lr": lrs["head_lr"]},
            ],
            weight_decay=0.01,
        )
        return optimizer, lrs
    lr = 5e-4 if regime in {"frozen", "lora"} else 3e-4
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01), {"lr": lr}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = count = 0
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            y = batch.pop("labels")
            logits = model(**batch)
            total_loss += float(loss_fn(logits, y).item()) * len(y)
            count += len(y)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y.cpu())
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    return total_loss / count, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def current_memory_mib(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    if psutil is None:
        return float("nan")
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))


def train(
    model: BertClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
    regime: str,
    max_epochs: int = 3,
    patience: int = 1,
) -> tuple[float, float, float, dict[str, torch.Tensor], dict[str, float]]:
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer, lrs = make_optimizer(model, regime)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step_times: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    peak = current_memory_mib(device)

    for _ in range(max_epochs):
        model.train()
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = move_batch(batch, device)
            y = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            loss = loss_fn(model(**batch), y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - step_start) * 1000
            if batch_idx > 5:
                step_times.append(elapsed_ms)
            peak = max(peak, current_memory_mib(device))
        val_loss, _, _ = evaluate(model, loaders["val"], device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    training_total = time.perf_counter() - start
    model.load_state_dict(best_state)
    step_ms = float(statistics.median(step_times)) if step_times else float("nan")
    return training_total, step_ms, peak, best_state, lrs


def directory_size_mib(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024**2)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024**2)


def save_trainable_artifact(model: BertClassifier, path: Path, regime: str, details: dict[str, Any]) -> float:
    path.mkdir(parents=True, exist_ok=True)
    trainable_names = set(details["trainable_names"])
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }
    torch.save(trainable_state, path / "trainable_state.pt")
    (path / "config.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return directory_size_mib(path)


def apply_lora(model: BertClassifier) -> tuple[BertClassifier, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query", "value"],
    )
    model.encoder = get_peft_model(model.encoder, config)
    freeze_base_for_lora(model)
    for name, param in model.encoder.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return model, {"r": 4, "alpha": 8, "dropout": 0.05, "target_modules": "query,value"}


def run_regime(
    regime: str,
    loaders: dict[str, DataLoader],
    device: torch.device,
    n_classes: int,
    artifact_root: Path,
) -> RegimeResult:
    set_seed(7)
    model = BertClassifier(MODEL_NAME, n_classes)
    details: dict[str, Any] = {}
    if regime == "frozen":
        freeze_all_encoder(model)
        assert encoder_trainable_parameters(model) == 0
    elif regime == "partial":
        unfrozen = freeze_for_partial(model)
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        assert all(name.startswith("classifier.") or ".encoder.layer.1." in name for name in trainable_names)
        details["unfrozen"] = unfrozen
    elif regime == "lora":
        model, lora_details = apply_lora(model)
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        assert all("lora_" in name or name.startswith("classifier.") for name in trainable_names)
        details.update(lora_details)
    else:
        raise ValueError(regime)

    total, trainable = count_parameters(model)
    details["trainable_names"] = [name for name, p in model.named_parameters() if p.requires_grad]
    training_total, step_ms, peak, _, lrs = train(model, loaders, device, regime)
    _, accuracy, macro_f1 = evaluate(model, loaders["test"], device)
    details["learning_rates"] = lrs
    saved_mib = save_trainable_artifact(model, artifact_root / regime, regime, {"regime": regime, **details})
    return RegimeResult(
        regime=regime,
        model_name=MODEL_NAME,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        total_parameters=int(total),
        trainable_parameters=int(trainable),
        trainable_percent=float(trainable / total * 100),
        training_step_ms=float(step_ms),
        training_total_s=float(training_total),
        peak_memory_mib=float(peak),
        saved_artifact_mib=float(saved_mib),
        details=details,
    )


def model_metadata(model_name: str = MODEL_NAME) -> dict[str, Any]:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    total, _ = count_parameters(model)
    return {
        "model_name": model_name,
        "total_parameters": int(total),
        "architecture": config.model_type,
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer": tokenizer,
    }
