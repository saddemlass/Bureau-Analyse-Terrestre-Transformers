from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel

from pretrained import MODEL_NAME, TextLabelDataset, apply_lora, directory_size_mib

PHASE16_MAX_MACRO_F1_DROP = 0.02
PHASE16_MAX_LENGTH = 64
PHASE16_BENCHMARK_N = 512
PHASE16_WARMUP = 16
PHASE16_REPEATS = 2


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class Phase16BertClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
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


def _collate_batch(tokenizer: Any, max_length: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        batch = tokenizer.pad(rows, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def load_phase14_model(n_classes: int, checkpoint_dir: Path) -> torch.nn.Module:
    config_path = checkpoint_dir / "config.json"
    state_path = checkpoint_dir / "trainable_state.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Artefact Phase 14 manquant: {checkpoint_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    regime = config.get("regime")
    model = Phase16BertClassifier(n_classes)
    if regime == "lora":
        model, _ = apply_lora(model)
    elif regime != "partial":
        raise RuntimeError(f"Regime Phase 14 non pris en charge pour Phase 16: {regime}")
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if name]
    required = set(config["trainable_names"])
    missing_required = sorted(required.intersection(missing))
    if unexpected or missing_required:
        raise RuntimeError(f"Chargement Phase 14 incoherent: missing={missing_required}, unexpected={unexpected}")
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
    model.eval()
    return model.cpu()


def make_phase16_loader(
    splits: dict[str, tuple[list[str], list[int]]],
    tokenizer: Any,
    max_length: int,
    limit: int,
) -> tuple[DataLoader, np.ndarray]:
    texts, labels = splits["test"]
    texts = texts[:limit]
    labels = labels[:limit]
    dataset = TextLabelDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_batch(tokenizer, max_length))
    return loader, np.asarray(labels, dtype=np.int64)


def save_state_artifact(model: torch.nn.Module, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": MODEL_NAME, "state_dict": model.state_dict()}, path)
    return directory_size_mib(path)


def predict_and_time(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    latencies: list[float] = []
    preds: list[int] = []
    batches = list(loader)

    def forward(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        except RuntimeError:
            return model(batch["input_ids"], batch["attention_mask"])

    with torch.inference_mode():
        for batch in batches[:PHASE16_WARMUP]:
            _ = batch["labels"]
            forward(batch)
        start_total = time.perf_counter()
        for _ in range(PHASE16_REPEATS):
            for batch in batches:
                _ = batch["labels"]
                start = time.perf_counter()
                logits = forward(batch)
                latencies.append((time.perf_counter() - start) * 1000)
                if len(preds) < len(batches):
                    preds.append(int(logits.argmax(dim=1).item()))
        total = time.perf_counter() - start_total
    return np.asarray(preds, dtype=np.int64), {
        "mean_latency_ms": float(statistics.mean(latencies)),
        "median_latency_ms": float(statistics.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "responses_per_second": float(len(latencies) / total),
    }


def evaluate_variant(
    variant: str,
    model: torch.nn.Module,
    loader: DataLoader,
    y_true: np.ndarray,
    disk_mib: float,
    baseline: dict[str, float] | None = None,
) -> dict[str, object]:
    pred, timing = predict_and_time(model, loader)
    macro = float(f1_score(y_true, pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, pred))
    if baseline is None:
        delta = compression = speedup = 0.0
        within = True
        decision = "REFERENCE"
    else:
        delta = macro - float(baseline["macro_f1"])
        compression = float(baseline["disk_mib"]) / disk_mib if disk_mib > 0 else float("nan")
        speedup = timing["responses_per_second"] / float(baseline["responses_per_second"])
        within = macro >= float(baseline["macro_f1"]) - PHASE16_MAX_MACRO_F1_DROP
        decision = "ACCEPTE" if within else "REFUSE"
    return {
        "variant": variant,
        "macro_f1": macro,
        "accuracy": acc,
        "delta_macro_f1": float(delta),
        "disk_mib": float(disk_mib),
        "compression_ratio": float(compression),
        "mean_latency_ms": timing["mean_latency_ms"],
        "median_latency_ms": timing["median_latency_ms"],
        "p95_latency_ms": timing["p95_latency_ms"],
        "responses_per_second": timing["responses_per_second"],
        "speedup": float(speedup),
        "within_margin": "OUI" if within else "NON",
        "decision": decision,
    }


def try_torchscript_export(model: torch.nn.Module, loader: DataLoader, path: Path) -> tuple[torch.nn.Module | None, float, str]:
    try:
        batch = next(iter(loader))
        wrapper = InferenceWrapper(copy.deepcopy(model)).eval()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, (batch["input_ids"], batch["attention_mask"]), strict=False)
            source = wrapper(batch["input_ids"], batch["attention_mask"])
            exported = traced(batch["input_ids"], batch["attention_mask"])
        if not torch.allclose(source, exported, atol=1e-4, rtol=1e-4):
            return None, 0.0, "TorchScript refuse: sorties non coherentes."
        path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(path))
        loaded = torch.jit.load(str(path), map_location="cpu").eval()
        return loaded, directory_size_mib(path), "TorchScript exporte et recharge."
    except Exception as exc:
        return None, 0.0, f"TorchScript non compatible: {type(exc).__name__}: {exc}"


def plot_phase16(rows: list[dict[str, object]], output_dir: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    x = np.arange(len(rows))
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - 0.18, [float(row["macro_f1"]) for row in rows], width=0.36, label="macro-F1")
    plt.bar(x + 0.18, [float(row["disk_mib"]) for row in rows], width=0.36, label="MiB disque")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("Phase 16 - Score et poids")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "phase16_score_size.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - 0.18, [float(row["mean_latency_ms"]) for row in rows], width=0.36, label="latence moyenne ms")
    plt.bar(x + 0.18, [float(row["responses_per_second"]) for row in rows], width=0.36, label="reponses/s")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("Phase 16 - Temps de reponse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "phase16_latency.png", dpi=160)
    plt.close()


def run_phase16(
    splits: dict[str, tuple[list[str], list[int]]],
    tokenizer: Any,
    classes: list[str],
    output_dir: Path,
    checkpoint_dir: Path = Path("checkpoints/phase14/lora"),
) -> dict[str, object]:
    assert PHASE16_MAX_MACRO_F1_DROP == 0.02
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    phase_dir = Path("checkpoints") / "phase16"
    phase_dir.mkdir(parents=True, exist_ok=True)
    loader, y_true = make_phase16_loader(splits, tokenizer, PHASE16_MAX_LENGTH, PHASE16_BENCHMARK_N)
    assert len(y_true) == len(list(loader))
    model = load_phase14_model(len(classes), checkpoint_dir)

    start = time.perf_counter()
    baseline_size = save_state_artifact(model, phase_dir / "baseline_fp32_state.pt")
    baseline = evaluate_variant("baseline", model, loader, y_true, baseline_size)

    quant_model = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    quant_size = save_state_artifact(quant_model, phase_dir / "dynamic_int8_state.pt")
    quant = evaluate_variant("quantification", quant_model, loader, y_true, quant_size, baseline)

    rows = [baseline, quant]
    export_note = ""
    exported, export_size, export_note = try_torchscript_export(model, loader, phase_dir / "torchscript_fp32.pt")
    if exported is not None:
        rows.append(evaluate_variant("torchscript", exported, loader, y_true, export_size, baseline))

    admissible = [row for row in rows[1:] if row["within_margin"] == "OUI"]
    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                float(row["speedup"]) - 1.0,
                float(row["compression_ratio"]) - 1.0,
                float(row["macro_f1"]),
            ),
        )
        distillation = (
            "Distillation identifiee comme troisieme piste mais non executee car une solution acceptable "
            "a deja ete obtenue avec un cout experimental bien inferieur."
        )
    else:
        best = baseline
        distillation = "Distillation serait la piste suivante car les optimisations sans entrainement ne respectent pas la marge."

    pd.DataFrame(rows).to_csv(output_dir / "phase16_optimization.csv", index=False)
    plot_phase16(rows, output_dir)
    return {
        "rows": rows,
        "best": best,
        "reference_model": str(checkpoint_dir),
        "historical_macro_f1": 0.12812138908869683,
        "protocol": {
            "device": "cpu",
            "max_length": PHASE16_MAX_LENGTH,
            "benchmark_n": len(y_true),
            "batch_size": 1,
            "warmup": PHASE16_WARMUP,
            "repeats": PHASE16_REPEATS,
            "margin": PHASE16_MAX_MACRO_F1_DROP,
            "tokenizer": MODEL_NAME,
        },
        "export_note": export_note,
        "distillation_note": distillation,
        "elapsed_s": time.perf_counter() - start,
    }
