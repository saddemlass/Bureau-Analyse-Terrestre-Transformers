from __future__ import annotations

import csv
import argparse
import os
import re
import time
import urllib.error
import urllib.request
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from models import (
    ConvTextClassifier,
    SingleHeadSelfAttention,
    TextClassifier,
    TwoHeadSelfAttention,
    positional_encoding,
    receptive_field_table,
)
from training import (
    build_vocab,
    class_scores,
    encode_texts,
    evaluate_model,
    make_loader,
    predict_classes,
    predict_logits,
    set_seeds,
    train_model,
)

from pretrained import (
    MODEL_NAME,
    choose_max_length,
    make_loaders as make_pretrained_loaders,
    model_metadata,
    run_regime,
    tokenizer_audit,
)


DATA_URLS = [
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-complete-geocoded-time-standardized.csv",
    "https://github.com/planetsig/ufo-reports/raw/master/csv-data/ufo-complete-geocoded-time-standardized.csv",
]
LOCAL_CSV = Path("releves_klaxo3.csv")
EXPECTED_COLUMNS = [
    "datetime",
    "city",
    "state",
    "country",
    "shape",
    "duration_seconds",
    "duration_hours_min",
    "comments",
    "date_posted",
    "latitude",
    "longitude",
]
MIN_PLAUSIBLE_BYTES = 10_000_000
MAX_RETRIES = 4
USER_AGENT = "Bureau-Analyse-Terrestre-Transformers/0.1 (+student-analysis)"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class LoadReport:
    physical_lines: int
    valid_lines: int
    repaired_lines: int
    skipped_lines: int


def download_dataset(path: Path = LOCAL_CSV) -> None:
    if path.exists() and path.stat().st_size >= MIN_PLAUSIBLE_BYTES:
        print(f"Dataset local trouve: {path} ({path.stat().st_size:,} octets)")
        return

    if path.exists():
        print(f"Fichier local trop petit ({path.stat().st_size:,} octets). Nouveau telechargement.")

    last_error: Exception | None = None
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    for url in DATA_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Telechargement dataset: tentative {attempt}/{MAX_RETRIES}")
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=45) as response:
                    status = getattr(response, "status", 200)
                    if status >= 500 or status in {408, 429}:
                        raise urllib.error.HTTPError(
                            url, status, "Erreur HTTP temporaire", response.headers, None
                        )
                    if status >= 400:
                        raise RuntimeError(f"Erreur HTTP {status} sur {url}")

                    with NamedTemporaryFile("wb", delete=False, dir=".") as tmp:
                        temp_name = Path(tmp.name)
                        while chunk := response.read(1024 * 1024):
                            tmp.write(chunk)

                size = temp_name.stat().st_size
                if size < MIN_PLAUSIBLE_BYTES:
                    temp_name.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Telechargement incomplet: {size:,} octets, "
                        f"attendu au moins {MIN_PLAUSIBLE_BYTES:,}"
                    )

                tmp_path.unlink(missing_ok=True)
                temp_name.replace(tmp_path)
                tmp_path.replace(path)
                print(f"Dataset telecharge: {path} ({size:,} octets)")
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"Echec: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        "Impossible de telecharger le dataset apres plusieurs essais. "
        "Verifiez la connexion ou placez le fichier officiel sous releves_klaxo3.csv."
    ) from last_error


def normalize_row(row: list[str]) -> tuple[list[str] | None, bool]:
    if len(row) == len(EXPECTED_COLUMNS):
        return row, False
    if len(row) > len(EXPECTED_COLUMNS):
        repaired = row[:7] + [",".join(row[7:-3])] + row[-3:]
        if len(repaired) == len(EXPECTED_COLUMNS):
            return repaired, True
    return None, False


def load_dataset(path: Path = LOCAL_CSV) -> tuple[pd.DataFrame, LoadReport]:
    physical_lines = 0
    records: list[list[str]] = []
    skipped_lines = repaired_lines = 0

    with path.open("r", encoding="utf-8", newline="", errors="replace") as handle:
        for physical_lines, row in enumerate(csv.reader(handle), start=1):
            normalized, repaired = normalize_row(row)
            if normalized is None:
                skipped_lines += 1
                continue
            repaired_lines += int(repaired)
            records.append(normalized)

    df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
    return df, LoadReport(physical_lines, len(df), repaired_lines, skipped_lines)


def parse_observation_datetime(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    has_2400 = raw.str.contains(r"\b24:00\b", regex=True, na=False)
    parsed = pd.to_datetime(
        raw.str.replace(r"\b24:00\b", "00:00", regex=True),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    parsed.loc[has_2400 & parsed.notna()] += pd.Timedelta(days=1)
    return parsed


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["observation_dt"] = parse_observation_datetime(prepared["datetime"])
    prepared["observation_date"] = prepared["observation_dt"].dt.date
    prepared["year"] = prepared["observation_dt"].dt.year
    prepared["month"] = prepared["observation_dt"].dt.month
    prepared["weekday"] = prepared["observation_dt"].dt.day_name()
    return prepared


def covered_calendar_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.date() - start.date()).days + 1


def july_fourth_stats(dated: pd.DataFrame, daily_counts: pd.Series) -> dict[str, object]:
    day_index = pd.to_datetime(daily_counts.index)
    july4_daily = daily_counts[(day_index.month == 7) & (day_index.day == 4)].sort_values(
        ascending=False
    )
    total = int(
        ((dated["observation_dt"].dt.month == 7) & (dated["observation_dt"].dt.day == 4)).sum()
    )
    start = dated["observation_dt"].min().normalize()
    end = dated["observation_dt"].max().normalize()
    calendar_days = sum(
        start <= pd.Timestamp(f"{year}-07-04") <= end
        for year in range(int(start.year), int(end.year) + 1)
    )
    top_count = int(july4_daily.iloc[0])

    return {
        "total": total,
        "days": int(len(july4_daily)),
        "calendar_days": calendar_days,
        "mean_on_observed_july4": float(july4_daily.mean()),
        "mean_on_calendar_july4": total / calendar_days,
        "top_date": pd.to_datetime(july4_daily.index[0]).date(),
        "top_count": top_count,
        "top_rank": int((daily_counts > top_count).sum() + 1),
    }


def annual_growth_verdict(annual: pd.Series) -> tuple[bool, list[tuple[int, int, int]]]:
    drops: list[tuple[int, int, int]] = []
    previous_count: int | None = None
    for year, count in annual.items():
        count = int(count)
        if previous_count is not None and count < previous_count:
            drops.append((int(year), previous_count, count))
        previous_count = count
    return not drops, drops


def top10_days(daily_counts: pd.Series) -> pd.DataFrame:
    top10 = daily_counts.sort_values(ascending=False, kind="mergesort").head(10)
    dates = pd.to_datetime(top10.index)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "nombre_releves": top10.to_numpy(),
            "jour_semaine": dates.day_name(),
            "mois": dates.month,
        }
    )


def compute_phase0(df: pd.DataFrame, year_min: int | None, year_max: int | None) -> dict[str, object]:
    dated = df.dropna(subset=["observation_dt"]).copy()
    if year_min is not None and year_max is not None:
        dated = dated[(dated["year"] >= year_min) & (dated["year"] <= year_max)].copy()

    start = dated["observation_dt"].min()
    end = dated["observation_dt"].max()
    daily_counts = dated.groupby("observation_date").size().sort_index()
    annual = dated.groupby("year").size().sort_index()
    growth_continuous, growth_drops = annual_growth_verdict(annual)
    records = len(dated)
    days = covered_calendar_days(start, end)

    return {
        "label": "donnees completes" if year_min is None else f"periode {year_min}-{year_max}",
        "df": dated,
        "start": start,
        "end": end,
        "days": days,
        "records": records,
        "mean_per_day": records / days,
        "weekday_pct": (dated["weekday"].value_counts(normalize=True) * 100).reindex(WEEKDAYS),
        "month_pct": (dated["month"].value_counts(normalize=True) * 100).reindex(range(1, 13)),
        "annual": annual,
        "growth_continuous": growth_continuous,
        "growth_drops": growth_drops,
        "top10": top10_days(daily_counts),
        "july4": july_fourth_stats(dated, daily_counts),
        "max_daily_count": int(daily_counts.max()),
        "max_daily_date": pd.to_datetime(daily_counts.idxmax()).date(),
    }


def save_outputs(result: dict[str, object]) -> None:
    result["top10"].to_csv(OUTPUT_DIR / "phase0_top10_journees.csv", index=False)

    annual = result["annual"]
    plt.figure(figsize=(10, 5.5))
    plt.plot(annual.index, annual.values, marker="o", linewidth=1.8)
    plt.title("Phase 0 - Volume annuel des releves")
    plt.xlabel("Annee")
    plt.ylabel("Nombre de releves")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase0_volume_annuel.png", dpi=160)
    plt.close()


def print_phase0(load_report: LoadReport, full: dict[str, object], selected: dict[str, object]) -> None:
    july4 = selected["july4"]
    weekday_pct = selected["weekday_pct"]
    month_pct = selected["month_pct"]

    print("\n=== PHASE 0 - REFAIRE LES CALCULS DU DISPARU ===")
    print(
        "Chargement: "
        f"{load_report.physical_lines} lignes physiques, "
        f"{load_report.valid_lines} lignes chargees, "
        f"{load_report.repaired_lines} reparees, "
        f"{load_report.skipped_lines} malformees ignorees apres journalisation."
    )
    print(
        "Date choisie: datetime, car elle indique le moment de l'observation; "
        "date_posted indique seulement la publication."
    )
    print(
        f"Test donnees completes: {full['records']} releves, "
        f"{full['days']} jours, moyenne {full['mean_per_day']:.2f}/jour."
    )
    print(f"Periode analysee: {selected['label']}")
    print(
        f"Date min: {selected['start'].date()} | "
        f"Date max: {selected['end'].date()} | "
        f"Jours couverts: {selected['days']} | "
        f"Releves retenus: {selected['records']}"
    )
    print(f"Moyenne par jour: {selected['mean_per_day']:.2f}")
    print(
        "4 juillet: "
        f"total={july4['total']}, "
        f"dates calendaires couvertes={july4['calendar_days']}, "
        f"moyenne calendaire={july4['mean_on_calendar_july4']:.2f}, "
        f"moyenne sur dates observees={july4['mean_on_observed_july4']:.2f}, "
        f"plus charge={july4['top_date']} ({july4['top_count']} releves)"
    )
    print(f"Samedi: {weekday_pct['Saturday']:.1f} %")
    print(f"Lundi: {weekday_pct['Monday']:.1f} %")
    print(f"Juillet: {month_pct[7]:.1f} %")
    print(f"Fevrier: {month_pct[2]:.1f} %")
    print(f"Maximum quotidien: {selected['max_daily_count']} releves le {selected['max_daily_date']}")
    print(
        f"4 juillet le plus charge: {july4['top_date']} "
        f"({july4['top_count']} releves), rang {july4['top_rank']}"
    )
    print(f"Croissance continue: {'VRAI' if selected['growth_continuous'] else 'FAUX'}")
    if selected["growth_drops"]:
        drops = ", ".join(f"{year}: {before}->{after}" for year, before, after in selected["growth_drops"])
        print(f"Baisses annuelles observees: {drops}")

    print("\nTop 10 journees les plus chargees:")
    print(selected["top10"].to_string(index=False))


def compute_phase1(df: pd.DataFrame) -> dict[str, object]:
    candidates = df[
        df["comments"].astype(str).str.len().between(80, 260)
        & df["shape"].astype(str).str.strip().ne("")
    ].copy()
    examples: list[dict[str, str]] = []
    for shape in ["light", "triangle", "fireball"]:
        row = candidates[candidates["shape"].str.lower() == shape].sort_values("observation_dt").iloc[0]
        examples.append(
            {
                "datetime": str(row["datetime"]),
                "city": str(row["city"]),
                "state": str(row["state"]),
                "country": str(row["country"]),
                "shape": str(row["shape"]),
                "comments": str(row["comments"]),
            }
        )
    return {
        "examples": examples,
        "task": {
            "input": "comments",
            "output": "shape",
            "sentence": "A partir du temoignage ecrit par un temoin, predire la forme de l'objet qu'il decrit.",
        },
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_phase2(df: pd.DataFrame) -> dict[str, object]:
    torch.manual_seed(7)
    candidates = df[
        df["comments"].astype(str).str.strip().ne("")
        & df["shape"].astype(str).str.strip().ne("")
    ].copy()
    candidates["source_index"] = candidates.index
    candidates["shape_key"] = candidates["shape"].astype(str).str.strip().str.lower()
    shape_order = candidates["shape_key"].value_counts().sort_values(ascending=False).index[:8]
    examples = (
        candidates[candidates["shape_key"].isin(shape_order)]
        .sort_values(["shape_key", "observation_dt", "source_index"])
        .groupby("shape_key", sort=True)
        .head(1)
        .sort_values("source_index")
        .reset_index(drop=True)
    )

    texts = examples["comments"].astype(str).tolist()
    shapes = examples["shape_key"].tolist()
    vocab = sorted({token for text in texts for token in tokenize(text)})
    token_to_idx = {token: i for i, token in enumerate(vocab)}
    shape_to_id = {shape: i for i, shape in enumerate(sorted(set(shapes)))}
    id_to_shape = {i: shape for shape, i in shape_to_id.items()}

    x = torch.zeros((len(texts), len(vocab)), dtype=torch.float32)
    for row_idx, text in enumerate(texts):
        for token in tokenize(text):
            x[row_idx, token_to_idx[token]] += 1.0
    y = torch.tensor([shape_to_id[shape] for shape in shapes], dtype=torch.long)

    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, len(shape_to_id)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []

    with torch.no_grad():
        initial_loss = float(loss_fn(model(x), y).item())

    iterations = 0
    for epoch in range(1, 5001):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        iterations = epoch
        losses.append(float(loss.item()))

        with torch.no_grad():
            predictions = model(x).argmax(dim=1)
            correct = int((predictions == y).sum().item())
        if correct == len(y) and losses[-1] < 0.02:
            break

    with torch.no_grad():
        final_logits = model(x)
        final_loss = float(loss_fn(final_logits, y).item())
        predictions = final_logits.argmax(dim=1)
        correct = int((predictions == y).sum().item())

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Phase 2 - Loss overfit sur 8 releves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = OUTPUT_DIR / "phase2_overfit_loss.png"
    plt.savefig(loss_path, dpi=160)
    plt.close()

    assert correct == 8, f"Phase 2 echouee: {correct}/8 predictions correctes"

    return {
        "examples": examples,
        "vocab_size": len(vocab),
        "shape_to_id": shape_to_id,
        "id_to_shape": id_to_shape,
        "predictions": [id_to_shape[int(pred)] for pred in predictions],
        "correct": correct,
        "iterations": iterations,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_path": loss_path,
        "optimizer": "Adam",
        "learning_rate": 0.05,
    }


def print_phase2(result: dict[str, object]) -> None:
    examples = result["examples"]
    predictions = result["predictions"]

    print("\n=== PHASE 2 - TEST D'ACCEPTATION SUR 8 RELEVES ===")
    print("Exemples retenus:")
    for i, row in examples.iterrows():
        comment = " ".join(str(row["comments"]).split())[:90]
        print(f"- index {row['source_index']} | shape={row['shape_key']} | {comment}")

    print("Classes presentes:", ", ".join(result["shape_to_id"].keys()))
    print(f"Vocabulaire: {result['vocab_size']} mots")
    print("\nPredictions finales:")
    print("exemple | shape reelle | shape predite | correct ?")
    for i, row in examples.iterrows():
        predicted = predictions[i]
        actual = row["shape_key"]
        print(f"{i + 1} | {actual} | {predicted} | {'oui' if predicted == actual else 'non'}")
    print(f"Accuracy sur les 8 exemples: {result['correct']}/8 = 100 %")
    print(f"Nombre d'iterations necessaires: {result['iterations']}")
    print(f"Loss initiale: {result['initial_loss']:.4f} | Loss finale: {result['final_loss']:.4f}")
    print(f"Courbe de loss: {result['loss_path']}")


def normalize_shape(shape: object) -> str:
    value = str(shape).strip().lower()
    return {"round": "circle", "changed": "changing"}.get(value, value)


def prepare_shape_task(df: pd.DataFrame, load_report: LoadReport, seed: int = 7) -> dict[str, object]:
    audit_shape = df["shape"].astype(str).str.strip().str.lower()
    normalized = audit_shape.map(lambda s: {"round": "circle", "changed": "changing"}.get(s, s))
    work = df.copy()
    work["shape_clean"] = normalized
    work["comments_clean"] = work["comments"].astype(str).str.strip()
    labelled = work[
        work["shape_clean"].ne("")
        & ~work["shape_clean"].isin(["unknown", "other"])
        & work["comments_clean"].ne("")
    ].copy()
    counts = labelled["shape_clean"].value_counts()
    kept_classes = sorted(counts[counts >= 10].index.tolist())
    labelled = labelled[labelled["shape_clean"].isin(kept_classes)].copy()
    labelled["row_id"] = labelled.index
    rare_counts = counts[counts < 10].sort_values(ascending=False)

    train_idx, temp_idx = train_test_split(
        labelled.index,
        test_size=0.30,
        random_state=seed,
        stratify=labelled["shape_clean"],
    )
    temp = labelled.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp.index,
        test_size=0.50,
        random_state=seed,
        stratify=temp["shape_clean"],
    )
    class_to_id = {name: i for i, name in enumerate(kept_classes)}
    labelled["y"] = labelled["shape_clean"].map(class_to_id).astype(int)
    return {
        "data": labelled,
        "train_idx": np.array(train_idx),
        "val_idx": np.array(val_idx),
        "test_idx": np.array(test_idx),
        "classes": kept_classes,
        "class_to_id": class_to_id,
        "audit": {
            "initial": len(df),
            "missing": int(audit_shape.eq("").sum()),
            "missing_valid_csv_rows": int(audit_shape.eq("").sum() - load_report.repaired_lines),
            "missing_repaired_rows": load_report.repaired_lines,
            "unknown": int(audit_shape.eq("unknown").sum()),
            "other": int(audit_shape.eq("other").sum()),
            "before_low_support_lines": int(counts.sum()),
            "before_low_support_classes": int(len(counts)),
            "low_support_removed": int(rare_counts.sum()),
            "low_support_classes_removed": int(len(rare_counts)),
            "low_support_classes": {str(k): int(v) for k, v in rare_counts.items()},
            "kept": len(labelled),
            "n_classes": len(kept_classes),
        },
    }


def subset(task: dict[str, object], split: str) -> pd.DataFrame:
    return task["data"].loc[task[f"{split}_idx"]].copy()


def majority_baseline(task: dict[str, object]) -> dict[str, float | str]:
    train = subset(task, "train")
    test = subset(task, "test")
    majority = train["y"].value_counts().idxmax()
    pred = np.full(len(test), majority)
    return {
        "class": task["classes"][int(majority)],
        "accuracy": accuracy_score(test["y"], pred),
        "macro_f1": f1_score(test["y"], pred, average="macro", zero_division=0),
        "train_time": 0.0,
    }


def linear_baseline(task: dict[str, object]) -> dict[str, float]:
    train = subset(task, "train")
    test = subset(task, "test")
    start = time.perf_counter()
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-z0-9]+\b", lowercase=True, max_features=250)
    x_train = vectorizer.fit_transform(train["comments_clean"])
    x_test = vectorizer.transform(test["comments_clean"])
    model = LogisticRegression(max_iter=400, solver="lbfgs", random_state=7)
    model.fit(x_train, train["y"])
    train_time = time.perf_counter() - start
    pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(test["y"], pred),
        "macro_f1": f1_score(test["y"], pred, average="macro", zero_division=0),
        "train_time": train_time,
        "vocab_size": len(vectorizer.vocabulary_),
    }


def make_torch_data(task: dict[str, object], max_vocab: int, max_len: int, batch_size: int) -> dict[str, object]:
    train = subset(task, "train")
    val = subset(task, "val")
    test = subset(task, "test")
    vocab = build_vocab(train["comments_clean"].tolist(), max_vocab=max_vocab, min_freq=2)
    x_train = encode_texts(train["comments_clean"].tolist(), vocab, max_len=max_len)
    x_val = encode_texts(val["comments_clean"].tolist(), vocab, max_len=max_len)
    x_test = encode_texts(test["comments_clean"].tolist(), vocab, max_len=max_len)
    y_train = torch.tensor(train["y"].to_numpy(), dtype=torch.long)
    y_val = torch.tensor(val["y"].to_numpy(), dtype=torch.long)
    y_test = torch.tensor(test["y"].to_numpy(), dtype=torch.long)
    return {
        "vocab": vocab,
        "train_loader": make_loader(x_train, y_train, batch_size, True),
        "val_loader": make_loader(x_val, y_val, batch_size, False),
        "test_loader": make_loader(x_test, y_test, batch_size, False),
        "x_val": x_val,
        "y_val_tensor": y_val,
        "x_test": x_test,
        "y_test_tensor": y_test,
        "train_y": y_train.numpy(),
        "val_y": y_val.numpy(),
        "test_y": y_test.numpy(),
    }


def run_torch_experiment(task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    set_seeds(7)
    data = make_torch_data(task, config["max_vocab"], config["max_len"], config["batch_size"])
    model = TextClassifier(
        len(data["vocab"]),
        len(task["classes"]),
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    val_loader = data["val_loader"]
    if config["batch_size"] == 4:
        val_loader = make_loader(data["x_val"], data["y_val_tensor"], 128, False)
    history = train_model(
        model,
        data["train_loader"],
        val_loader,
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        max_train_batches=config.get("max_train_batches"),
    )
    eval_loader = data["test_loader"]
    if config["batch_size"] == 4:
        eval_loader = make_loader(data["x_test"], data["y_test_tensor"], 128, False)
    metrics = evaluate_model(model, eval_loader, data["test_y"])
    raw_text = subset(task, "train")["comments_clean"].iloc[0]
    tokens = tokenize(raw_text)
    ids = [data["vocab"].get(token, 1) for token in tokens[: min(16, len(tokens))]]
    return {
        "model": model,
        "data": data,
        "history": history,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_time": history.train_time,
        "vocab_size": len(data["vocab"]),
        "example": {"text": raw_text, "tokens": tokens[:16], "ids": ids},
        "config": config,
    }


def plot_losses(path: Path, title: str, train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def compute_phase3(task: dict[str, object]) -> dict[str, object]:
    majority = majority_baseline(task)
    linear = linear_baseline(task)
    attempts = []
    configs = [
        {
            "name": "base",
            "max_vocab": 9000,
            "max_len": 80,
            "batch_size": 128,
            "emb_dim": 96,
            "hidden_dim": 160,
            "dropout": 0.25,
            "lr": 0.003,
            "max_epochs": 12,
            "patience": 4,
        },
        {
            "name": "vocab_12000",
            "max_vocab": 12000,
            "max_len": 80,
            "batch_size": 128,
            "emb_dim": 96,
            "hidden_dim": 160,
            "dropout": 0.25,
            "lr": 0.003,
            "max_epochs": 12,
            "patience": 4,
        },
    ]
    chosen = None
    for config in configs:
        result = run_torch_experiment(task, config)
        attempts.append(
            f"{config['name']}: macro-F1={result['macro_f1']:.4f}, temps={result['train_time']:.2f}s"
        )
        if result["macro_f1"] > linear["macro_f1"]:
            chosen = result
            break
    if chosen is None:
        chosen = result
    plot_losses(
        OUTPUT_DIR / "phase3_train_val_loss.png",
        "Phase 3 - Loss train/validation",
        chosen["history"].train_losses,
        chosen["history"].val_losses,
    )
    return {"majority": majority, "linear": linear, "torch": chosen, "attempts": attempts}


def unstable_eval(model: torch.nn.Module, loader, y_true: np.ndarray, repeats: int = 5) -> list[float]:
    scores = []
    model.train()
    with torch.no_grad():
        for _ in range(repeats):
            preds = []
            for xb, _ in loader:
                preds.append(model(xb).argmax(dim=1))
            scores.append(f1_score(y_true, torch.cat(preds).numpy(), average="macro", zero_division=0))
    model.eval()
    return scores


def compute_phase4(task: dict[str, object], phase3: dict[str, object]) -> dict[str, object]:
    base_config = phase3["torch"]["config"]
    panne1_scores = unstable_eval(
        phase3["torch"]["model"], phase3["torch"]["data"]["test_loader"], phase3["torch"]["data"]["test_y"]
    )
    train_score = evaluate_model(
        phase3["torch"]["model"],
        phase3["torch"]["data"]["train_loader"],
        phase3["torch"]["data"]["train_y"],
    )["macro_f1"]

    n_classes = len(task["classes"])
    y_true = phase3["torch"]["data"]["test_y"]
    good_loss = phase3["torch"]["history"].val_losses
    model_preds = []
    phase3["torch"]["model"].eval()
    with torch.no_grad():
        for xb, _ in phase3["torch"]["data"]["test_loader"]:
            model_preds.append(phase3["torch"]["model"](xb).argmax(dim=1))
    decoded_wrong = (torch.cat(model_preds).numpy() + 1) % n_classes
    panne2_bad_f1 = f1_score(y_true, decoded_wrong, average="macro", zero_division=0)

    flat_config = {**base_config, "name": "panne3", "lr": 1e-7, "max_epochs": 4, "patience": 4}
    panne3 = run_torch_experiment(task, flat_config)

    plt.figure(figsize=(8, 4.8))
    plt.plot(panne1_scores, marker="o", label="test macro-F1 avec dropout actif")
    plt.axhline(phase3["torch"]["macro_f1"], color="black", linestyle="--", label="eval correct")
    plt.title("Phase 4 - Panne 1: oubli de model.eval()")
    plt.xlabel("Evaluation repetee")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_panne1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(good_loss, label="validation loss")
    plt.axhline(panne2_bad_f1, color="red", linestyle="--", label="macro-F1 avec labels decales")
    plt.title("Phase 4 - Panne 2: mapping de classes decale")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_panne2.png", dpi=160)
    plt.close()

    plot_losses(
        OUTPUT_DIR / "phase4_panne3.png",
        "Phase 4 - Panne 3: learning rate trop faible",
        panne3["history"].train_losses,
        panne3["history"].val_losses,
    )
    return {
        "panne1": {"scores": panne1_scores, "train_f1": train_score},
        "panne2": {"bad_f1": panne2_bad_f1, "losses": good_loss},
        "panne3": {
            "first_loss": panne3["history"].train_losses[0],
            "last_loss": panne3["history"].train_losses[-1],
            "macro_f1": panne3["macro_f1"],
        },
    }


def compute_phase5(task: dict[str, object], phase3: dict[str, object]) -> dict[str, object]:
    base = phase3["torch"]
    base_config = base["config"]
    experiments = []
    candidates = [
        ("emb_dim 48", {**base_config, "emb_dim": 48}),
        ("batch_size 512", {**base_config, "batch_size": 512}),
        ("hidden_dim 80", {**base_config, "hidden_dim": 80}),
        ("max_len 60", {**base_config, "max_len": 60}),
        ("patience 2", {**base_config, "patience": 2}),
    ]
    for name, config in candidates:
        result = run_torch_experiment(task, {**config, "name": name})
        experiments.append(
            {
                "reglage": name,
                "temps": result["train_time"],
                "facteur_gain": base["train_time"] / result["train_time"],
                "macro_f1": result["macro_f1"],
                "ecart_score": result["macro_f1"] - base["macro_f1"],
                "config": config,
            }
        )

    final_candidates = [
        {**base_config, "emb_dim": 48, "name": "phase5_final_emb48"},
        {**base_config, "max_len": 60, "name": "phase5_final_len60"},
        {**base_config, "patience": 2, "name": "phase5_final_patience2"},
    ]
    final = None
    for config in final_candidates:
        candidate = run_torch_experiment(task, config)
        if candidate["macro_f1"] >= base["macro_f1"] and candidate["train_time"] < base["train_time"]:
            final = candidate
            break
        if final is None or candidate["macro_f1"] > final["macro_f1"]:
            final = candidate

    pd.DataFrame([{k: v for k, v in row.items() if k != "config"} for row in experiments]).to_csv(
        OUTPUT_DIR / "phase5_experiments.csv", index=False
    )
    plt.figure(figsize=(8, 4.8))
    plt.plot(base["history"].elapsed_times, base["history"].val_losses, label="Phase 3")
    plt.plot(final["history"].elapsed_times, final["history"].val_losses, label="Phase 5")
    plt.title("Phase 5 - Loss validation en fonction du temps")
    plt.xlabel("Temps ecoule (s)")
    plt.ylabel("Validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5_time_comparison.png", dpi=160)
    plt.close()
    return {"experiments": experiments, "final": final}


def token_length_stats(task: dict[str, object]) -> dict[str, object]:
    lengths = task["data"]["comments_clean"].map(lambda text: len(tokenize(text)))
    return {
        "max_tokens_before_truncation": int(lengths.max()),
        "median_tokens": float(lengths.median()),
        "pct_covered_by_60": float((lengths <= 60).mean()),
    }


def run_conv_experiment(task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    set_seeds(7)
    data = make_torch_data(task, config["max_vocab"], config["max_len"], config["batch_size"])
    model = ConvTextClassifier(
        len(data["vocab"]),
        len(task["classes"]),
        emb_dim=config["emb_dim"],
        channels=config["channels"],
        dilations=tuple(config["dilations"]),
        kernel=config["kernel"],
        dropout=config["dropout"],
        norm=config["norm"],
    )
    history = train_model(
        model,
        data["train_loader"],
        data["val_loader"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        max_train_batches=config.get("max_train_batches"),
    )
    eval_loader = data["test_loader"]
    if config["batch_size"] == 4:
        eval_loader = make_loader(data["x_test"], data["y_test_tensor"], 128, False)
    metrics = evaluate_model(model, eval_loader, data["test_y"])
    return {
        "model": model,
        "data": data,
        "history": history,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_time": history.train_time,
        "config": config,
    }


def phase6_config(norm: str = "batch", batch_size: int = 128) -> dict[str, object]:
    config = {
        "name": f"conv_{norm}_b{batch_size}",
        "max_vocab": 12000,
        "max_len": 60,
        "batch_size": batch_size,
        "emb_dim": 32,
        "channels": 16,
        "dilations": [1, 2, 4, 8, 16],
        "kernel": 3,
        "dropout": 0.20,
        "norm": norm,
        "lr": 0.003,
        "max_epochs": 2,
        "patience": 1,
    }
    return config


def phase7_config(norm: str, batch_size: int, max_train_batches: int | None = None) -> dict[str, object]:
    config = phase6_config(norm, batch_size)
    if max_train_batches is not None:
        config["max_train_batches"] = max_train_batches
    return config


def token_start_probe(task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    set_seeds(7)
    train = subset(task, "train")
    vocab = build_vocab(train["comments_clean"].tolist(), max_vocab=config["max_vocab"], min_freq=2)
    row = train[train["comments_clean"].map(lambda text: len(tokenize(text)) >= 4)].iloc[0]
    tokens = tokenize(row["comments_clean"])
    replacement = "zzztoken"
    changed_tokens = [replacement] + tokens[1:]
    original = " ".join(tokens)
    changed = " ".join(changed_tokens)
    x_original = encode_texts([original], vocab, max_len=config["max_len"])
    x_changed = encode_texts([changed], vocab, max_len=config["max_len"])
    model = ConvTextClassifier(
        len(vocab),
        len(task["classes"]),
        emb_dim=config["emb_dim"],
        channels=config["channels"],
        dilations=tuple(config["dilations"]),
        kernel=config["kernel"],
        dropout=config["dropout"],
        norm=config["norm"],
    )
    model.eval()
    with torch.no_grad():
        diff = model(x_original) - model(x_changed)
    return {
        "source_index": int(row["row_id"]),
        "text": str(row["comments_clean"]),
        "token_original": tokens[0],
        "token_modified": replacement,
        "logit_l2": float(torch.linalg.vector_norm(diff).item()),
        "logit_max_abs": float(diff.abs().max().item()),
    }


def compute_phase6(task: dict[str, object], phase3: dict[str, object] | None = None, phase5: dict[str, object] | None = None) -> dict[str, object]:
    config = phase6_config("batch", 128)
    stats = token_length_stats(task)
    rf_rows = receptive_field_table([config["kernel"]] * len(config["dilations"]), config["dilations"], [1] * len(config["dilations"]))
    pd.DataFrame(rf_rows).to_csv(OUTPUT_DIR / "phase6_receptive_field.csv", index=False)
    probe = token_start_probe(task, config)
    result = run_conv_experiment(task, config)
    plot_losses(
        OUTPUT_DIR / "phase6_train_val_loss.png",
        "Phase 6 - Loss train/validation modele convolutionnel",
        result["history"].train_losses,
        result["history"].val_losses,
    )
    reference = phase5["final"] if phase5 is not None else (phase3["torch"] if phase3 is not None else None)
    return {
        "stats": stats,
        "rf_rows": rf_rows,
        "probe": probe,
        "conv": result,
        "reference": reference,
    }


def compute_phase7(task: dict[str, object], phase6: dict[str, object]) -> dict[str, object]:
    old_batch4 = run_conv_experiment(task, phase7_config("batch", 4, max_train_batches=10))
    fixed_batch4 = run_conv_experiment(task, phase7_config("group", 4, max_train_batches=6000))
    fixed_normal = run_conv_experiment(task, phase7_config("group", phase6["conv"]["config"]["batch_size"]))

    plt.figure(figsize=(9, 5))
    plt.plot(old_batch4["history"].val_losses, label="ancien BatchNorm, batch=4")
    plt.plot(fixed_batch4["history"].val_losses, label="corrige GroupNorm, batch=4")
    plt.title("Phase 7 - Comparaison batch size 4")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_batch4_comparison.png", dpi=160)
    plt.close()

    xb, _ = next(iter(fixed_batch4["data"]["test_loader"]))
    ok_batch1 = False
    fixed_batch4["model"].eval()
    with torch.no_grad():
        logits = fixed_batch4["model"](xb[:1])
        ok_batch1 = tuple(logits.shape) == (1, len(task["classes"])) and torch.isfinite(logits).all().item()

    return {
        "old_batch4": old_batch4,
        "fixed_batch4": fixed_batch4,
        "fixed_normal": fixed_normal,
        "batch1_ok": bool(ok_batch1),
    }


def plural_variants(word: str) -> set[str]:
    variants = {word}
    if word.endswith("s"):
        variants.add(word + "es")
    elif word.endswith("y"):
        variants.add(word[:-1] + "ies")
    else:
        variants.add(word + "s")
    return variants


def forbidden_shape_words(classes: list[str]) -> list[str]:
    words: set[str] = set()
    for shape in classes:
        words.update(plural_variants(shape))
    words.update({"round", "rounds", "circle", "circles", "changed", "changing", "light", "lights"})
    return sorted(words)


def forbidden_regex(words: list[str]) -> re.Pattern[str]:
    pattern = "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", re.IGNORECASE)


def count_forbidden(texts: pd.Series, regex: re.Pattern[str]) -> int:
    return int(texts.astype(str).map(lambda text: bool(regex.search(text))).sum())


def masked_shape_task(task: dict[str, object], words: list[str]) -> tuple[dict[str, object], dict[str, int]]:
    regex = forbidden_regex(words)
    masked = {**task, "data": task["data"].copy()}
    before = count_forbidden(masked["data"]["comments_clean"], regex)
    masked["data"]["comments_clean"] = masked["data"]["comments_clean"].astype(str).map(
        lambda text: regex.sub(" <MASKSHAPE> ", text)
    )
    after = count_forbidden(masked["data"]["comments_clean"], regex)
    assert after == 0, f"Phase 8 echouee: {after} releves contiennent encore un mot interdit"
    return masked, {"before": before, "after": after}


def plot_before_after(path: Path, before: dict[str, object], after: dict[str, object]) -> None:
    labels = ["accuracy", "macro-F1"]
    before_vals = [before["accuracy"], before["macro_f1"]]
    after_vals = [after["accuracy"], after["macro_f1"]]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(7, 4.8))
    plt.bar(x - width / 2, before_vals, width, label="avant")
    plt.bar(x + width / 2, after_vals, width, label="apres masquage")
    plt.xticks(x, labels)
    plt.ylim(0, max(before_vals + after_vals) * 1.25)
    plt.title("Phase 8 - Avant/apres interdiction des mots de forme")
    plt.ylabel("Score test")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def compute_phase8(task: dict[str, object], phase7: dict[str, object]) -> dict[str, object]:
    words = forbidden_shape_words(task["classes"])
    masked_task, counts = masked_shape_task(task, words)
    before = phase7.get("fixed_batch4", phase7["fixed_normal"])
    after_config = phase7_config(
        "group",
        before["config"]["batch_size"],
        before["config"].get("max_train_batches"),
    )
    after = run_conv_experiment(masked_task, after_config)
    before_eval_loader = make_loader(before["data"]["x_test"], before["data"]["y_test_tensor"], 128, False)
    after_eval_loader = make_loader(after["data"]["x_test"], after["data"]["y_test_tensor"], 128, False)
    y_before = predict_classes(before["model"], before_eval_loader)
    y_after = predict_classes(after["model"], after_eval_loader)
    scores_before = class_scores(before["data"]["test_y"], y_before, task["classes"]).rename(
        columns={"precision": "precision_avant", "recall": "recall_avant", "f1": "f1_avant", "support": "support_avant"}
    )
    scores_after = class_scores(after["data"]["test_y"], y_after, task["classes"]).rename(
        columns={"precision": "precision_apres", "recall": "recall_apres", "f1": "f1_apres", "support": "support_apres"}
    )
    scores = scores_before.merge(scores_after, on="classe")
    scores["delta_f1"] = scores["f1_apres"] - scores["f1_avant"]
    scores.to_csv(OUTPUT_DIR / "phase8_class_scores.csv", index=False)
    plot_before_after(OUTPUT_DIR / "phase8_before_after.png", before, after)
    return {
        "words": words,
        "counts": counts,
        "masked_task": masked_task,
        "before": before,
        "after": after,
        "class_scores": scores,
        "most_impacted": scores.sort_values("delta_f1").head(3),
    }


def phase8_reference_from_report(path: Path = Path("RAPPORT.md")) -> dict[str, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    section = text.split("## Phase 8", 1)
    if len(section) < 2:
        return None
    body = section[1].split("## Phase 9", 1)[0]
    match = re.search(r"\|\s*apres interdiction\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", body)
    if not match:
        return None
    return {"accuracy": float(match.group(1)), "macro_f1": float(match.group(2))}


def phase14_splits(masked_task: dict[str, object]) -> dict[str, tuple[list[str], list[int]]]:
    splits = {}
    for split in ["train", "val", "test"]:
        rows = subset(masked_task, split)
        splits[split] = (rows["comments_clean"].astype(str).tolist(), rows["y"].astype(int).tolist())
    return splits


def plot_phase14_score_cost(rows: list[dict[str, object]], path: Path) -> None:
    visible = [row for row in rows if row["regime"] != "phase8_reference"]
    plt.figure(figsize=(7, 4.8))
    for row in visible:
        plt.scatter(row["trainable_parameters"], row["macro_f1"], s=70)
        plt.annotate(str(row["regime"]), (row["trainable_parameters"], row["macro_f1"]), xytext=(6, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Parametres entrainables")
    plt.ylabel("Macro-F1 test")
    plt.title("Phase 14 - Score / cout")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def append_report_phase14(result: dict[str, object]) -> None:
    rows = result["rows"]

    def fmt(value: object, suffix: str = "") -> str:
        if value == "NA" or pd.isna(value):
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}{suffix}" if suffix == "" else f"{value:.1f}{suffix}"
        return str(value)

    table = "\n".join(
        "| {regime} | {macro_f1:.4f} | {accuracy:.4f} | {trainable_parameters} | {training_step_ms:.1f} ms | {peak_memory_mib:.1f} MiB | {saved_artifact_mib:.2f} MiB |".format(**row)
        for row in rows
        if row["regime"] != "phase8_reference"
    )
    reference = result["phase8_reference"]
    lora = result["lora_status"]
    section = f"""

## Phase 14 — Le cerveau emprunté, et sa facture

### Reference
Phase 8, meme split, vocabulaire formes interdit. Reference historique : accuracy={reference['accuracy']:.4f}, macro-F1={reference['macro_f1']:.4f}.

### Modele emprunte
`{result['model']['model_name']}` est retenu car il est tres petit : {result['model']['total_parameters']} parametres encodeur, {result['model']['num_hidden_layers']} couches, {result['model']['hidden_size']} dimensions cachees. Longueur retenue : {result['max_length']}, avec padding dynamique.

### Regime 1 — gele
L'encodeur BERT ne bouge pas ; seule la tete lineaire de classification est entrainee.

### Regime 2 — fine-tuning partiel
Les embeddings et la premiere couche restent geles ; seule la derniere couche Transformer et la tete bougent. La derniere couche utilise lr={result['partial_lrs']['last_layer_lr']}, la tete lr={result['partial_lrs']['head_lr']} pour adapter plus vite la sortie que l'entree.

### Regime 3 — LoRA
{lora}

| Regime | Macro-F1 | Accuracy | Parametres modifies | Step | Memoire | Sauvegarde |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Le Bureau peut se payer {result['recommendation']['regime']} parce qu'il combine score et cout : {result['recommendation']['reason']}.

Fichiers : `outputs/phase14_regimes.csv`, `outputs/phase14_score_cost.png`.
"""
    report = Path("RAPPORT.md")
    text = report.read_text(encoding="utf-8", errors="replace") if report.exists() else ""
    marker = "\n## Phase 14 "
    if marker in text:
        text = text.split(marker)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    report.write_text(text + "\n", encoding="utf-8")


def compute_phase14(task: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    words = forbidden_shape_words(task["classes"])
    masked_task, counts = masked_shape_task(task, words)
    assert len(masked_task["data"]) == task["audit"]["kept"]
    assert len(task["classes"]) == 19
    assert np.array_equal(task["train_idx"], masked_task["train_idx"])
    assert np.array_equal(task["val_idx"], masked_task["val_idx"])
    assert np.array_equal(task["test_idx"], masked_task["test_idx"])
    assert counts["after"] == 0

    phase8_reference = phase8_reference_from_report()
    if phase8_reference is None:
        raise RuntimeError("Reference Phase 8 introuvable dans RAPPORT.md; Phase 14 ne doit pas reentrainer Phase 8.")

    meta = model_metadata(MODEL_NAME)
    tokenizer = meta.pop("tokenizer")
    splits = phase14_splits(masked_task)
    all_texts = [text for split in splits.values() for text in split[0]]
    length_stats = tokenizer_audit(all_texts, tokenizer)
    max_length = choose_max_length(length_stats)
    loaders = make_pretrained_loaders(splits, tokenizer, max_length=max_length, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path("checkpoints") / "phase14"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    measured = []
    for regime in ["frozen", "partial"]:
        measured.append(run_regime(regime, loaders, device, len(task["classes"]), checkpoint_root))

    lora_status = "LoRA non realise : dependance PEFT indisponible ou echec technique."
    try:
        measured.append(run_regime("lora", loaders, device, len(task["classes"]), checkpoint_root))
        lora_result = measured[-1]
        lora_status = (
            "LoRA realise : r={r}, alpha={alpha}, dropout={dropout}, modules={target_modules}.".format(
                **lora_result.details
            )
        )
    except Exception as exc:
        lora_status = f"LoRA non realise : {type(exc).__name__}: {exc}"

    rows: list[dict[str, object]] = [
        {
            "regime": "phase8_reference",
            "model_name": "reseau_maison_phase8",
            "accuracy": phase8_reference["accuracy"],
            "macro_f1": phase8_reference["macro_f1"],
            "total_parameters": "NA",
            "trainable_parameters": "NA",
            "trainable_percent": "NA",
            "training_step_ms": "NA",
            "training_total_s": "NA",
            "peak_memory_mib": "NA",
            "saved_artifact_mib": "NA",
        }
    ]
    rows.extend([r.__dict__ | {"details": json.dumps(r.details, sort_keys=True)} for r in measured])
    csv_rows = [{k: v for k, v in row.items() if k != "details"} for row in rows]
    pd.DataFrame(csv_rows).to_csv(OUTPUT_DIR / "phase14_regimes.csv", index=False)
    plot_phase14_score_cost(csv_rows, OUTPUT_DIR / "phase14_score_cost.png")

    candidates = sorted(
        measured,
        key=lambda r: (r.macro_f1 - 0.000002 * r.trainable_parameters - 0.0005 * r.saved_artifact_mib),
        reverse=True,
    )
    chosen = candidates[0]
    recommendation = {
        "regime": chosen.regime,
        "reason": (
            f"il donne macro-F1={chosen.macro_f1:.4f} avec {chosen.trainable_parameters} parametres modifies "
            f"et {chosen.saved_artifact_mib:.2f} MiB a sauvegarder"
        ),
    }
    result = {
        "dataset_rows": len(masked_task["data"]),
        "classes": task["classes"],
        "split": {"train": len(task["train_idx"]), "val": len(task["val_idx"]), "test": len(task["test_idx"])},
        "remaining_forbidden": counts["after"],
        "model": meta,
        "device": str(device),
        "length_stats": length_stats,
        "max_length": max_length,
        "phase8_reference": phase8_reference,
        "regimes": measured,
        "rows": csv_rows,
        "lora_status": lora_status,
        "partial_lrs": next(r.details["learning_rates"] for r in measured if r.regime == "partial"),
        "recommendation": recommendation,
        "elapsed_s": time.perf_counter() - start,
    }
    append_report_phase14(result)
    return result


def explain_tokens(model: torch.nn.Module, vocab: dict[str, int], text: str, max_len: int, pred_class: int) -> pd.DataFrame:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(token, 1) for token in tokens]
    x = torch.tensor([ids + [0] * (max_len - len(ids))], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        full_score = torch.softmax(model(x), dim=1)[0, pred_class].item()
        rows = []
        for pos, token in enumerate(tokens):
            occluded = x.clone()
            occluded[0, pos] = 0
            score = torch.softmax(model(occluded), dim=1)[0, pred_class].item()
            rows.append({"token": token, "position": pos, "contribution": full_score - score})
    return pd.DataFrame(rows)


def plot_explanation(path: Path, title: str, contributions: pd.DataFrame) -> None:
    visible = contributions.reindex(contributions["contribution"].abs().sort_values(ascending=False).index).head(25)
    visible = visible.sort_values("contribution")
    colors = ["#b91c1c" if value < 0 else "#047857" for value in visible["contribution"]]
    labels = [f"{row.token} ({row.position})" for row in visible.itertuples()]
    plt.figure(figsize=(8, max(4.8, 0.28 * len(visible))))
    plt.barh(labels, visible["contribution"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Baisse de probabilite quand le token est masque")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def case_comment(contrib: pd.DataFrame, actual: str, predicted: str) -> list[str]:
    top_pos = contrib.sort_values("contribution", ascending=False).head(4)["token"].tolist()
    top_neg = contrib.sort_values("contribution", ascending=True).head(3)["token"].tolist()
    return [
        f"Le modele retient surtout {', '.join(top_pos)} pour choisir `{predicted}`.",
        f"Il laisse peu peser {', '.join(top_neg)}, alors qu'un humain lirait aussi le contexte complet.",
        f"Ce cas montre que le dataset associe des mots descriptifs courts a `{actual}`/`{predicted}`, pas une comprehension robuste de la scene.",
    ]


def compute_phase9(task: dict[str, object], phase8: dict[str, object]) -> dict[str, object]:
    final = phase8["after"]
    test_loader = make_loader(final["data"]["x_test"], final["data"]["y_test_tensor"], 128, False)
    logits = predict_logits(final["model"], test_loader)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = final["data"]["test_y"]
    y_pred = probs.argmax(axis=1)
    margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    test_rows_original = subset(task, "test").reset_index(drop=True)
    test_rows_masked = subset(phase8["masked_task"], "test").reset_index(drop=True)

    choices = {
        "correct": int(np.where(y_pred == y_true)[0][0]),
        "wrong": int(np.where(y_pred != y_true)[0][0]),
        "hesitant": int(np.argsort(margins)[0]),
    }
    paths = {
        "correct": OUTPUT_DIR / "phase9_correct.png",
        "wrong": OUTPUT_DIR / "phase9_wrong.png",
        "hesitant": OUTPUT_DIR / "phase9_hesitant.png",
    }
    cases = {}
    for name, pos in choices.items():
        top2 = probs[pos].argsort()[-2:][::-1]
        contrib = explain_tokens(
            final["model"],
            final["data"]["vocab"],
            test_rows_masked.loc[pos, "comments_clean"],
            final["config"]["max_len"],
            int(y_pred[pos]),
        )
        actual = task["classes"][int(y_true[pos])]
        predicted = task["classes"][int(y_pred[pos])]
        plot_explanation(paths[name], f"Phase 9 - {name}: {predicted}", contrib)
        row = test_rows_original.loc[pos]
        cases[name] = {
            "source_index": int(row["row_id"]),
            "datetime": str(row["datetime"]),
            "city": str(row["city"]),
            "actual": actual,
            "predicted": predicted,
            "top2": [(task["classes"][int(i)], float(probs[pos, i])) for i in top2],
            "margin": float(margins[pos]),
            "comments": str(row["comments_clean"]),
            "masked_comments": str(test_rows_masked.loc[pos, "comments_clean"]),
            "contributions": contrib,
            "top_words": contrib.sort_values("contribution", ascending=False).head(5)["token"].tolist(),
            "comment": case_comment(contrib, actual, predicted),
            "path": paths[name],
        }
    return {"cases": cases, "method": "occlusion leave-one-token-out sur la probabilite de la classe predite"}


PRONOUNS = {"it", "they", "he", "she", "this", "that", "them", "its"}


def choose_phase10_record(df: pd.DataFrame) -> dict[str, object]:
    candidates = []
    for idx, row in df.iterrows():
        comment = str(row["comments"]).strip()
        tokens = tokenize(comment)
        if (
            comment
            and 8 <= len(tokens) <= 32
            and any(token in PRONOUNS for token in tokens)
            and str(row["shape"]).strip()
        ):
            candidates.append((idx, row, tokens))
            break
    if not candidates:
        raise RuntimeError("Aucun releve avec commentaire, longueur raisonnable et pronom simple.")
    idx, row, tokens = candidates[0]
    pronoun_index = next(i for i, token in enumerate(tokens) if token in PRONOUNS)
    return {
        "source_index": int(idx),
        "datetime": str(row["datetime"]),
        "city": str(row["city"]),
        "shape": str(row["shape"]),
        "comments": str(row["comments"]).strip(),
        "tokens": tokens,
        "pronoun_index": pronoun_index,
        "pronoun": tokens[pronoun_index],
    }


def build_phase10_embeddings(tokens: list[str], embedding_dim: int = 32) -> dict[str, object]:
    torch.manual_seed(1010)
    vocab = {token: i for i, token in enumerate(dict.fromkeys(tokens))}
    token_ids = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)
    embedding = torch.nn.Embedding(len(vocab), embedding_dim)
    x = embedding(token_ids)
    return {"vocab": vocab, "token_ids": token_ids, "embedding": embedding, "x": x}


def run_manual_attention(x: torch.Tensor, head: SingleHeadSelfAttention) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return head(x)


def assert_attention_shapes(result: dict[str, torch.Tensor], x: torch.Tensor, tol: float = 1e-6) -> dict[str, object]:
    seq_len = x.shape[0]
    weights = result["weights"]
    output = result["output"]
    assert tuple(weights.shape) == (seq_len, seq_len), f"weights shape incorrecte: {tuple(weights.shape)}"
    assert tuple(output.shape) == tuple(x.shape), f"output shape incompatible: {tuple(output.shape)} vs {tuple(x.shape)}"
    row_sums = weights.sum(dim=-1)
    max_error = torch.max(torch.abs(row_sums - 1.0)).item()
    assert max_error < tol, f"Sommes des lignes hors tolerance: {max_error}"
    return {
        "row_sum_min": float(row_sums.min().item()),
        "row_sum_max": float(row_sums.max().item()),
        "row_sum_max_error": float(max_error),
    }


def plot_attention_matrix(path: Path, weights: torch.Tensor, tokens: list[str], title: str) -> None:
    fig_width = max(7.0, min(14.0, 0.48 * len(tokens)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.78))
    im = ax.imshow(weights.detach().numpy(), cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Tokens consultes")
    ax.set_ylabel("Tokens qui posent la question")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def deterministic_permutation(tokens: list[str]) -> torch.Tensor:
    order = sorted(range(len(tokens)), key=lambda i: (len(tokens[i]), tokens[i], i))
    if order == list(range(len(tokens))):
        order = list(reversed(order))
    return torch.tensor(order, dtype=torch.long)


def compute_phase10(df: pd.DataFrame) -> dict[str, object]:
    record = choose_phase10_record(df)
    embedding_dim = 32
    emb = build_phase10_embeddings(record["tokens"], embedding_dim)
    torch.manual_seed(1011)
    head = SingleHeadSelfAttention(embedding_dim)
    attention = run_manual_attention(emb["x"], head)
    checks = assert_attention_shapes(attention, emb["x"])
    plot_attention_matrix(
        OUTPUT_DIR / "phase10_attention_matrix.png",
        attention["weights"],
        record["tokens"],
        "Phase 10 - Matrice d'attention mono-tete manuelle",
    )
    pronoun_index = int(record["pronoun_index"])
    pronoun_weights = attention["weights"][pronoun_index]
    top_index = int(torch.argmax(pronoun_weights).item())
    return {
        "record": record,
        "embedding_dim": embedding_dim,
        "x": emb["x"],
        "head": head,
        "attention": attention,
        "checks": checks,
        "pronoun_weights": pronoun_weights.detach().numpy().tolist(),
        "pronoun_top_index": top_index,
        "pronoun_top_token": record["tokens"][top_index],
        "pronoun_top_weight": float(pronoun_weights[top_index].item()),
        "figure": OUTPUT_DIR / "phase10_attention_matrix.png",
    }


def compute_phase11(phase10: dict[str, object]) -> dict[str, object]:
    tokens = phase10["record"]["tokens"]
    x = phase10["x"]
    head = phase10["head"]
    perm = deterministic_permutation(tokens)
    inverse = torch.argsort(perm)

    original = run_manual_attention(x, head)
    permuted = run_manual_attention(x[perm], head)
    before_diff = torch.max(torch.abs(permuted["output"][inverse] - original["output"])).item()

    pe = positional_encoding(x.shape[0], x.shape[1], x.device)
    original_pos = run_manual_attention(x + pe, head)
    permuted_pos = run_manual_attention(x[perm] + pe, head)
    after_diff = torch.max(torch.abs(permuted_pos["output"][inverse] - original_pos["output"])).item()

    assert before_diff < 1e-6, f"Equivariance sans position non verifiee: {before_diff}"
    assert after_diff > max(1e-4, before_diff * 1000), f"Effet positionnel trop faible: {after_diff}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, weights, title in [
        (axes[0], original["weights"], "Sans position"),
        (axes[1], original_pos["weights"], "Avec position sinusoidale"),
    ]:
        im = ax.imshow(weights.detach().numpy(), cmap="viridis", vmin=0.0)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=70, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Tokens consultes")
    axes[0].set_ylabel("Tokens qui posent la question")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    fig.suptitle("Phase 11 - Attention avant/apres encodage positionnel")
    fig.savefig(OUTPUT_DIR / "phase11_position_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "permutation": perm.tolist(),
        "permuted_tokens": [tokens[i] for i in perm.tolist()],
        "before_diff": float(before_diff),
        "after_diff": float(after_diff),
        "encoding": "sinusoidal deterministe",
        "figure": OUTPUT_DIR / "phase11_position_comparison.png",
    }


def append_report_phases_10_to_11(phase10: dict[str, object], phase11: dict[str, object]) -> None:
    record = phase10["record"]
    shapes = phase10["attention"]
    checks = phase10["checks"]
    section = f"""

## Phase 10 — Chaque mot interroge les autres

Une tete d'attention manuelle projette chaque embedding en `Q`, `K` et `V`. `Q` represente la question posee par un token, `K` l'etiquette comparee chez les autres tokens, et `V` le contenu melange. Les scores sont calcules explicitement par `Q @ K.T / sqrt(d_k)`, puis `softmax` transforme chaque ligne en poids qui somment a 1. La sortie est le melange `weights @ V`.

- index original : {record['source_index']}
- datetime : {record['datetime']}
- city : {record['city']}
- shape : {record['shape']}
- commentaire : `{record['comments']}`
- tokens : `{record['tokens']}`
- formes : X={tuple(phase10['x'].shape)}, Q={tuple(shapes['q'].shape)}, K={tuple(shapes['k'].shape)}, V={tuple(shapes['v'].shape)}, weights={tuple(shapes['weights'].shape)}, output={tuple(shapes['output'].shape)}
- preuve lignes = 1 : min={checks['row_sum_min']:.8f}, max={checks['row_sum_max']:.8f}, erreur max={checks['row_sum_max_error']:.8g}
- figure : `outputs/phase10_attention_matrix.png`
- case pronom : ligne {record['pronoun_index']} (`{record['pronoun']}`), colonne {phase10['pronoun_top_index']} (`{phase10['pronoun_top_token']}`), poids={phase10['pronoun_top_weight']:.6f}

Ces poids viennent d'un mecanisme non entraine : ils montrent comment lire la matrice, pas une comprehension de la coreference.

## Phase 11 — Le Conseil mélange vos mots

1. L'attention seule compare les contenus mais ne contient aucune information d'ordre.
2. Permuter les tokens permute les sorties de la meme facon : apres realignement, l'ecart est ~0.
3. L'encodage positionnel est ajoute aux embeddings avant `Q/K/V` ; la position influence donc questions, cles et valeurs.

- tokens originaux : `{record['tokens']}`
- tokens permutes : `{phase11['permuted_tokens']}`
- permutation : {phase11['permutation']}
- ecart avant position : {phase11['before_diff']:.10g}
- ecart apres position : {phase11['after_diff']:.10g}
- encodage : {phase11['encoding']}
- figure : `outputs/phase11_position_comparison.png`
"""
    path = Path("RAPPORT.md")
    text = path.read_text(encoding="utf-8")
    marker = "\n## Phase 10 — Chaque mot interroge les autres"
    text = text.split(marker)[0].rstrip() + section
    path.write_text(text, encoding="utf-8")


def print_phase10_to_11(phase10: dict[str, object], phase11: dict[str, object], elapsed: float) -> None:
    record = phase10["record"]
    attention = phase10["attention"]
    checks = phase10["checks"]
    print("\n=== PHASE 10 - CHAQUE MOT INTERROGE LES AUTRES ===")
    print(f"index={record['source_index']} | datetime={record['datetime']} | city={record['city']} | shape={record['shape']}")
    print(record["comments"])
    print(f"tokens={record['tokens']}")
    print(f"seq_len={len(record['tokens'])} | embedding_dim={phase10['embedding_dim']} | X={tuple(phase10['x'].shape)}")
    print(
        f"Q={tuple(attention['q'].shape)} | K={tuple(attention['k'].shape)} | "
        f"V={tuple(attention['v'].shape)} | weights={tuple(attention['weights'].shape)} | "
        f"output={tuple(attention['output'].shape)}"
    )
    print(
        f"sommes lignes min={checks['row_sum_min']:.8f} | "
        f"max={checks['row_sum_max']:.8f} | erreur max={checks['row_sum_max_error']:.10g}"
    )
    print(f"pronom index={record['pronoun_index']} | token={record['pronoun']}")
    print(f"poids ligne pronom={phase10['pronoun_weights']}")
    print(f"top consulte={phase10['pronoun_top_token']} | poids={phase10['pronoun_top_weight']:.6f}")

    print("\n=== PHASE 11 - LE CONSEIL MELANGE VOS MOTS ===")
    print(f"tokens originaux={record['tokens']}")
    print(f"tokens permutes={phase11['permuted_tokens']}")
    print(f"permutation={phase11['permutation']}")
    print(f"ecart avant position={phase11['before_diff']:.10g}")
    print(f"ecart apres position={phase11['after_diff']:.10g}")
    print(f"encodage={phase11['encoding']}")
    print(f"temps phase 10-11={elapsed:.2f}s")


def run_phase10_to_11(df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], float]:
    start = time.perf_counter()
    phase10 = compute_phase10(df)
    phase11 = compute_phase11(phase10)
    append_report_phases_10_to_11(phase10, phase11)
    elapsed = time.perf_counter() - start
    return phase10, phase11, elapsed


def compute_phase12() -> dict[str, object]:
    device = torch.device("cpu")
    embedding_dim = 32
    lengths = [16, 32, 64, 128, 256, 512]
    torch.manual_seed(1212)
    head = SingleHeadSelfAttention(embedding_dim).to(device).eval()
    rows = []

    for seq_len in lengths:
        generator = torch.Generator(device=device).manual_seed(1200 + seq_len)
        x = torch.randn(seq_len, embedding_dim, generator=generator, device=device)
        with torch.no_grad():
            for _ in range(5):
                result = head(x)
            assert tuple(result["weights"].shape) == (seq_len, seq_len)

            timings = []
            for _ in range(30):
                start = time.perf_counter()
                result = head(x)
                timings.append((time.perf_counter() - start) * 1000)

        cells = seq_len * seq_len
        rows.append(
            {
                "seq_len": seq_len,
                "median_time_ms": float(np.median(timings)),
                "attention_cells": cells,
                "attention_memory_mib": cells * 4 / (1024**2),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "phase12_attention_scaling.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(table["seq_len"], table["median_time_ms"], marker="o", label="temps median")
    ax1.set_xlabel("Longueur de sequence n")
    ax1.set_ylabel("Temps median attention (ms)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(table["seq_len"], table["attention_cells"], color="tab:orange", linestyle="--", label="n^2")
    ax2.set_ylabel("Coefficients attention n^2")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase12_attention_scaling.png", dpi=170)
    plt.close(fig)

    time_ratios = [
        {
            "from": int(table.loc[i - 1, "seq_len"]),
            "to": int(table.loc[i, "seq_len"]),
            "time_ratio": float(table.loc[i, "median_time_ms"] / table.loc[i - 1, "median_time_ms"]),
            "cell_ratio": float(table.loc[i, "attention_cells"] / table.loc[i - 1, "attention_cells"]),
        }
        for i in range(1, len(table))
    ]
    alpha = float(np.polyfit(np.log(table["seq_len"]), np.log(table["median_time_ms"]), 1)[0])
    return {
        "device": str(device),
        "embedding_dim": embedding_dim,
        "lengths": lengths,
        "table": table,
        "ratios": time_ratios,
        "alpha": alpha,
        "figure": OUTPUT_DIR / "phase12_attention_scaling.png",
        "csv": OUTPUT_DIR / "phase12_attention_scaling.csv",
    }


def assert_two_head_attention(result: dict[str, object], x: torch.Tensor) -> dict[str, object]:
    seq_len = x.shape[0]
    checks = {}
    for name in ["head1", "head2"]:
        head = result[name]
        assert isinstance(head, dict)
        weights = head["weights"]
        output = head["output"]
        assert tuple(weights.shape) == (seq_len, seq_len), f"{name} weights incorrect: {tuple(weights.shape)}"
        assert tuple(output.shape) == (seq_len, x.shape[1] // 2), f"{name} output incorrect: {tuple(output.shape)}"
        row_sums = weights.sum(dim=-1)
        max_error = torch.max(torch.abs(row_sums - 1.0)).item()
        assert max_error < 1e-6, f"{name} lignes hors tolerance: {max_error}"
        checks[name] = float(max_error)
    assert tuple(result["concat"].shape) == tuple(x.shape)
    assert tuple(result["output"].shape) == tuple(x.shape)
    return checks


def compute_phase13(df: pd.DataFrame) -> dict[str, object]:
    record = choose_phase10_record(df)
    d_model = 32
    emb = build_phase10_embeddings(record["tokens"], d_model)
    torch.manual_seed(1313)
    module = TwoHeadSelfAttention(d_model).eval()
    with torch.no_grad():
        result = module(emb["x"])
    checks = assert_two_head_attention(result, emb["x"])
    head1 = result["head1"]
    head2 = result["head2"]
    weights1 = head1["weights"]
    weights2 = head2["weights"]
    mean_abs_diff = torch.mean(torch.abs(weights1 - weights2)).item()
    cosine = torch.nn.functional.cosine_similarity(weights1.flatten(), weights2.flatten(), dim=0).item()

    pronoun_index = int(record["pronoun_index"])
    top1 = int(torch.argmax(weights1[pronoun_index]).item())
    top2 = int(torch.argmax(weights2[pronoun_index]).item())

    vmax = max(float(weights1.max().item()), float(weights2.max().item()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, weights, title in [
        (axes[0], weights1, "Tete 1"),
        (axes[1], weights2, "Tete 2"),
    ]:
        im = ax.imshow(weights.detach().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(record["tokens"])))
        ax.set_yticks(range(len(record["tokens"])))
        ax.set_xticklabels(record["tokens"], rotation=70, ha="right", fontsize=7)
        ax.set_yticklabels(record["tokens"], fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Tokens consultes")
    axes[0].set_ylabel("Tokens qui posent la question")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    fig.suptitle("Phase 13 - Deux tetes d'attention manuelles")
    fig.savefig(OUTPUT_DIR / "phase13_two_heads.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "record": record,
        "x": emb["x"],
        "module": module,
        "result": result,
        "checks": checks,
        "d_model": d_model,
        "num_heads": 2,
        "head_dim": module.head_dim,
        "mean_abs_diff": float(mean_abs_diff),
        "cosine": float(cosine),
        "pronoun_index": pronoun_index,
        "top1_index": top1,
        "top1_token": record["tokens"][top1],
        "top1_weight": float(weights1[pronoun_index, top1].item()),
        "top2_index": top2,
        "top2_token": record["tokens"][top2],
        "top2_weight": float(weights2[pronoun_index, top2].item()),
        "figure": OUTPUT_DIR / "phase13_two_heads.png",
    }


def append_report_phases_12_to_13(phase12: dict[str, object], phase13: dict[str, object]) -> None:
    table_rows = "\n".join(
        f"| {int(row.seq_len)} | {row.median_time_ms:.6f} | {int(row.attention_cells)} | {row.attention_memory_mib:.6f} |"
        for row in phase12["table"].itertuples()
    )
    ratio_rows = "\n".join(
        f"| {r['from']} -> {r['to']} | {r['time_ratio']:.3f} | {r['cell_ratio']:.1f} |"
        for r in phase12["ratios"]
    )
    record = phase13["record"]
    result = phase13["result"]
    section = f"""

## Phase 12 — Le prix des regards

Chaque token compare sa requete aux cles de tous les tokens. La matrice `weights` possede donc `n x n` coefficients : sa memoire croit en `O(n^2)`, et le produit `QK^T` augmente fortement avec la longueur. Le temps reel ne suit pas obligatoirement un facteur exact x4 a chaque doublement, car les petits tenseurs subissent l'overhead Python, le cache et la vectorisation.

| seq_len | temps median ms | cellules attention | memoire MiB |
|---:|---:|---:|---:|
{table_rows}

| doublement | ratio temps observe | ratio cellules |
|---:|---:|---:|
{ratio_rows}

Diagnostic log-log `log(time) ~ alpha * log(n)` : alpha={phase12['alpha']:.3f}. Figure : `outputs/phase12_attention_scaling.png`. Donnees : `outputs/phase12_attention_scaling.csv`.

## Phase 13 — Deux paires d'yeux

Une tete correspond a une famille de projections `Q/K/V`. Ici, deux tetes manuelles possedent deux familles independantes et travaillent chacune dans un sous-espace de dimension 16. Leurs sorties `[n,16]` sont concatenees en `[n,32]`, puis `Wo` les reprojette vers `d_model=32`.

- vrai releve : index {record['source_index']}, commentaire `{record['comments']}`
- tokens : `{record['tokens']}`
- dimensions : X={tuple(phase13['x'].shape)}, head1 output={tuple(result['head1']['output'].shape)}, head2 output={tuple(result['head2']['output'].shape)}, concat={tuple(result['concat'].shape)}, final={tuple(result['output'].shape)}
- weights tete 1={tuple(result['head1']['weights'].shape)}, weights tete 2={tuple(result['head2']['weights'].shape)}
- preuve lignes = 1 : erreur max tete 1={phase13['checks']['head1']:.8g}, erreur max tete 2={phase13['checks']['head2']:.8g}
- difference moyenne absolue weights1/weights2 : {phase13['mean_abs_diff']:.8f}
- similarite cosinus aplatie : {phase13['cosine']:.8f}
- pronom : `{record['pronoun']}` ; tete 1 regarde surtout `{phase13['top1_token']}` poids={phase13['top1_weight']:.6f} ; tete 2 regarde surtout `{phase13['top2_token']}` poids={phase13['top2_weight']:.6f}
- figure : `outputs/phase13_two_heads.png`

Ces poids ne sont pas entraines : les deux projections produisent des patrons d'attention differents, mais on ne peut pas attribuer un role linguistique reel aux tetes.
"""
    path = Path("RAPPORT.md")
    text = path.read_text(encoding="utf-8")
    marker = "\n## Phase 12 — Le prix des regards"
    text = text.split(marker)[0].rstrip() + section
    path.write_text(text, encoding="utf-8")


def print_phase12_to_13(phase12: dict[str, object], phase13: dict[str, object], elapsed: float) -> None:
    result = phase13["result"]
    record = phase13["record"]
    print("\n=== PHASE 12 - LE PRIX DES REGARDS ===")
    print(f"device={phase12['device']} | embedding_dim={phase12['embedding_dim']} | longueurs={phase12['lengths']}")
    print(phase12["table"].to_string(index=False))
    for ratio in phase12["ratios"]:
        print(f"ratio {ratio['from']}->{ratio['to']}: temps={ratio['time_ratio']:.3f} | cellules={ratio['cell_ratio']:.1f}")
    print(f"alpha_log_log={phase12['alpha']:.3f}")

    print("\n=== PHASE 13 - DEUX PAIRES D'YEUX ===")
    print(f"index={record['source_index']} | commentaire={record['comments']}")
    print(f"tokens={record['tokens']}")
    print(f"seq_len={len(record['tokens'])} | d_model={phase13['d_model']} | tetes=2 | head_dim={phase13['head_dim']}")
    print(
        f"X={tuple(phase13['x'].shape)} | out1={tuple(result['head1']['output'].shape)} | "
        f"out2={tuple(result['head2']['output'].shape)} | concat={tuple(result['concat'].shape)} | "
        f"final={tuple(result['output'].shape)}"
    )
    print(f"weights tete1={tuple(result['head1']['weights'].shape)} | weights tete2={tuple(result['head2']['weights'].shape)}")
    print(f"erreur lignes tete1={phase13['checks']['head1']:.10g} | tete2={phase13['checks']['head2']:.10g}")
    print(f"mean_abs_diff={phase13['mean_abs_diff']:.8f} | cosine={phase13['cosine']:.8f}")
    print(
        f"pronom={record['pronoun']} | tete1 top={phase13['top1_token']} ({phase13['top1_weight']:.6f}) | "
        f"tete2 top={phase13['top2_token']} ({phase13['top2_weight']:.6f})"
    )
    print(f"temps phase 12-13={elapsed:.2f}s")


def run_phase12_to_13(df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], float]:
    start = time.perf_counter()
    phase12 = compute_phase12()
    phase13 = compute_phase13(df)
    append_report_phases_12_to_13(phase12, phase13)
    elapsed = time.perf_counter() - start
    return phase12, phase13, elapsed


def append_report(task: dict[str, object], phase3: dict[str, object], phase4: dict[str, object], phase5: dict[str, object]) -> None:
    audit = task["audit"]
    classes = ", ".join(task["classes"])
    p3 = phase3["torch"]
    p5 = phase5["final"]
    exp_rows = "\n".join(
        f"| {e['reglage']} | {e['temps']:.2f}s | {e['facteur_gain']:.2f} | {e['macro_f1']:.4f} | {e['ecart_score']:+.4f} |"
        for e in phase5["experiments"]
    )
    section = f"""

## Phase 3 - Battre le service statistique

Regle finale `shape` : les lignes sans `shape` restent dans le dataset general mais sont exclues de cette tache supervisee. Notre dataset charge compte {audit['missing']} `shape` manquantes : {audit['missing_valid_csv_rows']} dans les lignes CSV directement valides, plus {audit['missing_repaired_rows']} dans les lignes malformees reparees. `unknown` et `other` sont retires car ce sont des categories fourre-tout. Les doublons evidents sont fusionnes : `round` vers `circle`, `changed` vers `changing`. Les 6 classes ayant moins de 10 exemples apres nettoyage sont exclues, car elles ne representent que {audit['low_support_removed']} releves et sont trop petites pour etre reparties proprement entre train, validation et test. Aucun sous-echantillonnage artificiel par classe n'est applique.

- lignes initiales : {audit['initial']}
- shape manquantes : {audit['missing']}
- unknown : {audit['unknown']}
- other : {audit['other']}
- lignes/classes avant filtre <10 : {audit['before_low_support_lines']} / {audit['before_low_support_classes']}
- classes <10 retirees : {audit['low_support_classes']}
- lignes retirees car classe trop rare : {audit['low_support_removed']}
- lignes gardees : {audit['kept']}
- nombre final de classes : {audit['n_classes']}
- classes finales : {classes}
- split : train {len(task['train_idx'])}, validation {len(task['val_idx'])}, test {len(task['test_idx'])}

Le vocabulaire du reseau est construit uniquement sur train. Exemple de passage numerique :

- texte brut : `{p3['example']['text'][:180]}`
- tokens : `{p3['example']['tokens']}`
- ids : `{p3['example']['ids']}`

Modele PyTorch : `Embedding` avec mean pooling masque, puis `Linear`, `ReLU`, `Dropout`, `Linear`. Pas de RNN, pas d'attention, pas de Transformer. La metrique principale est le macro-F1.

| modele | accuracy test | macro-F1 test | temps entrainement |
| --- | ---: | ---: | ---: |
| Majorite | {phase3['majority']['accuracy']:.4f} | {phase3['majority']['macro_f1']:.4f} | {phase3['majority']['train_time']:.2f}s |
| Lineaire | {phase3['linear']['accuracy']:.4f} | {phase3['linear']['macro_f1']:.4f} | {phase3['linear']['train_time']:.2f}s |
| PyTorch | {p3['accuracy']:.4f} | {p3['macro_f1']:.4f} | {p3['train_time']:.2f}s |

Journal d'essais PyTorch : {'; '.join(phase3['attempts'])}.

La courbe train loss / validation loss est enregistree dans `outputs/phase3_train_val_loss.png`.

## Phase 4 - Le carnet de pannes

| panne | geste exact | signature observee | test diagnostic rapide | correction |
| --- | --- | --- | --- | --- |
| 1 | Evaluer avec `model.train()` donc dropout actif | score test instable : {min(phase4['panne1']['scores']):.4f} a {max(phase4['panne1']['scores']):.4f}, alors que le train est {phase4['panne1']['train_f1']:.4f} | repeter deux fois la meme evaluation sans changer les donnees | appeler `model.eval()` avant validation/test |
| 2 | Decaler volontairement le mapping de sortie au decodage | la loss validation descend mais le macro-F1 interprete tombe a {phase4['panne2']['bad_f1']:.4f} | comparer `class_to_id` et `id_to_class` utilises au train et au reporting | conserver un mapping unique et versionne pendant toute l'experience |
| 3 | Mettre un learning rate quasi nul (`1e-7`) | loss train quasi plate : {phase4['panne3']['first_loss']:.4f} vers {phase4['panne3']['last_loss']:.4f} | verifier la norme des gradients et la valeur du learning rate | remettre un learning rate compatible avec Adam (`0.003` ici) |

Figures : `outputs/phase4_panne1.png`, `outputs/phase4_panne2.png`, `outputs/phase4_panne3.png`.

## Phase 5 - Le budget de calcul

| reglage | temps | facteur gain | macro-F1 | ecart de score |
| --- | ---: | ---: | ---: | ---: |
{exp_rows}

- temps Phase 3 : {p3['train_time']:.2f}s
- macro-F1 Phase 3 : {p3['macro_f1']:.4f}
- temps Phase 5 : {p5['train_time']:.2f}s
- macro-F1 Phase 5 : {p5['macro_f1']:.4f}
- facteur final : {p3['train_time'] / p5['train_time']:.2f}

Le modele final rapide combine les reglages retenus puis est reentraine proprement sur le meme split et les memes classes. Aller trop vite peut couter en generalisation : reduire trop le modele, tronquer trop le texte ou arreter trop tot peut enlever de l'information utile et degrade le macro-F1.

La comparaison temporelle est enregistree dans `outputs/phase5_time_comparison.png`. Les experiences sont enregistrees dans `outputs/phase5_experiments.csv`.
"""
    path = Path("RAPPORT.md")
    text = path.read_text(encoding="utf-8")
    marker = "\n## Phase 3 - Battre le service statistique"
    text = text.split(marker)[0].rstrip() + section
    path.write_text(text, encoding="utf-8")


def append_report_phases_6_to_9(
    task: dict[str, object],
    phase6: dict[str, object],
    phase7: dict[str, object],
    phase8: dict[str, object],
    phase9: dict[str, object],
) -> None:
    rf_table = "\n".join(
        f"| {r['couche']} | {r['kernel']} | {r['dilation']} | {r['stride']} | {r['champ_ajoute']} | {r['champ_cumule']} |"
        for r in phase6["rf_rows"]
    )
    words = ", ".join(phase8["words"])
    impacted = "\n".join(
        f"| {row.classe} | {row.f1_avant:.4f} | {row.f1_apres:.4f} | {row.delta_f1:+.4f} |"
        for row in phase8["most_impacted"].itertuples()
    )
    focus_rows = phase8["class_scores"][phase8["class_scores"]["classe"].isin(["light", "triangle", "circle"])]
    focus = "\n".join(
        f"| {row.classe} | {row.precision_avant:.4f} | {row.recall_avant:.4f} | {row.f1_avant:.4f} | {row.precision_apres:.4f} | {row.recall_apres:.4f} | {row.f1_apres:.4f} |"
        for row in focus_rows.itertuples()
    )
    cases_md = []
    for name, case in phase9["cases"].items():
        top2 = ", ".join(f"{label}={prob:.3f}" for label, prob in case["top2"])
        comments = "\n".join(f"- {line}" for line in case["comment"])
        cases_md.append(
            f"""### {name}
- index original : {case['source_index']}
- datetime : {case['datetime']}
- city : {case['city']}
- vraie shape : {case['actual']}
- shape predite : {case['predicted']}
- top2 : {top2}
- marge top1-top2 : {case['margin']:.4f}
- temoignage : `{case['comments']}`
- temoignage masque lu par le modele : `{case['masked_comments']}`
- mots les plus influents : {', '.join(case['top_words'])}
{comments}
- figure : `{case['path']}`
"""
        )
    section = f"""

## Phase 6 — Le champ de vision du modèle

- longueur max avant troncature : {phase6['stats']['max_tokens_before_truncation']} tokens
- longueur mediane : {phase6['stats']['median_tokens']:.1f} tokens
- `max_len` accepte : {phase6['conv']['config']['max_len']} tokens, ce qui couvre {phase6['stats']['pct_covered_by_60'] * 100:.1f} % des textes supervises
- architecture : `Embedding -> projection -> Conv1d dilatees residuelles + BatchNorm -> pooling masque -> MLP`

| couche | kernel | dilation | stride | champ ajoute | champ cumule |
| ---: | ---: | ---: | ---: | ---: | ---: |
{rf_table}

Le champ receptif final vaut {phase6['rf_rows'][-1]['champ_cumule']} tokens, donc il couvre le `max_len` de {phase6['conv']['config']['max_len']}. Avant entrainement, j'ai modifie le premier token d'un vrai releve : `{phase6['probe']['token_original']}` vers `{phase6['probe']['token_modified']}`. L'ecart max absolu des logits vaut {phase6['probe']['logit_max_abs']:.6f}, donc la sortie change deja avec des poids identiques.

| modele | accuracy | macro-F1 | temps |
| --- | ---: | ---: | ---: |
| reference Phase 3/5 | {phase6['reference']['accuracy']:.4f} | {phase6['reference']['macro_f1']:.4f} | {phase6['reference']['train_time']:.2f}s |
| convolution Phase 6 | {phase6['conv']['accuracy']:.4f} | {phase6['conv']['macro_f1']:.4f} | {phase6['conv']['train_time']:.2f}s |

La courbe est `outputs/phase6_train_val_loss.png`, le tableau est `outputs/phase6_receptive_field.csv`.

## Phase 7 — Quatre relevés à la fois

Le modele Phase 6 contenait `BatchNorm1d`. Avec un batch de 4, ses statistiques dependent fortement des trois autres releves places dans le meme lot. J'ai corrige le modele avec `GroupNorm`, qui normalise les canaux de chaque releve independamment des autres exemples. La prediction d'un releve ne doit pas dependre des autres releves du batch.

Pour garder un temps raisonnable, les deux experiences en batch 4 sont plafonnees a {phase7['old_batch4']['config'].get('max_train_batches')} lots par epoque. Le batch utilise par le modele reste bien 4.

| experience | accuracy | macro-F1 | temps |
| --- | ---: | ---: | ---: |
| ancien BatchNorm, batch=4 | {phase7['old_batch4']['accuracy']:.4f} | {phase7['old_batch4']['macro_f1']:.4f} | {phase7['old_batch4']['train_time']:.2f}s |
| corrige GroupNorm, batch=4 | {phase7['fixed_batch4']['accuracy']:.4f} | {phase7['fixed_batch4']['macro_f1']:.4f} | {phase7['fixed_batch4']['train_time']:.2f}s |
| corrige GroupNorm, batch normal | {phase7['fixed_normal']['accuracy']:.4f} | {phase7['fixed_normal']['macro_f1']:.4f} | {phase7['fixed_normal']['train_time']:.2f}s |

Inference batch=1 : {'OUI' if phase7['batch1_ok'] else 'NON'}. Figure : `outputs/phase7_batch4_comparison.png`.

## Phase 8 — Le Conseil a lu trois relevés

Liste interdite construite depuis les classes retenues et les fusions connues : {words}.

Politique : je remplace les mots par `<MASKSHAPE>`. Le token garde l'information qu'un mot interdit etait present, mais interdit la recopie directe du nom de classe. Le remplacement utilise des bornes de mots, donc `light` ne coupe pas un mot plus long.

- releves avec mot interdit avant traitement : {phase8['counts']['before']}
- releves avec mot interdit apres traitement : {phase8['counts']['after']}

| modele | accuracy | macro-F1 |
| --- | ---: | ---: |
| avant interdiction | {phase8['before']['accuracy']:.4f} | {phase8['before']['macro_f1']:.4f} |
| apres interdiction | {phase8['after']['accuracy']:.4f} | {phase8['after']['macro_f1']:.4f} |

Classes chutant le plus :

| classe | F1 avant | F1 apres | delta |
| --- | ---: | ---: | ---: |
{impacted}

Comparaison demandee :

| classe | precision avant | recall avant | F1 avant | precision apres | recall apres | F1 apres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{focus}

Le macro-F1 chute plus directement que l'accuracy quand les petites classes perdent leurs indices lexicaux. L'accuracy reste dominee par les classes frequentes, alors que le macro-F1 donne le meme poids moyen a chaque classe.

Fichiers : `outputs/phase8_class_scores.csv`, `outputs/phase8_before_after.png`.

## Phase 9 — Rendre des comptes sur trois décisions

Methode : {phase9['method']}. Pour chaque token, je masque seulement ce token et je mesure la baisse de probabilite de la classe predite.

{''.join(cases_md)}
"""
    path = Path("RAPPORT.md")
    text = path.read_text(encoding="utf-8")
    marker = "\n## Phase 6 — Le champ de vision du modèle"
    text = text.split(marker)[0].rstrip() + section
    path.write_text(text, encoding="utf-8")


def print_phase3_to_5(task: dict[str, object], phase3: dict[str, object], phase4: dict[str, object], phase5: dict[str, object]) -> None:
    audit = task["audit"]
    print("\n=== PHASE 3 - BATTRE LE SERVICE STATISTIQUE ===")
    print(
        f"Lignes initiales={audit['initial']} | manquantes={audit['missing']} | "
        f"unknown={audit['unknown']} | other={audit['other']} | gardees={audit['kept']} | "
        f"classes={audit['n_classes']}"
    )
    print("Classes finales:", ", ".join(task["classes"]))
    print(f"Split: train={len(task['train_idx'])}, validation={len(task['val_idx'])}, test={len(task['test_idx'])}")
    ex = phase3["torch"]["example"]
    print("Exemple brut -> tokens -> ids:")
    print(str(ex["text"])[:180])
    print(ex["tokens"])
    print(ex["ids"])
    print("Modele | accuracy | macro-F1 | temps")
    print(f"Majorite | {phase3['majority']['accuracy']:.4f} | {phase3['majority']['macro_f1']:.4f} | 0.00s")
    print(f"Lineaire | {phase3['linear']['accuracy']:.4f} | {phase3['linear']['macro_f1']:.4f} | {phase3['linear']['train_time']:.2f}s")
    print(f"PyTorch | {phase3['torch']['accuracy']:.4f} | {phase3['torch']['macro_f1']:.4f} | {phase3['torch']['train_time']:.2f}s")
    print("\n=== PHASE 4 - LE CARNET DE PANNES ===")
    print(f"Panne 1 dropout actif: macro-F1 instable {phase4['panne1']['scores']}")
    print(f"Panne 2 labels decales: macro-F1={phase4['panne2']['bad_f1']:.4f}")
    print(f"Panne 3 lr trop faible: loss {phase4['panne3']['first_loss']:.4f}->{phase4['panne3']['last_loss']:.4f}")
    print("\n=== PHASE 5 - LE BUDGET DE CALCUL ===")
    print(f"Phase 3: {phase3['torch']['train_time']:.2f}s, macro-F1={phase3['torch']['macro_f1']:.4f}")
    print(f"Phase 5: {phase5['final']['train_time']:.2f}s, macro-F1={phase5['final']['macro_f1']:.4f}")
    print(f"Facteur acceleration: {phase3['torch']['train_time'] / phase5['final']['train_time']:.2f}")
    print(pd.DataFrame([{k: v for k, v in e.items() if k != "config"} for e in phase5["experiments"]]).to_string(index=False))


def print_phase6_to_9(phase6: dict[str, object], phase7: dict[str, object], phase8: dict[str, object], phase9: dict[str, object]) -> None:
    print("\n=== PHASE 6 - LE CHAMP DE VISION DU MODELE ===")
    print(
        f"Longueur max={phase6['stats']['max_tokens_before_truncation']} | "
        f"mediane={phase6['stats']['median_tokens']:.1f} | "
        f"max_len={phase6['conv']['config']['max_len']}"
    )
    print(pd.DataFrame(phase6["rf_rows"]).to_string(index=False))
    print(
        f"Token debut: {phase6['probe']['token_original']} -> {phase6['probe']['token_modified']} | "
        f"max diff logits={phase6['probe']['logit_max_abs']:.6f}"
    )
    print(
        f"Phase 6: accuracy={phase6['conv']['accuracy']:.4f} | "
        f"macro-F1={phase6['conv']['macro_f1']:.4f} | temps={phase6['conv']['train_time']:.2f}s"
    )

    print("\n=== PHASE 7 - QUATRE RELEVES A LA FOIS ===")
    print(
        f"Ancien batch=4: accuracy={phase7['old_batch4']['accuracy']:.4f} | "
        f"macro-F1={phase7['old_batch4']['macro_f1']:.4f}"
    )
    print(
        f"Corrige GroupNorm batch=4: accuracy={phase7['fixed_batch4']['accuracy']:.4f} | "
        f"macro-F1={phase7['fixed_batch4']['macro_f1']:.4f}"
    )
    print(
        f"Corrige GroupNorm batch normal: accuracy={phase7['fixed_normal']['accuracy']:.4f} | "
        f"macro-F1={phase7['fixed_normal']['macro_f1']:.4f}"
    )
    print(f"Inference batch=1 reussie: {'OUI' if phase7['batch1_ok'] else 'NON'}")

    print("\n=== PHASE 8 - INTERDIRE LES MOTS DE FORME ===")
    print(f"Liste interdite ({len(phase8['words'])} mots): {', '.join(phase8['words'])}")
    print(
        f"Avant={phase8['counts']['before']} releves avec mot interdit | "
        f"apres={phase8['counts']['after']}"
    )
    print(
        f"Avant: accuracy={phase8['before']['accuracy']:.4f}, macro-F1={phase8['before']['macro_f1']:.4f} | "
        f"Apres: accuracy={phase8['after']['accuracy']:.4f}, macro-F1={phase8['after']['macro_f1']:.4f}"
    )
    print("Classes chutant le plus:")
    print(phase8["most_impacted"][["classe", "f1_avant", "f1_apres", "delta_f1"]].to_string(index=False))

    print("\n=== PHASE 9 - EXPLIQUER TROIS DECISIONS ===")
    for name, case in phase9["cases"].items():
        print(
            f"{name}: index={case['source_index']} | {case['actual']} -> {case['predicted']} | "
            f"top mots={', '.join(case['top_words'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bureau d'analyse terrestre")
    parser.add_argument(
        "--phase",
        choices=["6", "7", "8", "9", "10", "11", "10-11", "12", "13", "12-13", "14"],
        help="executer rapidement une phase ciblee",
    )
    return parser.parse_args()


def run_late_phases(task: dict[str, object], phase3: dict[str, object] | None = None, phase5: dict[str, object] | None = None) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    phase6 = compute_phase6(task, phase3, phase5)
    phase7 = compute_phase7(task, phase6)
    phase8 = compute_phase8(task, phase7)
    phase9 = compute_phase9(task, phase8)
    return phase6, phase7, phase8, phase9


def main() -> None:
    args = parse_args()
    download_dataset()
    raw_df, load_report = load_dataset()
    df = prepare_dates(raw_df)

    if args.phase in {"10", "11", "10-11"}:
        phase10, phase11, elapsed = run_phase10_to_11(df)
        print_phase10_to_11(phase10, phase11, elapsed)
        print(f"\nPhase demandee terminee: {args.phase}")
        return

    if args.phase in {"12", "13", "12-13"}:
        phase12, phase13, elapsed = run_phase12_to_13(df)
        print_phase12_to_13(phase12, phase13, elapsed)
        print(f"\nPhase demandee terminee: {args.phase}")
        return

    shape_task = prepare_shape_task(df, load_report)

    if args.phase == "14":
        phase14 = compute_phase14(shape_task)
        print("\n=== PHASE 14 - LE CERVEAU EMPRUNTE, ET SA FACTURE ===")
        print(f"lignes={phase14['dataset_rows']} | classes={len(phase14['classes'])} | split={phase14['split']}")
        print(f"remaining_forbidden_words={phase14['remaining_forbidden']}")
        print(
            f"modele={phase14['model']['model_name']} | params={phase14['model']['total_parameters']} | "
            f"architecture={phase14['model']['architecture']} layers={phase14['model']['num_hidden_layers']}"
        )
        print(f"device={phase14['device']}")
        stats = phase14["length_stats"]
        print(
            f"longueurs tokenizer mediane={stats['median']:.1f} p95={stats['p95']:.1f} "
            f"p99={stats['p99']:.1f} max={stats['max']} | max_length={phase14['max_length']}"
        )
        print(
            f"Reference Phase 8: accuracy={phase14['phase8_reference']['accuracy']:.4f} | "
            f"macro-F1={phase14['phase8_reference']['macro_f1']:.4f}"
        )
        print(pd.DataFrame(phase14["rows"]).to_string(index=False))
        print(f"Temps Phase 14 ciblee={phase14['elapsed_s']:.2f}s")
        print(f"\nPhase demandee terminee: {args.phase}")
        return

    if args.phase:
        phase6 = compute_phase6(shape_task)
        if args.phase == "6":
            print("\n=== PHASE 6 - LE CHAMP DE VISION DU MODELE ===")
            print(pd.DataFrame(phase6["rf_rows"]).to_string(index=False))
            print(f"accuracy={phase6['conv']['accuracy']:.4f} | macro-F1={phase6['conv']['macro_f1']:.4f}")
            return
        phase7 = compute_phase7(shape_task, phase6)
        if args.phase == "7":
            print("\n=== PHASE 7 - QUATRE RELEVES A LA FOIS ===")
            print(f"ancien batch4 macro-F1={phase7['old_batch4']['macro_f1']:.4f}")
            print(f"corrige batch4 macro-F1={phase7['fixed_batch4']['macro_f1']:.4f}")
            print(f"batch1 OK={'OUI' if phase7['batch1_ok'] else 'NON'}")
            return
        phase8 = compute_phase8(shape_task, phase7)
        if args.phase == "8":
            print("\n=== PHASE 8 - INTERDIRE LES MOTS DE FORME ===")
            print(f"remaining_forbidden={phase8['counts']['after']}")
            print(f"apres macro-F1={phase8['after']['macro_f1']:.4f}")
            return
        phase9 = compute_phase9(shape_task, phase8)
        print_phase6_to_9(phase6, phase7, phase8, phase9)
        print(f"\nPhase demandee terminee: {args.phase}")
        return

    full = compute_phase0(df, None, None)
    selected = compute_phase0(df, 1990, 2014)
    compute_phase1(selected["df"])
    phase2 = compute_phase2(df)
    phase3 = compute_phase3(shape_task)
    phase4 = compute_phase4(shape_task, phase3)
    phase5 = compute_phase5(shape_task, phase3)
    phase6, phase7, phase8, phase9 = run_late_phases(shape_task, phase3, phase5)

    save_outputs(selected)
    append_report(shape_task, phase3, phase4, phase5)
    append_report_phases_6_to_9(shape_task, phase6, phase7, phase8, phase9)
    print_phase0(load_report, full, selected)
    print_phase2(phase2)
    print_phase3_to_5(shape_task, phase3, phase4, phase5)
    print_phase6_to_9(phase6, phase7, phase8, phase9)
    print("\nSorties generees:")
    print(f"- {OUTPUT_DIR / 'phase0_top10_journees.csv'}")
    print(f"- {OUTPUT_DIR / 'phase0_volume_annuel.png'}")
    print(f"- {OUTPUT_DIR / 'phase2_overfit_loss.png'}")
    print(f"- {OUTPUT_DIR / 'phase3_train_val_loss.png'}")
    print(f"- {OUTPUT_DIR / 'phase4_panne1.png'}")
    print(f"- {OUTPUT_DIR / 'phase4_panne2.png'}")
    print(f"- {OUTPUT_DIR / 'phase4_panne3.png'}")
    print(f"- {OUTPUT_DIR / 'phase5_time_comparison.png'}")
    print(f"- {OUTPUT_DIR / 'phase5_experiments.csv'}")
    print(f"- {OUTPUT_DIR / 'phase6_receptive_field.csv'}")
    print(f"- {OUTPUT_DIR / 'phase6_train_val_loss.png'}")
    print(f"- {OUTPUT_DIR / 'phase7_batch4_comparison.png'}")
    print(f"- {OUTPUT_DIR / 'phase8_class_scores.csv'}")
    print(f"- {OUTPUT_DIR / 'phase8_before_after.png'}")
    print(f"- {OUTPUT_DIR / 'phase9_correct.png'}")
    print(f"- {OUTPUT_DIR / 'phase9_wrong.png'}")
    print(f"- {OUTPUT_DIR / 'phase9_hesitant.png'}")


if __name__ == "__main__":
    main()
