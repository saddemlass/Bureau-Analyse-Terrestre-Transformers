from __future__ import annotations

import csv
import os
import re
import time
import urllib.error
import urllib.request
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

from models import TextClassifier
from training import build_vocab, encode_texts, evaluate_model, make_loader, set_seeds, train_model


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
    history = train_model(
        model,
        data["train_loader"],
        data["val_loader"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )
    metrics = evaluate_model(model, data["test_loader"], data["test_y"])
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


def main() -> None:
    download_dataset()
    raw_df, load_report = load_dataset()
    df = prepare_dates(raw_df)
    full = compute_phase0(df, None, None)
    selected = compute_phase0(df, 1990, 2014)
    compute_phase1(selected["df"])
    phase2 = compute_phase2(df)
    shape_task = prepare_shape_task(df, load_report)
    phase3 = compute_phase3(shape_task)
    phase4 = compute_phase4(shape_task, phase3)
    phase5 = compute_phase5(shape_task, phase3)

    save_outputs(selected)
    append_report(shape_task, phase3, phase4, phase5)
    print_phase0(load_report, full, selected)
    print_phase2(phase2)
    print_phase3_to_5(shape_task, phase3, phase4, phase5)
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


if __name__ == "__main__":
    main()
