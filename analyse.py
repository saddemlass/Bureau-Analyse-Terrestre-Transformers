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
import pandas as pd
import torch


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


def main() -> None:
    download_dataset()
    raw_df, load_report = load_dataset()
    df = prepare_dates(raw_df)
    full = compute_phase0(df, None, None)
    selected = compute_phase0(df, 1990, 2014)
    compute_phase1(selected["df"])
    phase2 = compute_phase2(df)

    save_outputs(selected)
    print_phase0(load_report, full, selected)
    print_phase2(phase2)
    print("\nSorties generees:")
    print(f"- {OUTPUT_DIR / 'phase0_top10_journees.csv'}")
    print(f"- {OUTPUT_DIR / 'phase0_volume_annuel.png'}")
    print(f"- {OUTPUT_DIR / 'phase2_overfit_loss.png'}")


if __name__ == "__main__":
    main()
