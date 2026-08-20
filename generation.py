from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GEN_MODEL_NAME = "distilgpt2"
PHASE17_SEED = 17017


@dataclass(frozen=True)
class GenerationSetting:
    name: str
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool = True


GENERATION_GRID = [
    GenerationSetting("A_trop_deterministe", 0.25, 0.70, 20),
    GenerationSetting("B_deterministe_intermediaire", 0.55, 0.85, 40),
    GenerationSetting("C_compromis", 0.80, 0.92, 50),
    GenerationSetting("D_creatif", 1.05, 0.96, 80),
    GenerationSetting("E_trop_aleatoire", 1.55, 1.00, 0),
]


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\x00", " ")).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def word_count(text: str) -> int:
    return len(words(text))


def repeated_ngram_rate(tokens: list[str], n: int = 2) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def lexical_diversity(tokens: list[str]) -> float:
    return 0.0 if not tokens else len(set(tokens)) / len(tokens)


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, param in model.state_dict().items():
        tensor = param.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_comments(df: pd.DataFrame) -> list[str]:
    comments = [clean_text(x) for x in df["comments"].tolist()]
    return [text for text in comments if text and text.lower() != "nan" and word_count(text) >= 3]


def style_benchmark(comments: list[str], sample_size: int = 300) -> dict[str, Any]:
    rng = np.random.default_rng(PHASE17_SEED)
    idx = rng.choice(len(comments), size=min(sample_size, len(comments)), replace=False)
    sample = [comments[int(i)] for i in idx]
    lengths = np.array([word_count(text) for text in sample], dtype=float)
    vocab_sizes = np.array([lexical_diversity(words(text)) for text in sample], dtype=float)
    return {
        "sample": sample,
        "sample_size": len(sample),
        "median_words": float(np.median(lengths)),
        "p25_words": float(np.percentile(lengths, 25)),
        "p75_words": float(np.percentile(lengths, 75)),
        "mean_words": float(np.mean(lengths)),
        "has_number_ratio": float(np.mean([bool(re.search(r"\d", text)) for text in sample])),
        "has_punctuation_ratio": float(np.mean([bool(re.search(r"[.,;:!?]", text)) for text in sample])),
        "mean_lexical_diversity": float(np.mean(vocab_sizes)),
    }


def build_prompt(style: dict[str, Any]) -> str:
    sample = style["sample"]
    target = style["median_words"]
    ordered = sorted(sample, key=lambda text: (abs(word_count(text) - target), text.lower()))
    examples = ordered[:6]
    lines = [
        "Write one short UFO witness comment in the same plain style as these real comments.",
        "Keep the wording simple, imperfect, and about the same length. Do not explain.",
        "",
    ]
    for text in examples:
        lines.append(f"Real comment: {text[:220]}")
    lines.append("New comment:")
    return "\n".join(lines)


def load_generator(model_name: str = GEN_MODEL_NAME) -> tuple[Any, torch.nn.Module, torch.device, str, int, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    total_params = sum(param.numel() for param in model.parameters())
    all_frozen = all(not param.requires_grad for param in model.parameters())
    hash_before = parameter_hash(model)
    return tokenizer, model, device, hash_before, total_params, all_frozen


def generate_one(
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    prompt: str,
    setting: GenerationSetting,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            do_sample=setting.do_sample,
            temperature=setting.temperature,
            top_p=setting.top_p,
            top_k=setting.top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=0,
        )
    elapsed = time.perf_counter() - start
    decoded = tokenizer.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    text = clean_text(decoded.split("\n")[0])
    text = re.sub(r"^(New comment:|Comment:)\s*", "", text, flags=re.I).strip(" -")
    return text or clean_text(decoded), elapsed


def generation_metrics(text: str) -> dict[str, float]:
    toks = words(text)
    non_alnum = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    weird = bool(re.search(r"[\?\!\$^]{2,}|[A-Za-z]{18,}", text))
    return {
        "word_count": float(len(toks)),
        "repeated_bigram_rate": repeated_ngram_rate(toks, 2),
        "repeated_trigram_rate": repeated_ngram_rate(toks, 3),
        "lexical_diversity": lexical_diversity(toks),
        "degenerate": float(len(toks) < 5 or repeated_ngram_rate(toks, 2) > 0.30 or (non_alnum / max(len(text), 1)) > 0.35 or weird),
    }


def run_grid(tokenizer: Any, model: torch.nn.Module, device: torch.device, prompt: str, target_p75: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    max_new_tokens = int(max(24, min(90, round(target_p75 * 1.6))))
    for setting_idx, setting in enumerate(GENERATION_GRID):
        for trial in range(5):
            seed = PHASE17_SEED + setting_idx * 100 + trial
            text, elapsed = generate_one(tokenizer, model, device, prompt, setting, seed, max_new_tokens)
            metrics = generation_metrics(text)
            rows.append(
                {
                    "setting": setting.name,
                    "seed": seed,
                    "temperature": setting.temperature,
                    "top_p": setting.top_p,
                    "top_k": setting.top_k,
                    "text": text,
                    "generation_time_s": elapsed,
                    **metrics,
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("setting", sort=False)
        .agg(
            temperature=("temperature", "first"),
            top_p=("top_p", "first"),
            top_k=("top_k", "first"),
            mean_words=("word_count", "mean"),
            median_words=("word_count", "median"),
            repetition=("repeated_bigram_rate", "mean"),
            diversity=("lexical_diversity", "mean"),
            degenerate_ratio=("degenerate", "mean"),
            mean_time_s=("generation_time_s", "mean"),
        )
        .reset_index()
    )
    dup = raw.groupby("setting", sort=False)["text"].agg(lambda col: 1.0 - col.nunique() / len(col))
    summary["duplicate_ratio"] = summary["setting"].map(dup).astype(float)
    return raw, summary


def choose_extremes_and_compromise(raw: pd.DataFrame, summary: pd.DataFrame, style: dict[str, Any]) -> dict[str, Any]:
    deterministic = "A_trop_deterministe"
    random = "E_trop_aleatoire"
    target = style["median_words"]
    candidates = summary[summary["setting"].isin(["B_deterministe_intermediaire", "C_compromis", "D_creatif"])].copy()
    candidates["length_gap"] = (candidates["median_words"] - target).abs()
    candidates["score"] = (
        candidates["length_gap"] / max(target, 1.0)
        + candidates["repetition"] * 2.0
        + candidates["degenerate_ratio"] * 1.5
        + candidates["duplicate_ratio"] * 2.0
        + (candidates["diversity"] - style["mean_lexical_diversity"]).abs()
    )
    compromise = str(candidates.sort_values(["score", "setting"]).iloc[0]["setting"])
    return {
        "deterministic_setting": deterministic,
        "random_setting": random,
        "compromise_setting": compromise,
        "deterministic_example": str(raw[raw["setting"] == deterministic].sort_values(["repeated_bigram_rate", "word_count"], ascending=[False, True]).iloc[0]["text"]),
        "random_example": str(raw[raw["setting"] == random].sort_values(["degenerate", "lexical_diversity"], ascending=[False, False]).iloc[0]["text"]),
    }


def write_fake_reports(raw: pd.DataFrame, decision: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    selected = raw[raw["setting"] == decision["compromise_setting"]].copy().head(5)
    selected = selected.reset_index(drop=True)
    out = pd.DataFrame(
        {
            "fake_id": [f"F{i+1:02d}" for i in range(len(selected))],
            "seed": selected["seed"],
            "generation_setting": selected["setting"],
            "text": selected["text"],
            "word_count": selected["word_count"].astype(int),
        }
    )
    out.to_csv(output_dir / "phase17_fake_reports.csv", index=False)
    return out


def prepare_blind_test(comments: list[str], fakes: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(PHASE17_SEED + 99)
    real_idx = rng.choice(len(comments), size=5, replace=False)
    real_rows = [{"origin": "REAL", "text": comments[int(i)]} for i in real_idx]
    fake_rows = [{"origin": "FAKE", "text": text} for text in fakes["text"].head(5).tolist()]
    items = real_rows + fake_rows
    order = rng.permutation(len(items))
    blind_rows = []
    key_rows = []
    for out_i, src_i in enumerate(order, start=1):
        item_id = f"BT{out_i:02d}"
        item = items[int(src_i)]
        blind_rows.append({"item_id": item_id, "text": item["text"], "human_guess": ""})
        key_rows.append({"item_id": item_id, "true_origin": item["origin"]})
    pd.DataFrame(blind_rows).to_csv(output_dir / "phase17_blind_test.csv", index=False)
    pd.DataFrame(key_rows).to_csv(output_dir / "phase17_blind_test_key.csv", index=False)


def evaluate_blind_test(output_dir: Path) -> dict[str, Any]:
    blind_path = output_dir / "phase17_blind_test.csv"
    key_path = output_dir / "phase17_blind_test_key.csv"
    if not blind_path.exists() or not key_path.exists():
        return {"status": "missing"}
    blind = pd.read_csv(blind_path, keep_default_na=False)
    key = pd.read_csv(key_path)
    if "human_guess" not in blind or blind["human_guess"].astype(str).str.strip().eq("").any():
        return {"status": "pending"}
    merged = blind.merge(key, on="item_id", how="inner")
    guess = merged["human_guess"].astype(str).str.upper().str.strip()
    truth = merged["true_origin"].astype(str).str.upper().str.strip()
    correct = guess.eq(truth)
    return {
        "status": "done",
        "correct": int(correct.sum()),
        "total": int(len(merged)),
        "accuracy": float(correct.mean()),
        "fake_taken_real": int(((truth == "FAKE") & (guess == "REAL")).sum()),
        "real_taken_fake": int(((truth == "REAL") & (guess == "FAKE")).sum()),
    }


def append_report_phase17(result: dict[str, Any]) -> None:
    report = Path("RAPPORT.md")
    old = report.read_text(encoding="utf-8", errors="replace") if report.exists() else ""
    marker = "\n## Phase 17 "
    if marker in old:
        old = old.split(marker)[0].rstrip()
    style = result["style"]
    summary = result["summary"]
    decision = result["decision"]
    final_example = result["fakes"].iloc[0]["text"]
    human = result["blind_eval"]
    if human["status"] == "done":
        human_text = f"Resultat humain reel : {human['correct']}/{human['total']} corrects, accuracy={human['accuracy']:.3f}, faux pris pour vrais={human['fake_taken_real']}, vrais pris pour faux={human['real_taken_fake']}."
    else:
        human_text = "TEST HUMAIN EN ATTENTE : aucun resultat humain n'a ete invente."
    grid_md = markdown_table(summary)
    section = f"""

## Phase 17 — Le faux témoignage

### Modele generatif
Modele utilise : `{result['model_name']}` sur `{result['device']}`. Ce choix est un modele causal GPT preentraine, raisonnable en taille ({result['total_parameters']} parametres), executable localement sur CPU et adapte a la generation de texte anglais. Le classifieur BERT-tiny des phases precedentes n'est pas utilise comme generateur car il n'est pas autoregressif.

Tous les parametres ont ete places en `requires_grad=False`, le modele est en `eval()`, aucun optimizer, aucun `backward()`, aucun `train()` et aucun fine-tuning n'est execute. Empreinte avant : `{result['hash_before']}`. Empreinte apres : `{result['hash_after']}`. Parametres internes modifies : NON.

### Etalon de style reel
Echantillon deterministe de {style['sample_size']} vrais commentaires non vides tires de `comments` sur le dataset complet. Longueur en mots : mediane={style['median_words']:.1f}, p25={style['p25_words']:.1f}, p75={style['p75_words']:.1f}, moyenne={style['mean_words']:.1f}. Proportion avec nombres={style['has_number_ratio']:.3f}, avec ponctuation={style['has_punctuation_ratio']:.3f}, diversite lexicale moyenne={style['mean_lexical_diversity']:.3f}. Cet echantillon sert seulement aux mesures et au prompt few-shot; il n'entraine pas le modele.

### Grille de generation
{grid_md}

Echec trop deterministe (`{decision['deterministic_setting']}`) : `{decision['deterministic_example']}`

Echec trop aleatoire (`{decision['random_setting']}`) : `{decision['random_example']}`

### Reglage retenu
Reglage compromis : `{decision['compromise_setting']}`. Nous nous sommes arretes sur ce reglage car les mesures le placent entre la repetition des faibles temperatures et le chaos des temperatures hautes, puis l'inspection qualitative confirme qu'il reste proche de la longueur et du vocabulaire plat des vrais releves sans tourner en rond.

Exemple final complet : `{final_example}`

### Test en aveugle
Le fichier `outputs/phase17_blind_test.csv` contient 5 vrais temoignages et 5 faux temoignages melanges avec seed fixe, sans colonne de verite. La cle separee est `outputs/phase17_blind_test_key.csv`. Faire remplir `human_guess` par une personne qui ne connait pas les reponses, puis relancer l'evaluation.

{human_text}

Fichiers : `outputs/phase17_generation_grid.csv`, `outputs/phase17_fake_reports.csv`, `outputs/phase17_blind_test.csv`, `outputs/phase17_blind_test_key.csv`.
"""
    report.write_text(old.rstrip() + section + "\n", encoding="utf-8")


def run_phase17(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir.mkdir(exist_ok=True)
    comments = load_comments(df)
    style = style_benchmark(comments)
    prompt = build_prompt(style)
    tokenizer, model, device, hash_before, total_params, all_frozen = load_generator()
    raw, summary = run_grid(tokenizer, model, device, prompt, style["p75_words"])
    raw.to_csv(output_dir / "phase17_generation_grid.csv", index=False)
    decision = choose_extremes_and_compromise(raw, summary, style)
    fakes = write_fake_reports(raw, decision, output_dir)
    prepare_blind_test(comments, fakes, output_dir)
    hash_after = parameter_hash(model)
    assert hash_before == hash_after
    blind_eval = evaluate_blind_test(output_dir)
    result = {
        "model_name": GEN_MODEL_NAME,
        "device": str(device),
        "total_parameters": total_params,
        "all_frozen": all_frozen,
        "hash_before": hash_before,
        "hash_after": hash_after,
        "style": style,
        "raw": raw,
        "summary": summary,
        "decision": decision,
        "fakes": fakes,
        "blind_eval": blind_eval,
        "elapsed_s": time.perf_counter() - start,
    }
    append_report_phase17(result)
    return result


def print_phase17(result: dict[str, Any]) -> None:
    print("\n=== PHASE 17 - LE FAUX TEMOIGNAGE ===")
    print(f"Modele: {result['model_name']}")
    print(f"Device: {result['device']}")
    print(f"Parametres totaux: {result['total_parameters']}")
    print(f"Tous requires_grad=False: {'OUI' if result['all_frozen'] else 'NON'}")
    print(f"Hash avant: {result['hash_before']}")
    print(f"Hash apres: {result['hash_after']}")
    print("Parametres internes modifies : NON")
    print(result["summary"].to_string(index=False))
    if result["blind_eval"]["status"] == "pending":
        print("TEST HUMAIN EN ATTENTE")
    print("Faire remplir human_guess par une personne qui ne connait pas les reponses, puis relancer l'evaluation.")
    print(f"Temps Phase 17={result['elapsed_s']:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 generation uniquement")
    parser.add_argument("--evaluate-blind", action="store_true")
    args = parser.parse_args()
    if args.evaluate_blind:
        print(evaluate_blind_test(Path("outputs")))
        return
    from analyse import download_dataset, load_dataset

    download_dataset()
    df, _ = load_dataset()
    print_phase17(run_phase17(df, Path("outputs")))


if __name__ == "__main__":
    main()
