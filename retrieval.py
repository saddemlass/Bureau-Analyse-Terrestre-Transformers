from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

try:
    from transformers import BertTokenizer
except ImportError:  # pragma: no cover - optional runtime dependency.
    BertTokenizer = None

from pretrained import MODEL_NAME


PHASE15_QUESTIONS = [
    "What sounds or noises did witnesses hear during the sightings?",
    "Which colors are most clearly described by witnesses?",
    "How do witnesses describe the movement of the objects?",
    "Are there sightings reported over cities or inhabited areas?",
    "What do witnesses report seeing at night?",
    "Do witnesses describe multiple objects or multiple lights?",
    "Which reports describe very long observations?",
    "Which reports describe very short observations?",
    "Are there reports of objects hovering or staying still?",
    "Did witnesses report a smell of chocolate or vanilla near the objects?",
]

PHASE15_TOKEN_BUDGET = 1500
PHASE15_TOP_CANDIDATES = 40
PHASE15_MAX_FEATURES = 60000
PHASE15_MIN_DF = 2
PHASE15_MIN_TOP_SCORE = 0.08
PHASE15_MIN_SHARED_TERMS = 1
PHASE15_MIN_COVERAGE_RATIO = 0.5
PHASE15_HUMAN_AUDIT = {
    (1, "tfidf"): (False, "TF-IDF s'abstient alors que les extraits contiennent du bruit exploitable."),
    (1, "naive"): (True, "Les citations parlent directement de sons entendus ou non entendus."),
    (2, "tfidf"): (False, "Les citations mentionnent surtout des couleurs sans repondre proprement a quelles couleurs."),
    (2, "naive"): (False, "Les sources parlent de couleurs mais ne soutiennent pas une reponse precise a quelles couleurs."),
    (3, "tfidf"): (True, "Les citations soutiennent des mouvements erratiques ou atypiques."),
    (3, "naive"): (True, "Les citations decrivent directement changement de direction, stationnaire puis mouvement ou deplacement."),
    (4, "tfidf"): (False, "L'abstention est trop prudente car des zones habitees ou nommees sont presentes."),
    (4, "naive"): (False, "L'abstention est incorrecte car des villes ou zones urbaines sont citees."),
    (5, "tfidf"): (True, "Les citations soutiennent des observations nocturnes d'objets ou lumieres."),
    (5, "naive"): (True, "Les sources citees parlent directement de choses vues la nuit."),
    (6, "tfidf"): (True, "Les citations soutiennent directement des objets ou lumieres multiples."),
    (6, "naive"): (True, "Les sources citees decrivent clairement plusieurs lumieres ou objets."),
    (7, "tfidf"): (False, "Les citations ne prouvent pas correctement de tres longues durees d'observation."),
    (7, "naive"): (False, "Le mot long est souvent interprete comme longueur physique ou expression, pas duree."),
    (8, "tfidf"): (True, "Les citations incluent des observations explicitement courtes."),
    (8, "naive"): (True, "Les citations contiennent short duration, short time ou short encounter."),
    (9, "tfidf"): (True, "Les citations soutiennent directement des objets en vol stationnaire."),
    (9, "naive"): (True, "Les citations parlent de hovering ou staying in one place."),
    (10, "tfidf"): (True, "L'abstention est correcte car les resultats ne prouvent pas chocolat ou vanille comme odeur."),
    (10, "naive"): (True, "L'abstention est correcte car les odeurs citees ne sont pas chocolat ou vanille."),
}

SIGNIFICANT_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "did",
    "does",
    "which",
    "what",
    "are",
    "there",
    "during",
    "near",
    "the",
    "a",
    "an",
    "witness",
    "witnesses",
    "report",
    "reports",
    "reported",
    "describe",
    "describes",
    "described",
    "clearly",
    "sighting",
    "sightings",
    "object",
    "objects",
}


@dataclass(frozen=True)
class RetrievedSource:
    source_id: str
    csv_index: int
    score: float
    datetime: str
    city: str
    state: str
    country: str
    shape: str
    comments: str


def phase15_tokenizer() -> Any | None:
    if BertTokenizer is None:
        return None
    try:
        return BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:  # noqa: BLE001 - documented fallback.
        return None


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def significant_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]{3,}", text.lower())
    return [term for term in terms if term not in SIGNIFICANT_STOPWORDS]


def token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return len(re.findall(r"\S+", text))


def source_context(source: RetrievedSource) -> str:
    fields = [
        f"[{source.source_id}]",
        f"datetime={source.datetime}",
        f"city={source.city}",
        f"state={source.state}",
        f"country={source.country}",
        f"shape={source.shape}",
        f"comments={source.comments}",
    ]
    return " | ".join(fields)


def excerpt(text: str, max_chars: int = 180) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class Phase15Retriever:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=False).rename(columns={"index": "csv_index"}).copy()
        self.df["source_id"] = self.df["csv_index"].map(lambda idx: f"R{int(idx)}")
        self.comments = self.df["comments"].map(clean_text).fillna("").tolist()
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=PHASE15_MIN_DF,
            max_features=PHASE15_MAX_FEATURES,
            norm="l2",
            dtype=np.float32,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.comments)
        self.count = CountVectorizer(lowercase=True, stop_words="english", binary=True)
        self.count_matrix = self.count.fit_transform(self.comments)

    @property
    def corpus_size(self) -> int:
        return len(self.df)

    def _sources_from_order(self, order: np.ndarray, scores: np.ndarray, limit: int) -> list[RetrievedSource]:
        sources = []
        for pos in order[:limit]:
            row = self.df.iloc[int(pos)]
            sources.append(
                RetrievedSource(
                    source_id=str(row["source_id"]),
                    csv_index=int(row["csv_index"]),
                    score=float(scores[int(pos)]),
                    datetime=clean_text(row.get("datetime", "")),
                    city=clean_text(row.get("city", "")),
                    state=clean_text(row.get("state", "")),
                    country=clean_text(row.get("country", "")),
                    shape=clean_text(row.get("shape", "")),
                    comments=clean_text(row.get("comments", "")),
                )
            )
        return sources

    def search_tfidf(self, question: str, limit: int = PHASE15_TOP_CANDIDATES) -> list[RetrievedSource]:
        query = self.tfidf.transform([question])
        scores = (self.tfidf_matrix @ query.T).toarray().ravel()
        order = np.lexsort((self.df["csv_index"].to_numpy(), -scores))
        return self._sources_from_order(order, scores, limit)

    def search_naive(self, question: str, limit: int = PHASE15_TOP_CANDIDATES) -> list[RetrievedSource]:
        terms = [term for term in significant_terms(question) if term in self.count.vocabulary_]
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if terms:
            cols = [self.count.vocabulary_[term] for term in sorted(set(terms))]
            scores = np.asarray(self.count_matrix[:, cols].sum(axis=1)).ravel().astype(np.float32)
        order = np.lexsort((self.df["csv_index"].to_numpy(), -scores))
        return self._sources_from_order(order, scores, limit)


def shared_term_count(question: str, source: RetrievedSource) -> int:
    question_terms = set(significant_terms(question))
    comment_terms = set(significant_terms(source.comments))
    return len(question_terms & comment_terms)


def select_with_budget(
    question: str,
    candidates: list[RetrievedSource],
    tokenizer: Any | None,
    budget: int = PHASE15_TOKEN_BUDGET,
) -> tuple[list[RetrievedSource], int]:
    selected: list[RetrievedSource] = []
    used = token_count("Question: " + question, tokenizer)
    for candidate in candidates:
        cost = token_count(source_context(candidate), tokenizer)
        if selected and used + cost > budget:
            break
        if not selected and used + cost > budget:
            continue
        selected.append(candidate)
        used += cost
    return selected, used


def should_abstain(question: str, selected: list[RetrievedSource], top_score: float) -> bool:
    if not selected or top_score < PHASE15_MIN_TOP_SCORE:
        return True
    question_terms = set(significant_terms(question))
    if not question_terms:
        return True
    best_shared = max(shared_term_count(question, source) for source in selected[:5])
    coverage = best_shared / len(question_terms)
    return best_shared < PHASE15_MIN_SHARED_TERMS or coverage < PHASE15_MIN_COVERAGE_RATIO


def build_answer(question: str, selected: list[RetrievedSource], top_score: float) -> str:
    if should_abstain(question, selected, top_score):
        return "Nous n'avons pas de releve suffisamment pertinent pour repondre a cette question."
    cited = sorted(
        selected,
        key=lambda source: (-shared_term_count(question, source), -source.score, source.csv_index),
    )[: min(3, len(selected))]
    ids = ", ".join(f"[{source.source_id}]" for source in cited)
    first = cited[0]
    return (
        f"Les releves recuperes donnent des elements directs sur la question: "
        f"le premier temoignage pertinent indique notamment \"{excerpt(first.comments, 120)}\" {ids}. "
        "Cette synthese reste limitee aux passages cites."
    )


def human_audit_decision(question_id: int, method: str) -> tuple[bool, str]:
    return PHASE15_HUMAN_AUDIT[(question_id, method)]


def run_phase15(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    tokenizer = phase15_tokenizer()
    retriever = Phase15Retriever(df)
    methods = {
        "tfidf": retriever.search_tfidf,
        "naive": retriever.search_naive,
    }
    question_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for qid, question in enumerate(PHASE15_QUESTIONS, start=1):
        for method, search in methods.items():
            candidates = search(question)
            selected, tokens_used = select_with_budget(question, candidates, tokenizer)
            top_score = candidates[0].score if candidates else 0.0
            answer = build_answer(question, selected, top_score)
            answer_sources = []
            if not should_abstain(question, selected, top_score):
                answer_sources = sorted(
                    selected,
                    key=lambda source: (-shared_term_count(question, source), -source.score, source.csv_index),
                )[: min(3, len(selected))]
            retrieved_ids = " ".join(source.source_id for source in selected)
            excerpts = "\n".join(
                f"[{source.source_id}] {source.datetime} | {source.city} | {source.state} | "
                f"{source.country} | {source.shape} | {excerpt(source.comments)}"
                for source in selected[:5]
            )
            source_correct, justification = human_audit_decision(qid, method)
            question_rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "method": method,
                    "retrieved_ids": retrieved_ids,
                    "tokens_used": tokens_used,
                    "token_budget": PHASE15_TOKEN_BUDGET,
                    "top_score": top_score,
                    "answer_or_status": answer,
                }
            )
            audit_rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "method": method,
                    "retrieved_ids": retrieved_ids,
                    "source_excerpts": excerpts,
                    "source_correct": str(source_correct).upper(),
                    "justification_humaine": justification,
                }
            )
            summaries.append(
                {
                    "question_id": qid,
                    "question": question,
                    "method": method,
                    "selected_count": len(selected),
                    "tokens_used": tokens_used,
                    "top_score": top_score,
                    "answer": answer,
                    "sources": selected,
                    "answer_sources": answer_sources,
                    "abstained": should_abstain(question, selected, top_score),
                    "source_correct": source_correct,
                }
            )

    output_dir.mkdir(exist_ok=True)
    questions_path = output_dir / "phase15_questions.csv"
    audit_path = output_dir / "phase15_source_audit.csv"
    pd.DataFrame(question_rows).to_csv(questions_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    comparison = pd.DataFrame(summaries)
    source_rates = comparison.groupby("method", as_index=False).agg(
        correct=("source_correct", "sum"),
        total=("source_correct", "count"),
    )
    source_rates["correct_source_rate"] = source_rates["correct"] / source_rates["total"]
    fig_path = output_dir / "phase15_retrieval_comparison.png"
    source_rates = source_rates.set_index("method").loc[["tfidf", "naive"]].reset_index()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(source_rates["method"], source_rates["correct_source_rate"] * 100, color=["#2563eb", "#16a34a"])
    plt.ylim(0, 100)
    plt.ylabel("reponses correctement sourcees (%)")
    plt.title("Proportion de réponses correctement sourcées — audit humain")
    for bar, rate in zip(bars, source_rates["correct_source_rate"], strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{rate:.0%}", ha="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    repeated_question = PHASE15_QUESTIONS[0]
    run1 = [source.source_id for source in retriever.search_tfidf(repeated_question)[:10]]
    run2 = [source.source_id for source in retriever.search_tfidf(repeated_question)[:10]]
    assert run1 == run2

    return {
        "elapsed_s": time.perf_counter() - start,
        "corpus_size": retriever.corpus_size,
        "questions": PHASE15_QUESTIONS,
        "token_budget": PHASE15_TOKEN_BUDGET,
        "tfidf_config": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "min_df": PHASE15_MIN_DF,
            "max_features": PHASE15_MAX_FEATURES,
            "norm": "l2",
        },
        "naive_config": "mots significatifs de la question, stopwords anglais retires, score=nombre de mots presents",
        "summaries": summaries,
        "question_rows": question_rows,
        "audit_rows": audit_rows,
        "human_audit_rows": len(audit_rows),
        "review_required": 0,
        "source_rates": {
            row.method: {
                "correct": int(row.correct),
                "total": int(row.total),
                "rate": float(row.correct_source_rate),
            }
            for row in source_rates.itertuples(index=False)
        },
        "determinism": {
            "question": repeated_question,
            "run1": run1,
            "run2": run2,
            "same": run1 == run2,
        },
        "outputs": [questions_path, audit_path, fig_path],
        "tokenizer": MODEL_NAME if tokenizer is not None else "fallback whitespace",
        "abstention_rule": (
            f"abstention si top_score < {PHASE15_MIN_TOP_SCORE}, si aucun terme informatif commun "
            f"n'apparait dans les 5 premieres sources, ou si la meilleure couverture des termes "
            f"informatifs est < {PHASE15_MIN_COVERAGE_RATIO:.2f}; seuils fixes avant evaluation"
        ),
    }
