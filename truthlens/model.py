"""Loading the trained model, making predictions and explaining them.

Saved model format (no pickle, so loading it can never run code):
    models/model.json    -> settings, vocabulary (list of terms), metadata
    models/weights.npz   -> idf, coef, bias  (loaded with allow_pickle=False)

How the prediction works:
1. clean the text (truthlens.text.clean_text)
2. TF-IDF: count each known word / word-pair, damp counts with 1 + log(count),
   multiply by how rare the term is (idf), then scale the vector to length 1
3. Logistic Regression: score = bias + sum(tfidf(term) * coef(term)),
   P(fake) = 1 / (1 + e^-score)

How the explanation works (exact, not an approximation):
each term's share of that sum, tfidf(term) * coef(term), is literally how
much it pushed the decision. Positive = towards FAKE, negative = towards REAL.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .text import clean_text

# Same token rule as scikit-learn's TfidfVectorizer default.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

FAKE_THRESHOLD = 0.65  # P(fake) at or above -> "Likely fake"
REAL_THRESHOLD = 0.35  # P(fake) at or below -> "Likely real"


@dataclass
class Prediction:
    label: str            # "FAKE", "REAL" or "UNCERTAIN"
    verdict: str          # human-friendly sentence
    prob_fake: float
    top_fake_terms: list[dict] = field(default_factory=list)
    top_real_terms: list[dict] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)  # [{"text", "weight"}]
    warnings: list[str] = field(default_factory=list)
    known_word_ratio: float = 0.0


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n_max: int) -> list[str]:
    out = list(tokens)
    for n in range(2, n_max + 1):
        out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return out


class Predictor:
    def __init__(self, terms: list[str], idf: np.ndarray, coef: np.ndarray,
                 bias: float, ngram_max: int = 2, meta: dict | None = None):
        if not (len(terms) == len(idf) == len(coef)):
            raise ValueError("Model files are inconsistent (vocabulary/weights size mismatch).")
        self.terms = terms
        self.index = {t: i for i, t in enumerate(terms)}
        self.idf = idf.astype(np.float64)
        self.coef = coef.astype(np.float64)
        self.bias = float(bias)
        self.ngram_max = ngram_max
        self.meta = meta or {}

    # ---------- saving / loading ----------
    def save(self, model_dir: Path = config.MODEL_DIR) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.json").write_text(json.dumps({
            "format": "truthlens-linear-v1",
            "ngram_max": self.ngram_max,
            "terms": self.terms,
            "meta": self.meta,
        }))
        np.savez_compressed(model_dir / "weights.npz", idf=self.idf.astype(np.float32),
                            coef=self.coef.astype(np.float32), bias=np.array([self.bias]))

    @classmethod
    def load(cls, model_dir: Path = config.MODEL_DIR) -> "Predictor":
        spec = json.loads((model_dir / "model.json").read_text())
        if spec.get("format") != "truthlens-linear-v1":
            raise ValueError("Unknown model format.")
        with np.load(model_dir / "weights.npz", allow_pickle=False) as w:
            idf, coef, bias = w["idf"], w["coef"], float(w["bias"][0])
        return cls(spec["terms"], idf, coef, bias, spec.get("ngram_max", 2), spec.get("meta", {}))

    # ---------- features ----------
    def vectorize(self, text: str) -> dict[int, float]:
        """TF-IDF vector of already-cleaned text, as {term_index: value}."""
        counts = Counter(g for g in ngrams(tokenize(text), self.ngram_max) if g in self.index)
        vec = {self.index[g]: (1.0 + math.log(c)) * self.idf[self.index[g]] for g, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {j: v / norm for j, v in vec.items()} if norm > 0 else {}

    def prob_fake(self, raw_text: str) -> float:
        vec = self.vectorize(clean_text(raw_text))
        score = self.bias + sum(v * self.coef[j] for j, v in vec.items())
        return 1.0 / (1.0 + math.exp(-score))

    # ---------- prediction + explanation ----------
    def predict(self, raw_text: str, top_k: int = 8) -> Prediction:
        text = clean_text(raw_text, max_chars=config.MAX_INPUT_CHARS)
        vec = self.vectorize(text)
        contrib = {self.terms[j]: v * self.coef[j] for j, v in vec.items()}
        score = self.bias + sum(contrib.values())
        prob_fake = 1.0 / (1.0 + math.exp(-score))

        ranked = sorted(contrib.items(), key=lambda kv: kv[1])
        top_real = [{"term": t, "weight": round(w, 4)} for t, w in ranked[:top_k] if w < 0]
        top_fake = [{"term": t, "weight": round(w, 4)} for t, w in reversed(ranked[-top_k:]) if w > 0]

        tokens = tokenize(text)
        known_ratio = sum(t in self.index for t in tokens) / len(tokens) if tokens else 0.0

        if not vec:
            # Nothing the model recognises: the score would be the bias alone,
            # which is not evidence either way.
            label, verdict = "UNCERTAIN", "The model doesn't recognise any of these words, so it can't judge this text."
        elif prob_fake >= FAKE_THRESHOLD:
            label, verdict = "FAKE", "The writing style resembles the fake news the model studied."
        elif prob_fake <= REAL_THRESHOLD:
            label, verdict = "REAL", "The writing style resembles the genuine news the model studied."
        else:
            label, verdict = "UNCERTAIN", "The text shows mixed signals, so the model won't call it either way."

        warnings = []
        if len(tokens) < 25:
            warnings.append(
                "Short text: a headline alone gives a weaker signal than a full article. "
                "Paste the article body too if you have it."
            )
        if not vec:
            warnings.append(
                "No recognisable English news words were found after cleaning "
                "(links, @handles and source names are removed)."
            )
        elif known_ratio < 0.6:
            warnings.append(
                "Many words are unfamiliar to the model (different topic, period or language), "
                "so treat this result with extra caution."
            )

        return Prediction(
            label=label, verdict=verdict, prob_fake=round(prob_fake, 4),
            top_fake_terms=top_fake, top_real_terms=top_real,
            highlights=self._highlights(text, contrib), warnings=warnings,
            known_word_ratio=round(known_ratio, 3),
        )

    def _highlights(self, text: str, contrib: dict[str, float]) -> list[dict]:
        """Split cleaned text into segments with a colouring weight.

        A term's contribution covers ALL its occurrences, so it is split evenly
        between them. Each word gets its own share plus half of the share of
        every word-pair (bigram) it belongs to. Summed over all words, the
        weights therefore equal the total contribution.
        """
        lower = text.lower()
        if len(lower) != len(text):
            # Rare scripts change length when lower-cased; skip highlighting
            # rather than risk marking the wrong words.
            return [{"text": text, "weight": 0.0}]
        matches = list(TOKEN_RE.finditer(lower))  # same tokens as tokenize()
        words = [m.group(0) for m in matches]
        pairs = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        word_n, pair_n = Counter(words), Counter(pairs)

        weights = [contrib.get(w, 0.0) / word_n[w] for w in words]
        for i, pair in enumerate(pairs):
            half = contrib.get(pair, 0.0) / pair_n[pair] / 2
            weights[i] += half
            weights[i + 1] += half

        segments, pos = [], 0
        for m, wt in zip(matches, weights):
            if m.start() > pos:
                segments.append({"text": text[pos:m.start()], "weight": 0.0})
            segments.append({"text": text[m.start():m.end()], "weight": round(wt, 4)})
            pos = m.end()
        if pos < len(text):
            segments.append({"text": text[pos:], "weight": 0.0})
        return segments
