"""Train, compare and evaluate TruthLens models, then save the demo model.

Run:  uv run --group train python scripts/train.py

Steps
1. Download WELFake (training data) and LIAR2 (hard cross-dataset test) if missing.
2. Clean text, remove label-leaking phrases, drop duplicates.
3. Split 70/15/15 (train/validation/test). Articles with the same headline are
   kept in the same split so near-copies can't leak from train into test.
4. Train three classic models on TF-IDF features, tune each on validation:
   Multinomial Naive Bayes, Logistic Regression, Linear SVM.
5. Evaluate on: WELFake test (articles), WELFake test (headlines only), LIAR2.
6. Save the Logistic Regression model for the web app (it gives probabilities
   and exact word-level explanations), plus metrics and figures for the report.

Note on labels: the WELFake dataset card says 1 = real, but the data says
otherwise (21,266 of 21,282 Reuters articles have label 0, and the paper reports
37,106 fake articles = the count of label 1). So here 1 = FAKE, 0 = REAL.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from truthlens import config  # noqa: E402
from truthlens.model import Predictor  # noqa: E402
from truthlens.text import clean_text, join_title_body  # noqa: E402

SEED = 42
RAW = config.DATA_DIR / "raw"
FIG = config.REPORTS_DIR / "figures"
SOURCES = {
    "welfake.parquet": "https://huggingface.co/datasets/davanstrien/WELFake/resolve/main/"
                       "data/train-00000-of-00001-290868f0a36350c5.parquet",
    "liar2_test.csv": "https://huggingface.co/datasets/chengxuphd/liar2/resolve/main/test.csv",
}

# Chart colours (validated reference palette: blue <-> red diverging pair).
REAL_C, FAKE_C, SERIES_2 = "#2a78d6", "#e34948", "#eb6834"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data
def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            log(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def load_welfake() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "welfake.parquet")
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["y"] = df["label"].astype(int)  # 1 = FAKE (see module docstring)
    df["title_clean"] = df["title"].map(clean_text)
    df["body_clean"] = df["text"].map(clean_text)
    # Clean title and body separately, then join (see join_title_body).
    df["doc"] = [join_title_body(t, b) for t, b in zip(df.title_clean, df.body_clean)]
    n0 = len(df)
    df = df[df["doc"].str.split().str.len() >= 5]
    df = df.drop_duplicates("doc")

    # Keys used to spot copies of the same story: normalised headline and the
    # first 300 characters of the body (catches the same body under a new title).
    norm = lambda s: s.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()  # noqa: E731
    df["k_title"] = norm(df["title_clean"])
    df["k_body"] = norm(df["body_clean"]).str[:300]

    # Drop stories whose copies carry conflicting labels (pure noise).
    for k in ("k_title", "k_body"):
        keyed = df[df[k].str.len() >= 20]
        mixed = keyed.groupby(k)["y"].nunique()
        df = df[~df[k].isin(mixed[mixed > 1].index)]
    log(f"WELFake: {n0} rows -> {len(df)} after cleaning, de-duplication, conflict removal")
    df = df.reset_index(drop=True)
    df["group"] = story_groups(df)
    return df


def story_groups(df: pd.DataFrame) -> np.ndarray:
    """Union-find: rows sharing a headline OR a body opening belong to one group,
    so all copies of a story land in the same split (no train/test leakage)."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in ("k_title", "k_body"):
        first = {}
        for i, key in enumerate(df[k]):
            if len(key) < 20:  # too short to identify a story
                continue
            if key in first:
                a, b = find(i), find(first[key])
                if a != b:
                    parent[a] = b
            else:
                first[key] = i
    return np.array([find(i) for i in range(len(df))])


def load_liar2() -> pd.DataFrame:
    """LIAR2 labels: 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true.
    Keep the clear cases only: 0-1 -> FAKE, 4-5 -> REAL."""
    df = pd.read_csv(RAW / "liar2_test.csv")
    df = df[df["label"].isin([0, 1, 4, 5])].copy()
    df["y"] = df["label"].isin([0, 1]).astype(int)
    df["doc"] = df["statement"].fillna("").map(clean_text)
    df = df[df["doc"].str.len() > 0]
    log(f"LIAR2 (clear true/false claims only): {len(df)} rows")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, rest = next(gss.split(df, groups=df["group"]))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va, te = next(gss2.split(rest_df, groups=rest_df["group"]))
    return df.iloc[tr], rest_df.iloc[va], rest_df.iloc[te]


def with_headlines(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Training examples: every article, plus its headline alone as an extra
    example, so the model also copes with headline-only input in the demo."""
    heads = df[df["title_clean"].str.split().str.len() >= 4]
    docs = df["doc"].tolist() + heads["title_clean"].tolist()
    ys = np.concatenate([df["y"].values, heads["y"].values])
    return docs, ys


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fake": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fake": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def fig_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0), facecolor=SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#256abf", "#0d366b"])
    ax.imshow(cm, cmap=cmap)
    labels = ["REAL", "FAKE"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("Actual", color=INK_2)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.45
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / total:.1%})", ha="center", va="center",
                    color="white" if dark else INK, fontsize=11)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_comparison(results: dict, path: Path) -> None:
    models = list(results)
    sets = [("welfake_test", "WELFake test (articles)", REAL_C), ("liar2", "LIAR2 claims (hard test)", SERIES_2)]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    _style(ax)
    for k, (key, name, color) in enumerate(sets):
        vals = [results[m][key]["f1_macro"] for m in models]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F1 score", color=INK_2)
    ax.axhline(0.5, color=INK_2, lw=0.8, ls="--")
    ax.text(len(models) - 0.5, 0.515, "coin flip", color=INK_2, fontsize=8, ha="right")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_top_terms(pred: Predictor, path: Path, k: int = 15) -> None:
    order = np.argsort(pred.coef)
    idx = np.concatenate([order[:k], order[-k:]])
    terms = [pred.terms[i] for i in idx]
    vals = pred.coef[idx]
    fig, ax = plt.subplots(figsize=(7, 7.5), facecolor=SURFACE)
    _style(ax)
    ax.barh(range(len(idx)), vals, color=[FAKE_C if v > 0 else REAL_C for v in vals], height=0.7)
    ax.set_yticks(range(len(idx)), terms, fontsize=9)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Model weight  (←  pushes towards REAL   |   pushes towards FAKE  →)", color=INK_2)
    ax.set_title(f"Strongest {k} words/phrases in each direction", color=INK, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    download()
    df = load_welfake()
    liar = load_liar2()
    train, val, test = split(df)
    log(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    X_docs, y_train = with_headlines(train)
    test_heads = test[test["title_clean"].str.split().str.len() >= 4]

    log("Fitting TF-IDF vectoriser ...")
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=5, max_df=0.8, max_features=100_000, dtype=np.float32)
    Xtr = vec.fit_transform(X_docs)
    Xva, Xte = vec.transform(val["doc"]), vec.transform(test["doc"])
    Xte_h, Xli = vec.transform(test_heads["title_clean"]), vec.transform(liar["doc"])

    candidates = {
        "Naive Bayes": [MultinomialNB(alpha=a) for a in (0.03, 0.1, 0.3, 1.0)],
        "Logistic Regression": [LogisticRegression(C=c, max_iter=2000, solver="liblinear")
                                for c in (0.25, 1, 4, 16)],
        "Linear SVM": [LinearSVC(C=c) for c in (0.1, 0.3, 1.0)],
    }
    results, best_models = {}, {}
    for name, models in candidates.items():
        scored = []
        for m in models:
            m.fit(Xtr, y_train)
            scored.append((f1_score(val["y"], m.predict(Xva), average="macro"), m))
        best_f1, best = max(scored, key=lambda s: s[0])
        best_models[name] = best
        results[name] = {
            "params": {k: v for k, v in best.get_params().items() if k in ("alpha", "C")},
            "val": metrics(val["y"], best.predict(Xva)),
            "welfake_test": metrics(test["y"], best.predict(Xte)),
            "welfake_test_headlines": metrics(test_heads["y"], best.predict(Xte_h)),
            "liar2": metrics(liar["y"], best.predict(Xli)),
        }
        log(f"{name}: params={results[name]['params']} val F1={best_f1:.4f} "
            f"test F1={results[name]['welfake_test']['f1_macro']} "
            f"headlines F1={results[name]['welfake_test_headlines']['f1_macro']} "
            f"LIAR2 F1={results[name]['liar2']['f1_macro']}")

    # ---- demo model = Logistic Regression
    lr = best_models["Logistic Regression"]
    meta = {
        "trained_on": str(date.today()),
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "params": results["Logistic Regression"]["params"],
        "train_examples": int(Xtr.shape[0]),
        "vocabulary_size": int(len(vec.vocabulary_)),
        "test_metrics": results["Logistic Regression"]["welfake_test"],
        "headline_metrics": results["Logistic Regression"]["welfake_test_headlines"],
        "liar2_metrics": results["Logistic Regression"]["liar2"],
    }
    pred = Predictor(list(vec.get_feature_names_out()), vec.idf_, lr.coef_.ravel(),
                     float(lr.intercept_[0]), ngram_max=2, meta=meta)
    pred.save()

    # ---- sanity check: our numpy/python inference == scikit-learn's
    reloaded = Predictor.load()
    sample = test["doc"].sample(200, random_state=SEED).tolist()
    ours = np.array([reloaded.prob_fake(t) for t in sample])
    theirs = lr.predict_proba(vec.transform(sample))[:, 1]
    gap = float(np.abs(ours - theirs).max())
    log(f"Inference check vs scikit-learn: max |difference| = {gap:.2e}")
    assert gap < 1e-3, "Re-implemented inference disagrees with scikit-learn"

    # ---- metrics with the app's own decision rule (0.35-0.65 = "Uncertain")
    from truthlens.model import FAKE_THRESHOLD, REAL_THRESHOLD

    def selective(y_true, probs) -> dict:
        decided = (probs >= FAKE_THRESHOLD) | (probs <= REAL_THRESHOLD)
        y_true = np.asarray(y_true)
        acc = accuracy_score(y_true[decided], (probs[decided] >= 0.5).astype(int)) if decided.any() else 0.0
        return {"coverage": round(float(decided.mean()), 4), "accuracy_when_decided": round(float(acc), 4)}

    app_rule = {
        "welfake_test": selective(test["y"], lr.predict_proba(Xte)[:, 1]),
        "welfake_test_headlines": selective(test_heads["y"], lr.predict_proba(Xte_h)[:, 1]),
        "liar2": selective(liar["y"], lr.predict_proba(Xli)[:, 1]),
    }
    baselines = {  # always guessing the most common class
        "welfake_test": round(float(max(test["y"].mean(), 1 - test["y"].mean())), 4),
        "liar2": round(float(max(liar["y"].mean(), 1 - liar["y"].mean())), 4),
    }
    log(f"App decision rule: {app_rule}")
    log(f"Majority-class baselines: {baselines}")

    # ---- reports
    FIG.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(test["y"], lr.predict(Xte))
    cm_l = confusion_matrix(liar["y"], lr.predict(Xli))
    fig_confusion(cm, "Logistic Regression: WELFake test", FIG / "confusion_welfake.png")
    fig_confusion(cm_l, "Logistic Regression: LIAR2 claims", FIG / "confusion_liar2.png")
    fig_comparison(results, FIG / "model_comparison.png")
    fig_top_terms(pred, FIG / "top_terms.png")

    summary = {
        "dataset": {"welfake_after_cleaning": len(df), "train": len(train), "val": len(val),
                    "test": len(test), "test_headlines": len(test_heads), "liar2": len(liar),
                    "train_examples_incl_headlines": int(Xtr.shape[0])},
        "results": results,
        "logistic_regression_with_app_rule": app_rule,
        "majority_class_baseline_accuracy": baselines,
        "confusion_welfake_test": cm.tolist(),
        "confusion_liar2": cm_l.tolist(),
        "inference_max_abs_diff": gap,
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    log("Done. Model -> models/, metrics + figures -> reports/")


if __name__ == "__main__":
    main()
