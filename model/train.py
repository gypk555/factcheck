"""
Step 3: train every model on the cached features and evaluate on the test split.

  * TF-IDF + Logistic Regression (classical baseline)
  * Text-only, Image-only, Concat fusion (deep baselines)
  * FND-CLIP (base paper)
  * XFND-CLIP (proposed) + 4 ablation variants

Each deep model is trained with 3 random seeds and we report mean +- std,
so a lucky run does not decide the comparison.

Outputs (results/):
  metrics.json          - all test metrics, per model, per seed + mean/std
  history_<model>.json  - loss / val-F1 per epoch (seed 0)
  preds_<model>.csv     - test-set probabilities (seed 0), used for plots
checkpoints/<model>.pt  - best weights (seed 0), used by explain.py / app.py
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

import config as C
from models import FocalLoss, build

SEEDS = [int(s) for s in os.environ.get("XFND_SEEDS", "42,7,2024").split(",")]

DEEP_MODELS = ["text_only", "image_only", "concat", "fnd_clip", "xfnd_clip",
               "xfnd_no_consistency", "xfnd_no_availability", "xfnd_no_attention",
               "xfnd_ce_loss"]
FOCAL_MODELS = {"xfnd_clip", "xfnd_no_consistency", "xfnd_no_availability", "xfnd_no_attention"}
KEYS = ("bert", "resnet", "clip_t", "clip_i", "has_image")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision_fake": precision_score(y, pred, zero_division=0),
        "recall_fake": recall_score(y, pred, zero_division=0),
        "f1_fake": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y, prob)
    except ValueError:
        out["roc_auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def load_split(split):
    return torch.load(os.path.join(C.FEATURE_DIR, f"{split}.pt"))


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def batches(d, bs, shuffle):
    n = len(d["label"])
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        j = idx[s:s + bs]
        if shuffle and len(j) < 2:      # BatchNorm needs >1 sample
            continue
        yield {k: d[k][j].to(C.DEVICE) for k in KEYS}, d["label"][j].to(C.DEVICE)


@torch.no_grad()
def predict(model, d):
    model.eval()
    probs = []
    for b, _ in batches(d, 512, False):
        logits, _ = model(b)
        probs.append(torch.softmax(logits, -1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def train_one(name, tr, va, sim_stats, seed):
    set_seed(seed)
    model = build(name, *sim_stats).to(C.DEVICE)
    if name in FOCAL_MODELS:
        counts = torch.bincount(tr["label"], minlength=2).float()
        alpha = (counts.sum() / (2 * counts)).to(C.DEVICE)
        loss_fn = FocalLoss(alpha)
    else:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, C.EPOCHS)

    best_f1, best_state, wait, hist = -1, None, 0, []
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for b, y in batches(tr, C.BATCH_SIZE, True):
            logits, _ = model(b)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        vp = predict(model, va)
        vm = metrics(va["label"].numpy(), vp)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_f1_macro": vm["f1_macro"],
                     "val_accuracy": vm["accuracy"]})
        if vm["f1_macro"] > best_f1:
            best_f1, wait = vm["f1_macro"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, hist


def tfidf_baseline(df):
    tr, te = df[df.split == "train"], df[df.split == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(tr.text), tr.label)
    prob = clf.predict_proba(vec.transform(te.text))[:, 1]
    pd.DataFrame({"uid": te.uid.values, "label": te.label.values, "prob_fake": prob}) \
        .to_csv(os.path.join(C.RESULTS_DIR, "preds_tfidf_lr.csv"), index=False)
    return metrics(te.label.values, prob)


def main():
    t_start = time.time()
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    print(f"train {len(tr['label'])} | val {len(va['label'])} | test {len(te['label'])} | "
          f"device {C.DEVICE}")

    # similarity statistics for FND-CLIP's standardisation, from training images only
    m = tr["has_image"] == 1
    sims = (tr["clip_t"][m] * tr["clip_i"][m]).sum(-1)
    sim_stats = (sims.mean().item(), max(sims.std().item(), 1e-3)) if m.any() else (0.25, 0.05)
    print(f"CLIP similarity mean={sim_stats[0]:.4f} std={sim_stats[1]:.4f}")

    df = pd.read_csv(os.path.join(C.DATA_DIR, "processed.csv"))
    results = {"tfidf_lr": {"seeds": [tfidf_baseline(df)]}}
    print("tfidf_lr", results["tfidf_lr"]["seeds"][0])

    for name in DEEP_MODELS:
        # the image-only model is only meaningful on items that have an image
        if name == "image_only":
            trs, vas, tes = (subset(d, d["has_image"] == 1) for d in (tr, va, te))
        else:
            trs, vas, tes = tr, va, te
        if len(trs["label"]) < 20:
            print(f"skipping {name}: not enough samples"); continue

        runs = []
        for k, seed in enumerate(SEEDS):
            model, hist = train_one(name, trs, vas, sim_stats, seed)
            prob = predict(model, tes)
            runs.append(metrics(tes["label"].numpy(), prob))
            if k == 0:
                torch.save({"name": name, "state": model.state_dict(), "sim_stats": sim_stats},
                           os.path.join(C.CKPT_DIR, f"{name}.pt"))
                with open(os.path.join(C.RESULTS_DIR, f"history_{name}.json"), "w") as f:
                    json.dump(hist, f, indent=1)
                pd.DataFrame({"uid": tes["uid"].numpy(), "label": tes["label"].numpy(),
                              "prob_fake": prob, "has_image": tes["has_image"].numpy()}) \
                    .to_csv(os.path.join(C.RESULTS_DIR, f"preds_{name}.csv"), index=False)
        results[name] = {"seeds": runs}
        print(name, runs[0])

    for name, r in results.items():
        keys = r["seeds"][0].keys()
        r["mean"] = {k: round(float(np.mean([s[k] for s in r["seeds"]])), 4) for k in keys}
        r["std"] = {k: round(float(np.std([s[k] for s in r["seeds"]])), 4) for k in keys}
    results["_meta"] = {"seeds": SEEDS, "device": C.DEVICE, "minutes": round((time.time() - t_start) / 60, 2),
                        "sim_mean": sim_stats[0], "sim_std": sim_stats[1]}
    with open(os.path.join(C.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {results['_meta']['minutes']} min. Results in {C.RESULTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
