"""
Step 4: turn the saved predictions and metrics into the tables and figures
used in the report and the presentation.

Outputs: results/results_table.csv, results/ablation_table.csv,
         results/subgroup_table.csv, results/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc

import config as C
from models import build

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN = "#2f5d8a", "#d9822b", "#9aa5b1", "#3c8d5a"

MAIN = ["tfidf_lr", "text_only", "image_only", "concat", "fnd_clip", "xfnd_clip"]
LABELS = {"tfidf_lr": "TF-IDF + LR", "text_only": "Text only (BERT)",
          "image_only": "Image only (ResNet)*", "concat": "Concat fusion",
          "fnd_clip": "FND-CLIP (base)", "xfnd_clip": "XFND-CLIP (proposed)",
          "xfnd_no_consistency": "w/o consistency features",
          "xfnd_no_availability": "w/o availability-aware attention",
          "xfnd_no_attention": "w/o modality attention",
          "xfnd_ce_loss": "w/o focal loss (plain CE)"}
ABL = ["xfnd_clip", "xfnd_no_consistency", "xfnd_no_availability", "xfnd_no_attention", "xfnd_ce_loss"]


def fig(name):
    return os.path.join(C.FIG_DIR, name)


def fmt(r, k):
    return f"{100 * r['mean'][k]:.2f} ± {100 * r['std'][k]:.2f}"


def tables(M):
    rows = []
    for m in MAIN:
        if m in M:
            r = M[m]
            rows.append({"Model": LABELS[m], "Accuracy (%)": fmt(r, "accuracy"),
                         "Precision-fake (%)": fmt(r, "precision_fake"),
                         "Recall-fake (%)": fmt(r, "recall_fake"),
                         "F1-fake (%)": fmt(r, "f1_fake"), "Macro-F1 (%)": fmt(r, "f1_macro"),
                         "ROC-AUC": f"{r['mean']['roc_auc']:.4f}"})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "results_table.csv"), index=False)

    rows = []
    for m in ABL:
        if m in M:
            r = M[m]
            rows.append({"Variant": LABELS[m], "Accuracy (%)": fmt(r, "accuracy"),
                         "Macro-F1 (%)": fmt(r, "f1_macro"),
                         "Δ Macro-F1 vs full": f"{100 * (r['mean']['f1_macro'] - M['xfnd_clip']['mean']['f1_macro']):+.2f}"})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "ablation_table.csv"), index=False)


def comparison_plot(M):
    ms = [m for m in MAIN if m in M]
    x = np.arange(len(ms))
    acc = [100 * M[m]["mean"]["accuracy"] for m in ms]
    f1 = [100 * M[m]["mean"]["f1_macro"] for m in ms]
    acc_e = [100 * M[m]["std"]["accuracy"] for m in ms]
    f1_e = [100 * M[m]["std"]["f1_macro"] for m in ms]
    fig_, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, acc, 0.4, yerr=acc_e, capsize=3, color=BLUE, label="Accuracy")
    ax.bar(x + 0.2, f1, 0.4, yerr=f1_e, capsize=3, color=ORANGE, label="Macro-F1")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - 0.2, a + acc_e[i] + 0.8, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f + f1_e[i] + 0.8, f"{f:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [LABELS[m] for m in ms], rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(max(0, min(acc + f1) - 15), 104)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Test performance (mean of 3 seeds)", pad=24)
    plt.savefig(fig("fig_model_comparison.png")); plt.close()


def confusion_plots():
    for m in ("fnd_clip", "xfnd_clip"):
        p = os.path.join(C.RESULTS_DIR, f"preds_{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        cm = confusion_matrix(d.label, (d.prob_fake >= 0.5).astype(int))
        fig_, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Real", "Fake"]); ax.set_yticks([0, 1], ["Real", "Fake"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(LABELS[m])
        plt.savefig(fig(f"fig_confusion_{m}.png")); plt.close()


def roc_plot():
    fig_, ax = plt.subplots(figsize=(5, 4.4))
    colors = [GREY, "#7b8fa6", "#b5a07a", "#8c6bb1", BLUE, ORANGE]
    for m, c in zip(MAIN, colors):
        p = os.path.join(C.RESULTS_DIR, f"preds_{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        fpr, tpr, _ = roc_curve(d.label, d.prob_fake)
        ax.plot(fpr, tpr, color=c, lw=2 if m == "xfnd_clip" else 1.3,
                label=f"{LABELS[m]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#cccccc", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_title("ROC curves (test set)")
    plt.savefig(fig("fig_roc.png")); plt.close()


def curves_plot():
    fig_, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for m, c in (("fnd_clip", BLUE), ("xfnd_clip", ORANGE)):
        p = os.path.join(C.RESULTS_DIR, f"history_{m}.json")
        if not os.path.exists(p):
            continue
        h = pd.DataFrame(json.load(open(p)))
        axes[0].plot(h.epoch, h.train_loss, color=c, label=LABELS[m])
        axes[1].plot(h.epoch, 100 * h.val_f1_macro, color=c, label=LABELS[m])
    axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Validation macro-F1 (%)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)
    plt.savefig(fig("fig_training_curves.png")); plt.close()


def ablation_plot(M):
    ms = [m for m in ABL if m in M]
    v = [100 * M[m]["mean"]["f1_macro"] for m in ms]
    e = [100 * M[m]["std"]["f1_macro"] for m in ms]
    fig_, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(range(len(ms)), v, xerr=e, capsize=3,
            color=[ORANGE] + [GREY] * (len(ms) - 1))
    for i, x in enumerate(v):
        ax.text(x + e[i] + 0.3, i, f"{x:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(ms)), [LABELS[m] for m in ms]); ax.invert_yaxis()
    ax.set_xlim(max(0, min(v) - 8), min(101, max(v) + max(e) + 3))
    ax.set_xlabel("Macro-F1 (%)"); ax.set_title("Ablation study of XFND-CLIP")
    plt.savefig(fig("fig_ablation.png")); plt.close()


def dataset_plots():
    df = pd.read_csv(os.path.join(C.DATA_DIR, "processed.csv"))
    fig_, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    cnt = df.label.value_counts().sort_index()
    axes[0].bar(["Real", "Fake"], cnt.values, color=[BLUE, ORANGE])
    axes[0].set_title("Class balance (after sampling)")
    for i, v in enumerate(cnt.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    cov = df.groupby("label").has_image.mean() * 100
    axes[1].bar(["Real", "Fake"], cov.values, color=[BLUE, ORANGE])
    axes[1].set_title("Items with a usable image (%)"); axes[1].set_ylim(0, 100)
    top = df.category.value_counts().head(6)
    axes[2].barh(top.index[::-1], top.values[::-1], color=GREEN)
    axes[2].set_title("Top categories")
    plt.savefig(fig("fig_dataset_stats.png")); plt.close()

    words = df.text.str.split().str.len()
    fig_, ax = plt.subplots(figsize=(5, 3))
    ax.hist([words[df.label == 0], words[df.label == 1]], bins=30, color=[BLUE, ORANGE],
            label=["Real", "Fake"], stacked=False, alpha=0.85)
    ax.set_xlabel("Headline length (words)"); ax.set_ylabel("Count"); ax.legend(frameon=False)
    plt.savefig(fig("fig_text_length.png")); plt.close()


@torch.no_grad()
def attention_and_similarity():
    """How the proposed model splits attention, and how CLIP similarity differs real vs fake."""
    ck = os.path.join(C.CKPT_DIR, "xfnd_clip.pt")
    if not os.path.exists(ck):
        return None
    ck = torch.load(ck, map_location=C.DEVICE)
    model = build("xfnd_clip", *ck["sim_stats"]).to(C.DEVICE)
    model.load_state_dict(ck["state"]); model.eval()
    te = torch.load(os.path.join(C.FEATURE_DIR, "test.pt"))
    b = {k: te[k].to(C.DEVICE) for k in ("bert", "resnet", "clip_t", "clip_i", "has_image")}
    _, ex = model(b)
    att = ex["att"].cpu().numpy()
    sim = ex["sim"].cpu().numpy()
    hi, y = te["has_image"].numpy() == 1, te["label"].numpy()

    groups = {"Real, with image": hi & (y == 0), "Fake, with image": hi & (y == 1),
              "Real, no image": ~hi & (y == 0), "Fake, no image": ~hi & (y == 1)}
    groups = {k: v for k, v in groups.items() if v.sum() > 0}
    fig_, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(groups))
    for j, (nm, c) in enumerate(zip(["Text", "Image", "Text-image (CLIP)"], [BLUE, GREEN, ORANGE])):
        vals = np.array([att[m, j].mean() for m in groups.values()])
        ax.barh(list(groups), vals, left=left, color=c, label=nm)
        left += vals
    ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.set_xlabel("Average attention weight"); ax.set_title("Where XFND-CLIP looks (test set)", pad=22)
    plt.savefig(fig("fig_attention.png")); plt.close()

    if hi.sum() > 10:
        fig_, ax = plt.subplots(figsize=(5, 3))
        ax.hist(sim[hi & (y == 0)], bins=30, alpha=0.7, color=BLUE, label="Real", density=True)
        ax.hist(sim[hi & (y == 1)], bins=30, alpha=0.7, color=ORANGE, label="Fake", density=True)
        ax.set_xlabel("CLIP text-image cosine similarity"); ax.set_ylabel("Density")
        ax.legend(frameon=False); ax.set_title("Text-image agreement")
        plt.savefig(fig("fig_similarity.png")); plt.close()
    return {k: {"n": int(m.sum()), "att_text": float(att[m, 0].mean()),
                "att_image": float(att[m, 1].mean()), "att_mm": float(att[m, 2].mean()),
                "clip_sim": float(sim[m].mean())} for k, m in groups.items()}


def subgroup_table():
    rows = []
    for m in ("text_only", "concat", "fnd_clip", "xfnd_clip"):
        p = os.path.join(C.RESULTS_DIR, f"preds_{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for nm, part in (("with image", d[d.has_image == 1]), ("without image", d[d.has_image == 0])):
            if len(part):
                acc = ((part.prob_fake >= 0.5).astype(int) == part.label).mean()
                rows.append({"Model": LABELS[m], "Subset": nm, "n": len(part),
                             "Accuracy (%)": round(100 * acc, 2)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "subgroup_table.csv"), index=False)


def main():
    M = json.load(open(os.path.join(C.RESULTS_DIR, "metrics.json")))
    tables(M); comparison_plot(M); confusion_plots(); roc_plot(); curves_plot()
    ablation_plot(M); dataset_plots(); subgroup_table()
    att = attention_and_similarity()
    if att:
        json.dump(att, open(os.path.join(C.RESULTS_DIR, "attention_summary.json"), "w"), indent=2)
    print(pd.read_csv(os.path.join(C.RESULTS_DIR, "results_table.csv")).to_string(index=False))
    print("\nAblation:\n" + pd.read_csv(os.path.join(C.RESULTS_DIR, "ablation_table.csv")).to_string(index=False))
    print(f"\nFigures saved in {C.FIG_DIR}")


if __name__ == "__main__":
    main()
