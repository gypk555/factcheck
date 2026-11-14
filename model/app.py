"""
Step 6: web demo.

Paste a news headline, optionally upload the image that came with it,
and the app shows the verdict together with the reasons behind it.
Every query is logged to results/demo_log.csv with a timestamp.

    python app.py            # local
    python app.py --share    # public link (use this on Colab)
"""
import argparse
import csv
import os
from datetime import datetime

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from explain import Explainer

EX = None
LOG = os.path.join(C.RESULTS_DIR, "demo_log.csv")


def get_explainer():
    global EX
    if EX is None:
        EX = Explainer()
    return EX


def modality_chart(p):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(["Text", "Image", "Text-image\nagreement"], [p["att_text"], p["att_image"], p["att_mm"]],
           color=["#2f5d8a", "#3c8d5a", "#d9822b"])
    ax.set_ylim(0, 1); ax.set_ylabel("weight")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def analyse(text, image):
    text = (text or "").strip()
    if len(text.split()) < 3:
        raise gr.Error("Please paste the full headline (at least 3 words).")
    ex = get_explainer()
    img = image.convert("RGB") if image is not None else None
    p = ex.predict(text, img)
    words = ex.word_importance(text, img)
    scale = max(1e-6, max(abs(v) for _, v in words))
    highlighted = [(w + " ", v / scale) for w, v in words]
    heat = ex.overlay(img, ex.gradcam(text, img)) if img is not None else None

    lines = [f"### Verdict: **{p['label']}**  (P(fake) = {p['prob_fake']:.2f})"]
    if p["clip_sim"] is not None:
        verdict = "matches" if p["clip_sim"] >= ex.sim_mean else "does not clearly match"
        lines.append(f"- Text–image similarity (CLIP): **{p['clip_sim']:.3f}** — the picture {verdict} the headline "
                     f"(training average {ex.sim_mean:.3f}).")
    else:
        lines.append("- No image given, so the decision is based on the text only.")
    top = sorted(words, key=lambda x: -x[1])[:3]
    pushers = ", ".join(f"`{w}`" for w, v in top if v > 0)
    lines.append("- Words pushing towards FAKE: " + (pushers or "none"))
    lines.append("\n*This is a research prototype. Always confirm with a fact-checker "
                 "such as PIB Fact Check before sharing.*")

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "text", "has_image", "prob_fake", "label", "clip_sim"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), text, img is not None,
                    round(p["prob_fake"], 4), p["label"], p["clip_sim"]])

    return ({"FAKE": p["prob_fake"], "REAL": 1 - p["prob_fake"]}, highlighted, heat,
            modality_chart(p), "\n".join(lines))


def ui():
    with gr.Blocks(title="XFND-CLIP: Explainable Fake News Detector") as demo:
        gr.Markdown("## XFND-CLIP — Explainable Multimodal Fake News Detection\n"
                    "Paste a news headline and (optionally) its image. The model says whether it looks "
                    "fake and shows *why*.")
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="News headline / post text", lines=4)
                image = gr.Image(label="Image shared with the news (optional)", type="pil")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Label(label="Prediction")
                notes = gr.Markdown()
        with gr.Row():
            words = gr.HighlightedText(label="Word importance (red = towards FAKE, blue = towards REAL)",
                                       color_map=None, show_legend=False)
        with gr.Row():
            heat = gr.Image(label="Grad-CAM: image regions that drove the decision")
            chart = gr.Plot(label="Modality weights")
        btn.click(analyse, [text, image], [verdict, words, heat, chart, notes])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ui().launch(share=ap.parse_args().share)
