"""
Step 5: explanation module of XFND-CLIP.

For one news post (headline + optional image) it gives:
  1. the verdict and confidence
  2. word importance  - drop each word, see how much P(fake) changes
  3. Grad-CAM heatmap - which image regions pushed the decision
  4. modality weights - how much text / image / text-image agreement counted
  5. CLIP similarity  - does the picture actually match the headline?

Usage:
  python explain.py --text "Headline here" --image photo.jpg
  python explain.py --from_test 6        # explain 6 test items for the report
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from encoders import FrozenEncoders, load_image
from models import build


class Explainer:
    def __init__(self, ckpt="xfnd_clip", encoders=None):
        self.enc = encoders or FrozenEncoders()
        ck = torch.load(os.path.join(C.CKPT_DIR, f"{ckpt}.pt"), map_location=C.DEVICE,
                        weights_only=True)  # only tensors/numbers, never code
        self.model = build(ck["name"], *ck["sim_stats"]).to(C.DEVICE).eval()
        self.model.load_state_dict(ck["state"])
        self.sim_mean = ck["sim_stats"][0]

    # ------------------------------------------------------------ core
    @torch.no_grad()
    def _image_feats(self, img):
        if img is None:
            return torch.zeros(1, C.RESNET_DIM, device=C.DEVICE), \
                   torch.zeros(1, C.CLIP_DIM, device=C.DEVICE), torch.zeros(1, device=C.DEVICE)
        res, ci = self.enc.encode_images([img])
        return res, ci, torch.ones(1, device=C.DEVICE)

    @torch.no_grad()
    def _probs(self, texts, res, ci, has):
        bert, ct = self.enc.encode_text(texts)
        n = len(texts)
        b = {"bert": bert, "clip_t": ct, "resnet": res.expand(n, -1),
             "clip_i": ci.expand(n, -1), "has_image": has.expand(n)}
        logits, ex = self.model(b)
        return torch.softmax(logits, -1)[:, 1], ex

    def predict(self, text, img=None):
        res, ci, has = self._image_feats(img)
        p, ex = self._probs([text], res, ci, has)
        att = ex["att"][0].tolist()
        return {"prob_fake": float(p[0]), "label": "FAKE" if p[0] >= 0.5 else "REAL",
                "att_text": att[0], "att_image": att[1], "att_mm": att[2],
                "clip_sim": float(ex["sim"][0]) if img is not None else None}

    def word_importance(self, text, img=None):
        """Occlusion: P(fake | full text) - P(fake | text without word i)."""
        words = text.split()
        if len(words) < 2:
            return [(w, 0.0) for w in words]
        res, ci, has = self._image_feats(img)
        variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        p, _ = self._probs(variants, res, ci, has)
        base = p[0].item()
        return [(w, base - p[i + 1].item()) for i, w in enumerate(words)]

    def gradcam(self, text, img):
        """Grad-CAM on ResNet-50 layer4 w.r.t. the 'fake' logit."""
        acts, grads = {}, {}
        layer = self.enc.resnet.layer4
        h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        try:
            with torch.no_grad():
                bert, ct = self.enc.encode_text([text])
                ci = self.enc._as_tensor(self.enc.clip.get_image_features(
                    pixel_values=self.enc.clip_input([img])))
                ci = F.normalize(ci, dim=-1)
            x = self.enc.resnet_input([img]).requires_grad_(True)
            res = self.enc.resnet(x)
            b = {"bert": bert, "clip_t": ct, "resnet": res, "clip_i": ci,
                 "has_image": torch.ones(1, device=C.DEVICE)}
            logits, _ = self.model(b)
            self.model.zero_grad()
            logits[0, 1].backward()
            a, g = acts["a"][0], grads["g"][0]                    # (2048,7,7)
            raw = (g.mean((1, 2))[:, None, None] * a).sum(0)
            cam = F.relu(raw)
            if cam.max() <= 0:          # every region argues against "fake": show magnitude instead
                cam = raw.abs()
            cam = cam / (cam.max() + 1e-8)
        finally:
            h1.remove(); h2.remove()
        cam = cam.detach().cpu().numpy()
        # undo the centre crop: map the 224 crop back onto a 256-resized frame
        w, h = img.size
        s = 256 / min(w, h)
        rw, rh = round(w * s), round(h * s)
        full = np.zeros((rh, rw), dtype=np.float32)
        top, left = (rh - 224) // 2, (rw - 224) // 2
        full[top:top + 224, left:left + 224] = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
        return np.array(Image.fromarray((full * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0

    @staticmethod
    def overlay(img, cam, alpha=0.45):
        heat = plt.get_cmap("jet")(cam)[..., :3]
        base = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        return Image.fromarray((255 * ((1 - alpha) * base + alpha * heat)).astype(np.uint8))

    # ------------------------------------------------------------ report figure
    def figure(self, text, img, out_path, true_label=None):
        pred = self.predict(text, img)
        wi = self.word_importance(text, img)
        ncol = 3 if img is not None else 2
        fig = plt.figure(figsize=(4.2 * ncol, 4.2))
        k = 1
        if img is not None:
            cam = self.gradcam(text, img)
            ax = fig.add_subplot(1, ncol, k); k += 1
            ax.imshow(self.overlay(img, cam)); ax.axis("off")
            ax.set_title(f"Grad-CAM (CLIP sim {pred['clip_sim']:.2f})", fontsize=9)
        ax = fig.add_subplot(1, ncol, k); k += 1
        words, vals = zip(*wi[:20])
        colors = ["#c0392b" if v > 0 else "#2f5d8a" for v in vals]
        ax.barh(range(len(words)), vals, color=colors)
        ax.set_yticks(range(len(words)), words, fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title("Word importance (red pushes to FAKE)", fontsize=9)
        ax = fig.add_subplot(1, ncol, k)
        ax.bar(["Text", "Image", "Text-image"], [pred["att_text"], pred["att_image"], pred["att_mm"]],
               color=["#2f5d8a", "#3c8d5a", "#d9822b"])
        ax.set_ylim(0, 1); ax.set_title("Modality weights", fontsize=9)
        title = f"Prediction: {pred['label']} (P(fake) = {pred['prob_fake']:.2f})"
        if true_label is not None:
            title += f"   |   Ground truth: {'FAKE' if true_label else 'REAL'}"
        short = text if len(text) < 110 else text[:107] + "..."
        fig.suptitle(f"{title}\n\"{short}\"", fontsize=9.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        return pred


def explain_test_items(n):
    """Pick a mix of correct and wrong predictions from the test set."""
    ex = Explainer()
    df = pd.read_csv(os.path.join(C.DATA_DIR, "processed.csv"))
    preds = pd.read_csv(os.path.join(C.RESULTS_DIR, "preds_xfnd_clip.csv"))
    d = preds.merge(df[["uid", "text", "image_path"]], on="uid")
    d = d[d.image_path.notna() & (d.image_path != "")]
    d["pred"] = (d.prob_fake >= 0.5).astype(int)
    picks = pd.concat([
        d[(d.label == 1) & (d.pred == 1)].sort_values("prob_fake", ascending=False).head(n // 3 + 1),
        d[(d.label == 0) & (d.pred == 0)].sort_values("prob_fake").head(n // 3 + 1),
        d[d.label != d.pred].head(n // 3),
    ]).head(n)
    out_dir = os.path.join(C.FIG_DIR, "explanations")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(picks.itertuples()):
        img = load_image(r.image_path)
        tag = "correct" if r.label == r.pred else "wrong"
        ex.figure(r.text, img, os.path.join(out_dir, f"explain_{i + 1}_{tag}.png"), r.label)
    print(f"Saved {len(picks)} explanation figures to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text"); ap.add_argument("--image")
    ap.add_argument("--from_test", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.FIG_DIR, "explain_custom.png"))
    a = ap.parse_args()
    if a.from_test:
        explain_test_items(a.from_test)
    elif a.text:
        e = Explainer()
        img = load_image(a.image) if a.image else None
        print(e.figure(a.text, img, a.out))
        print("saved", a.out)
    else:
        ap.print_help()
