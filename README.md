# XFND-CLIP: web interface

A website for **XFND-CLIP**, an explainable multimodal fake news detector for Indian news. You type a post's claim and add its picture; the model (running on a Colab GPU) says **FAKE** or **REAL** and shows why:

- **Which words mattered**: each word is removed in turn to see how the fake score changes
- **What the model relied on**: attention weights for text, image and text-image agreement
- **Does the picture match the claim?**: CLIP similarity against the training average
- **Which part of the image mattered**: Grad-CAM heatmap from ResNet-50

The site has four screens: **Check a post**, **How it works**, **Results** (test accuracy of 10 models, the 5-post live test, the most confident mistakes) and **History** (posts checked in this browser).

## How the pieces fit

```
 Website (Render, always on)  ──── HTTPS + access key ────►  Model server (Google Colab, T4 GPU)
 web/  static HTML/CSS/JS                                      colab/server.py  +  your model code
                                                               exposed with a temporary Cloudflare link
```

The website is static, so it is free and always online. The model is too big for free hosting, so it runs in Colab only while the notebook is open. Without Colab, the site still shows How it works, Results and History, and says the model is offline.

## Folder

| Path | What it is |
|---|---|
| `web/` | The website (no build step) |
| `web/data/` | Results from `XFND_extra.zip`: `metrics.json`, `live_test.csv`, `wrong_examples.csv`, figures |
| `colab/XFND_CLIP_Server.ipynb` | **The notebook to run on demo day** |
| `colab/server.py` | The model server (the notebook contains a copy of it) |
| `colab/build_notebook.py` | Regenerates the notebook after changing `server.py` |
| `model/` | The XFND-CLIP project code (from `XFND-CLIP.zip`); one change: the model file is loaded with `weights_only=True` |
| `dev/mock_server.py` | Local preview with a fake model, for working on the design |
| `tests/` | Automated checks |

## One-time setup

### 1. Put the site online (Render)
1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com): **New → Blueprint**, pick the repository. `render.yaml` does the rest.
3. Note the address Render gives you, e.g. `https://factcheck-bbfr.onrender.com`.
4. In `colab/build_notebook.py`, set `SITE_URL` to that address, run `uv run python colab/build_notebook.py`, and commit. (Or just edit `SITE_URL` in Box 4 inside Colab.)

### 2. Train once and save the model to Google Drive (Colab)
1. Open `colab/XFND_CLIP_Server.ipynb` in Google Colab with the Google account you'll use on demo day.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run **Box 1**: upload `XFND-CLIP.zip` when asked.
4. Run **Box 2**: upload `archive.zip` (the IFND dataset) when asked. It trains once (30–40 min) and saves `xfnd_clip.pt` to **Google Drive → XFND**.

Both zips and the trained model stay in Google Drive, so you never upload them again. The dataset never needs to be on your own computer.

## Demo day

1. Open the notebook in Colab (T4 GPU) and run **Boxes 1 to 4**. Boxes 1–2 take about a minute each now, because everything is loaded from Drive.
2. Wait for the green box. Copy the **connection code**, open the website, click **Connect**, paste it and press **Connect**. The sidebar shows **Model online**.
3. Keep the Colab tab open. When you're done, run **Box 5**.

Every run creates a new link and a new secret key. Old links stop working when Colab stops.

## Working on the website locally

```bash
uv sync
uv run python dev/mock_server.py      # prints the address and a connection code (a FAKE model answers)
uv run pytest -q                      # tests
```

The mock model invents numbers from keywords. It exists only to check the design.

## Security

- The Colab server only answers requests that carry the secret key from the notebook (a new one every run).
- Uploads: images only (JPG, PNG, WEBP, BMP, GIF), 10 MB and 24 megapixels at most; claims of 3 to 1,000 characters.
- The website has a strict Content-Security-Policy (no inline code, no third-party scripts) and inserts all text safely (never as HTML).
- History is kept in the visitor's own browser; nothing is stored on a server.
- The model file is loaded with `weights_only=True`, so it can only contain numbers, not code.

## Credits

- Base paper: Y. Zhou, Y. Yang, Q. Ying, Z. Qian, X. Zhang, "Multimodal Fake News Detection via CLIP-Guided Learning", IEEE ICME 2023.
- Dataset: D. K. Sharma, S. Garg, "IFND: a benchmark dataset for fake news detection", *Complex & Intelligent Systems* 9, 2023.
- Fonts: Atkinson Hyperlegible and Source Serif 4 (SIL Open Font License), self-hosted.
