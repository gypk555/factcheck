# XFND-CLIP: Explainable Multimodal Fake News Detection on Indian News

M.Tech (CSE) mini project. The model reads a news headline together with the image
that was shared with it, decides whether the post is fake, and explains the decision:
which words mattered, which part of the image mattered, how much it trusted each
modality, and whether the picture actually matches the headline.

**Base paper:** Y. Zhou, Y. Yang, Q. Ying, Z. Qian and X. Zhang, "Multimodal Fake News
Detection via CLIP-Guided Learning," *2023 IEEE International Conference on Multimedia
and Expo (ICME)*, Brisbane, Australia, 2023. IEEE Xplore document 10219997.

**Dataset:** IFND — D. K. Sharma and S. Garg, "IFND: a benchmark dataset for fake
news detection," *Complex & Intelligent Systems*, 9, 2843–2863, 2023.

## Project structure

| File | What it does |
|---|---|
| `config.py` | All paths and hyper-parameters in one place |
| `data_prep.py` | Loads IFND, cleans text, balances classes, downloads images, splits 70/15/15 |
| `encoders.py` | Frozen BERT, ResNet-50 and CLIP ViT-B/32 encoders |
| `features.py` | Encodes every item once and caches the features |
| `models.py` | Baselines, FND-CLIP (base paper) and XFND-CLIP (proposed) |
| `train.py` | Trains every model with 3 seeds, saves metrics and predictions |
| `evaluate.py` | Builds the result tables and all figures |
| `explain.py` | Word importance, Grad-CAM, modality weights, CLIP similarity |
| `app.py` | Gradio web demo with prediction logging |
| `XFND_CLIP_Colab.ipynb` | Runs the whole pipeline on Google Colab |

## How to run (Google Colab, free T4 GPU)

1. Upload `XFND-CLIP.zip` to the root of Google Drive.
2. Open `XFND_CLIP_Colab.ipynb` in Colab and pick a T4 GPU runtime.
3. Run the cells in order. You will be asked to upload `IFND.csv`.

## How to run (own machine with a GPU)

```bash
pip install -r requirements.txt
# put IFND.csv in data/
python data_prep.py
python features.py
python train.py
python evaluate.py
python explain.py --from_test 6
python app.py
```

To use Fakeddit instead, put `multimodal_train.tsv` in `data/` and prefix the
commands with `XFND_DATASET=fakeddit`.

## What is new compared with the base paper

1. **Cross-modal consistency features** — the element-wise difference and product of
   the CLIP text and image embeddings go into the multimodal branch.
2. **Availability-aware attention** — the attention network knows whether an image
   exists and cannot give weight to a missing one. This matters for IFND, where many
   image links no longer work.
3. **Class-balanced focal loss** for the remaining class imbalance.
4. **Explanation module** — the base paper only outputs real/fake; this project
   shows *why*.
