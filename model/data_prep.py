"""
Step 1: load the raw dataset, clean it, download the news images and
make train / validation / test splits.

Output: data/processed.csv and results/data_stats.json

Usage:
    python data_prep.py                    # IFND (default)
    XFND_DATASET=fakeddit python data_prep.py
"""
import io
import json
import os
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config as C

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) XFND-CLIP research script"}


# ------------------------------------------------------------------ loading
def _pick(cols, *candidates):
    """Return the first column whose lower-cased name matches a candidate."""
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def load_ifnd(path):
    # The public IFND csv is not always utf-8 and some rows contain stray
    # commas, so try a few encodings and fall back to a forgiving parser.
    df = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            df = pd.read_csv(path, encoding=enc, engine="python", sep=None,
                             on_bad_lines="skip")
            break
    if df is None:
        df = pd.read_csv(path, encoding="latin-1", engine="python", sep=None, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"Read {len(df)} rows, columns: {list(df.columns)}")
    text_col = _pick(df.columns, "statement", "news headings", "headline", "title", "text")
    img_col = _pick(df.columns, "image", "image link", "image_url", "img")
    label_col = _pick(df.columns, "label", "labels", "class")
    cat_col = _pick(df.columns, "category", "topic")
    date_col = _pick(df.columns, "date")
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {list(df.columns)}. "
                         "Send this message to get the loader adjusted.")
    if img_col is None:
        print("WARNING: no image column found - this file has text only, so the image "
              "models will have nothing to learn from. Check you downloaded the full IFND csv.")

    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "image_url": df[img_col].astype(str) if img_col else "",
        "raw_label": df[label_col].astype(str).str.strip().str.lower(),
        "category": df[cat_col].astype(str) if cat_col else "NA",
        "date": df[date_col].astype(str) if date_col else "NA",
    })
    # IFND uses TRUE / Fake. We use 1 = fake, 0 = real.
    fake_words = {"fake", "false", "1", "f"}
    real_words = {"true", "real", "0", "t"}
    out = out[out.raw_label.isin(fake_words | real_words)].copy()
    out["label"] = out.raw_label.isin(fake_words).astype(int)
    return out.drop(columns="raw_label")


def load_fakeddit(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["hasImage"] == True]  # noqa: E712
    # Fakeddit: 2_way_label 1 = real, 0 = fake -> flip to our convention
    return pd.DataFrame({
        "text": df["clean_title"].astype(str),
        "image_url": df["image_url"].astype(str),
        "label": (df["2_way_label"] == 0).astype(int),
        "category": df.get("subreddit", "NA"),
        "date": df.get("created_utc", "NA"),
    })


# ------------------------------------------------------------------ cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(t):
    t = URL_RE.sub(" ", str(t))
    t = re.sub(r"<[^>]+>", " ", t)            # stray html tags
    t = re.sub(r"[^\x00-\x7F]+", " ", t)       # English-only: drop non-ascii junk
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------------------------------------------------------ images
def _img_path(url):
    return os.path.join(C.IMAGE_DIR, hashlib.md5(url.encode()).hexdigest() + ".jpg")


def download_one(url):
    """Download, verify and resize one image. Returns local path or ''."""
    if not isinstance(url, str) or not url.startswith("http"):
        return ""
    path = _img_path(url)
    if os.path.exists(path):
        return path
    try:
        r = requests.get(url, headers=HEADERS, timeout=C.IMG_TIMEOUT)
        if r.status_code != 200 or len(r.content) < 2000:   # tiny = placeholder/icon
            return ""
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 64:
            return ""
        img.thumbnail((448, 448))
        img.save(path, "JPEG", quality=90)
        return path
    except Exception:
        return ""


def download_images(urls):
    paths = [""] * len(urls)
    with ThreadPoolExecutor(C.IMG_DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(download_one, u): i for i, u in enumerate(urls)}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading images"):
            paths[futs[f]] = f.result()
    return paths


# ------------------------------------------------------------------ main
def main():
    stats = {"dataset": C.DATASET}
    if C.DATASET == "ifnd":
        df = load_ifnd(C.IFND_CSV)
    else:
        df = load_fakeddit(C.FAKEDDIT_TSV)
    stats["raw_rows"] = int(len(df))
    stats["raw_fake"] = int(df.label.sum())
    stats["raw_real"] = int((df.label == 0).sum())

    df["text"] = df.text.map(clean_text)
    df = df[df.text.str.split().str.len() >= C.MIN_TEXT_WORDS]
    df = df.drop_duplicates(subset="text")
    stats["after_cleaning"] = int(len(df))

    # balance: keep up to MAX_PER_CLASS of each class
    parts = []
    for lab in (0, 1):
        sub = df[df.label == lab]
        parts.append(sub.sample(min(len(sub), C.MAX_PER_CLASS), random_state=C.SEED))
    df = pd.concat(parts).sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    stats["sampled_rows"] = int(len(df))

    df["image_path"] = download_images(df.image_url.tolist())
    df["has_image"] = (df.image_path != "").astype(int)
    stats["images_downloaded"] = int(df.has_image.sum())
    stats["image_coverage_pct"] = round(100 * df.has_image.mean(), 2)

    df["uid"] = range(len(df))
    train, test = train_test_split(df, test_size=C.TEST_SIZE, stratify=df.label, random_state=C.SEED)
    train, val = train_test_split(train, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
                                  stratify=train.label, random_state=C.SEED)
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    for s in ("train", "val", "test"):
        part = df[df.split == s]
        stats[f"{s}_rows"] = int(len(part))
        stats[f"{s}_fake"] = int(part.label.sum())
        stats[f"{s}_with_image"] = int(part.has_image.sum())
    stats["category_counts"] = df.category.value_counts().head(10).to_dict()

    df.to_csv(os.path.join(C.DATA_DIR, "processed.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
