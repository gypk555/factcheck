"""
Step 2: run every news item through the frozen encoders once and cache the
features on disk. Training then only touches these small tensors.

Output: data/features/{train,val,test}.pt
"""
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

import config as C
from encoders import FrozenEncoders, load_image

BS = 64


@torch.no_grad()
def extract(df, enc):
    n = len(df)
    out = {
        "bert": torch.zeros(n, C.BERT_DIM),
        "resnet": torch.zeros(n, C.RESNET_DIM),
        "clip_t": torch.zeros(n, C.CLIP_DIM),
        "clip_i": torch.zeros(n, C.CLIP_DIM),
        "has_image": torch.zeros(n),
        "label": torch.tensor(df.label.values, dtype=torch.long),
        "uid": torch.tensor(df.uid.values, dtype=torch.long),
    }
    texts = df.text.tolist()
    paths = df.image_path.fillna("").tolist()

    for s in tqdm(range(0, n, BS), desc="Encoding"):
        e = min(s + BS, n)
        bert, clip_t = enc.encode_text(texts[s:e])
        out["bert"][s:e] = bert.cpu()
        out["clip_t"][s:e] = clip_t.cpu()

        # images: encode only the ones that exist; missing ones stay zero
        imgs, idx = [], []
        for j in range(s, e):
            im = load_image(paths[j]) if paths[j] else None
            if im is not None:
                imgs.append(im)
                idx.append(j)
        if imgs:
            res, clip_i = enc.encode_images(imgs)
            out["resnet"][idx] = res.cpu()
            out["clip_i"][idx] = clip_i.cpu()
            out["has_image"][idx] = 1.0
    return out


def main():
    df = pd.read_csv(os.path.join(C.DATA_DIR, "processed.csv"))
    print(f"Loading encoders on {C.DEVICE} ...")
    enc = FrozenEncoders()
    for split in ("train", "val", "test"):
        part = df[df.split == split].reset_index(drop=True)
        t0 = time.time()
        feats = extract(part, enc)
        torch.save(feats, os.path.join(C.FEATURE_DIR, f"{split}.pt"))
        print(f"{split}: {len(part)} items, {int(feats['has_image'].sum())} with image, "
              f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
