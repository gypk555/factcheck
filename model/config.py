"""
Central settings for the XFND-CLIP project.
Change values here instead of editing the individual scripts.
"""
import os
import torch

# ---------------- paths ----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# raw dataset files (put the CSV/TSV you download here)
IFND_CSV = os.path.join(DATA_DIR, "IFND.csv")
FAKEDDIT_TSV = os.path.join(DATA_DIR, "multimodal_train.tsv")

# "ifnd" (default, Indian news) or "fakeddit" (fallback)
DATASET = os.environ.get("XFND_DATASET", "ifnd")

# ---------------- data ----------------
# IFND has ~37.8k real and ~7.3k fake items. We keep every fake item and
# sample real items so training is not swamped by the majority class.
MAX_PER_CLASS = int(os.environ.get("XFND_MAX_PER_CLASS", 8000))
MIN_TEXT_WORDS = 4            # drop headlines that are too short to judge
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42
IMG_DOWNLOAD_WORKERS = 32
IMG_TIMEOUT = 10              # seconds per image request

# ---------------- backbones ----------------
BERT_NAME = "bert-base-uncased"
CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_TEXT_LEN = 64             # headlines are short, 64 tokens is plenty

# feature sizes produced by the frozen encoders
BERT_DIM = 768
RESNET_DIM = 2048
CLIP_DIM = 512

# ---------------- training ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 256
DROPOUT = 0.3
PATIENCE = 6                  # early stopping on validation macro-F1

for _d in (DATA_DIR, IMAGE_DIR, FEATURE_DIR, RESULTS_DIR, FIG_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
