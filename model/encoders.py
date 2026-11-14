"""
Frozen pre-trained encoders used by the project.

  * BERT (bert-base-uncased)      -> 768-d text feature   (CLS token)
  * ResNet-50 (ImageNet)          -> 2048-d image feature (global avg pool)
  * CLIP ViT-B/32 text + image    -> 512-d each, in a shared space

All encoders stay frozen. Only the small fusion network in models.py is
trained, which keeps training fast enough for a free Colab GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor

import config as C

RESNET_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FrozenEncoders(nn.Module):
    def __init__(self, device=C.DEVICE):
        super().__init__()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(C.BERT_NAME)
        self.bert = AutoModel.from_pretrained(C.BERT_NAME)

        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        rn.fc = nn.Identity()
        self.resnet = rn

        self.clip = CLIPModel.from_pretrained(C.CLIP_NAME)
        self.clip_proc = CLIPProcessor.from_pretrained(C.CLIP_NAME)

        for p in self.parameters():
            p.requires_grad = False
        self.eval().to(device)

    # ---------------- text ----------------
    def encode_text(self, texts):
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=C.MAX_TEXT_LEN, return_tensors="pt").to(self.device)
        bert = self.bert(**enc).last_hidden_state[:, 0]          # CLS
        cenc = self.clip_proc(text=texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=77).to(self.device)
        clip_t = self._as_tensor(self.clip.get_text_features(**cenc))
        return bert, F.normalize(clip_t, dim=-1)

    # ---------------- images ----------------
    def resnet_input(self, imgs):
        return torch.stack([RESNET_TF(im) for im in imgs]).to(self.device)

    def clip_input(self, imgs):
        return self.clip_proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)

    def encode_images(self, imgs):
        res = self.resnet(self.resnet_input(imgs))
        clip_i = self._as_tensor(self.clip.get_image_features(pixel_values=self.clip_input(imgs)))
        return res, F.normalize(clip_i, dim=-1)

    @staticmethod
    def _as_tensor(out):
        # newer transformers versions may return an output object instead of a tensor
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            if getattr(out, attr, None) is not None:
                return getattr(out, attr)
        return out[0]


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None
