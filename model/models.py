"""
Classification networks that sit on top of the cached features.

  TextOnly      - BERT feature only
  ImageOnly     - ResNet feature only
  ConcatFusion  - BERT + ResNet simply concatenated (early fusion)
  FNDCLIP       - our re-implementation of the base paper (Zhou et al., IEEE ICME 2023)
  XFNDCLIP      - proposed model: FNDCLIP + cross-modal consistency features,
                  image-availability aware attention, and explanation outputs

Every model returns (logits, extras). `extras` is a dict that the
explanation code reads (attention scores, CLIP similarity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


def mlp(inp, out, hidden=C.HIDDEN, drop=C.DROPOUT):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop),
        nn.Linear(hidden, out),
    )


def proj(inp, out=C.HIDDEN, drop=C.DROPOUT):
    return nn.Sequential(nn.Linear(inp, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(drop))


# ------------------------------------------------------------------ baselines
class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = mlp(C.BERT_DIM, 2)

    def forward(self, b):
        return self.head(b["bert"]), {}


class ImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = mlp(C.RESNET_DIM, 2)

    def forward(self, b):
        return self.head(b["resnet"]), {}


class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = mlp(C.BERT_DIM + C.RESNET_DIM, 2)

    def forward(self, b):
        return self.head(torch.cat([b["bert"], b["resnet"]], -1)), {}


# ------------------------------------------------------------------ base paper
class FNDCLIP(nn.Module):
    """
    Re-implementation of FND-CLIP (Zhou et al., IEEE ICME 2023):
      - text branch  : BERT  -> projection
      - image branch : ResNet-> projection
      - multimodal   : [CLIP_text ; CLIP_image] scaled by the standardised
                       CLIP similarity, then projected
      - modality-wise attention gives one weight per branch
      - weighted branches are concatenated and classified
    """

    def __init__(self, sim_mean=0.25, sim_std=0.05):
        super().__init__()
        self.register_buffer("sim_mean", torch.tensor(float(sim_mean)))
        self.register_buffer("sim_std", torch.tensor(float(sim_std)))
        self.p_text = proj(C.BERT_DIM)
        self.p_img = proj(C.RESNET_DIM)
        self.p_mm = proj(2 * C.CLIP_DIM)
        self.att = nn.Sequential(nn.Linear(3 * C.HIDDEN, C.HIDDEN // 2), nn.ReLU(),
                                 nn.Linear(C.HIDDEN // 2, 3))
        self.cls = mlp(3 * C.HIDDEN, 2)

    def similarity_weight(self, b):
        sim = (b["clip_t"] * b["clip_i"]).sum(-1)                 # cosine (both L2-normalised)
        w = torch.sigmoid((sim - self.sim_mean) / self.sim_std)   # standardised -> (0,1)
        return sim, w * b["has_image"]                            # no image -> no multimodal signal

    def branches(self, b):
        sim, w = self.similarity_weight(b)
        t = self.p_text(b["bert"])
        i = self.p_img(b["resnet"]) * b["has_image"].unsqueeze(-1)
        m = self.p_mm(torch.cat([b["clip_t"], b["clip_i"]], -1) * w.unsqueeze(-1))
        return t, i, m, sim, w

    def forward(self, b):
        t, i, m, sim, w = self.branches(b)
        a = torch.sigmoid(self.att(torch.cat([t, i, m], -1)))     # (B,3)
        z = torch.cat([t * a[:, :1], i * a[:, 1:2], m * a[:, 2:]], -1)
        return self.cls(z), {"att": a, "sim": sim, "sim_w": w}


# ------------------------------------------------------------------ proposed
class XFNDCLIP(FNDCLIP):
    """
    Proposed model. Changes over FND-CLIP:
      1. Cross-modal consistency features: |t - i| and t * i of the CLIP
         embeddings are fed to the multimodal branch, so the network sees
         *where* text and image disagree, not only a single similarity score.
      2. Availability-aware attention: the attention network also receives the
         has_image flag, so it learns to lean on text when the image is
         missing (common in IFND because many image links are dead).
      3. Softmax attention, so the three scores add up to 1 and can be read
         directly as "how much each modality contributed" in the explanation.
    The loss (class-balanced focal loss) is set in train.py.

    The flags let train.py run the ablation study.
    """

    def __init__(self, sim_mean=0.25, sim_std=0.05, use_consistency=True,
                 use_avail_att=True, use_attention=True):
        super().__init__(sim_mean, sim_std)
        self.use_consistency = use_consistency
        self.use_avail_att = use_avail_att
        self.use_attention = use_attention
        mm_in = (4 if use_consistency else 2) * C.CLIP_DIM
        self.p_mm = proj(mm_in)
        att_in = 3 * C.HIDDEN + (1 if use_avail_att else 0)
        self.att = nn.Sequential(nn.Linear(att_in, C.HIDDEN // 2), nn.ReLU(),
                                 nn.Linear(C.HIDDEN // 2, 3))

    def branches(self, b):
        sim, w = self.similarity_weight(b)
        ct, ci = b["clip_t"], b["clip_i"]
        mm = [ct, ci]
        if self.use_consistency:
            mm += [(ct - ci).abs(), ct * ci]
        t = self.p_text(b["bert"])
        i = self.p_img(b["resnet"]) * b["has_image"].unsqueeze(-1)
        m = self.p_mm(torch.cat(mm, -1) * w.unsqueeze(-1))
        return t, i, m, sim, w

    def forward(self, b):
        t, i, m, sim, w = self.branches(b)
        if self.use_attention:
            att_in = [t, i, m]
            if self.use_avail_att:
                att_in.append(b["has_image"].unsqueeze(-1))
            logits = self.att(torch.cat(att_in, -1))
            if self.use_avail_att:
                # an absent image cannot carry weight
                mask = torch.stack([torch.ones_like(b["has_image"]), b["has_image"],
                                    b["has_image"]], -1)
                logits = logits.masked_fill(mask == 0, -1e4)
            a = torch.softmax(logits, -1)
        else:
            a = torch.full((t.size(0), 3), 1 / 3, device=t.device)
        z = torch.cat([t * a[:, :1], i * a[:, 1:2], m * a[:, 2:]], -1)
        return self.cls(z), {"att": a, "sim": sim, "sim_w": w}


# ------------------------------------------------------------------ loss
class FocalLoss(nn.Module):
    """Class-balanced focal loss (Lin et al., 2017) for the class imbalance in IFND."""

    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, -1)
        logp_t = logp.gather(1, y.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        return (-self.alpha[y] * (1 - p_t) ** self.gamma * logp_t).mean()


def build(name, sim_mean=0.25, sim_std=0.05):
    if name == "text_only":
        return TextOnly()
    if name == "image_only":
        return ImageOnly()
    if name == "concat":
        return ConcatFusion()
    if name == "fnd_clip":
        return FNDCLIP(sim_mean, sim_std)
    if name == "xfnd_clip":
        return XFNDCLIP(sim_mean, sim_std)
    if name == "xfnd_no_consistency":
        return XFNDCLIP(sim_mean, sim_std, use_consistency=False)
    if name == "xfnd_no_availability":
        return XFNDCLIP(sim_mean, sim_std, use_avail_att=False)
    if name == "xfnd_no_attention":
        return XFNDCLIP(sim_mean, sim_std, use_attention=False)
    if name == "xfnd_ce_loss":
        return XFNDCLIP(sim_mean, sim_std)
    raise ValueError(name)
