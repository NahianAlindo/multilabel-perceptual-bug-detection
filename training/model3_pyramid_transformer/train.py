"""
Model 3: Feature Pyramid + Transformer
Swin-S (timm) + TransformerEncoder + 1D multi-scale FPN + anchor-free detection head.
ActionFormer concept (Zhang et al., ECCV 2022), implemented with standard PyTorch + timm.

Modes:
  --mode train   HPO (Optuna TPE) → full training → multi-task early stopping → auto test
  --mode infer   Single video → result.json (backend-compatible)
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARN] timm not installed — backbone will fall back to ResNet18")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] optuna not installed — HPO disabled")

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

from sklearn.metrics import (
    average_precision_score, f1_score, hamming_loss,
    roc_auc_score, label_ranking_average_precision_score,
    precision_score, recall_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ─────────────────────────────────────────────────────────────────
CANON_BUG_TYPES = [
    "z-clipping", "corrupted_texture",
    "geometry_corruption", "z-fighting", "boundary_hole"
]
NUM_CLASSES = len(CANON_BUG_TYPES)
SEED = 42

# ── helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def norm_bug_type(s: str) -> str:
    # Canonical forms mix hyphens and underscores ("z-clipping" but
    # "corrupted_texture"), so blanket replacement would break 3 of 5 classes.
    s2 = s.strip().lower().replace(" ", "_").replace("-", "_")
    if s2 == "z_clipping": return "z-clipping"
    if s2 == "z_fighting": return "z-fighting"
    return s2

def load_frames_decord(video_path: str, start_sec: float, end_sec: float,
                        fps: float, img_size: int) -> Optional[torch.Tensor]:
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        native_fps = vr.get_avg_fps()
        start_f = int(start_sec * native_fps)
        end_f   = int(end_sec   * native_fps)
        n_frames = max(1, int((end_sec - start_sec) * fps))
        indices  = np.linspace(start_f, max(start_f, end_f - 1), n_frames, dtype=int)
        indices  = np.clip(indices, 0, len(vr) - 1)
        frames   = vr.get_batch(indices).asnumpy()  # [T,H,W,C]
        out = []
        for f in frames:
            f = cv2.resize(f, (img_size, img_size)) if CV2_AVAILABLE else \
                np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(f).permute(2,0,1).float().unsqueeze(0)/255.,
                    size=(img_size, img_size)
                ).squeeze(0).permute(1,2,0).numpy() * 255, dtype=np.uint8)
            out.append(torch.from_numpy(f).permute(2,0,1).float() / 255.)
        frames_t = torch.stack(out)  # [T,3,H,W]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        return (frames_t - mean) / std
    except Exception as e:
        print(f"[WARN] decord failed for {video_path}: {e}")
        return None

def load_frames_cv2(video_path: str, start_sec: float, end_sec: float,
                     fps: float, img_size: int) -> Optional[torch.Tensor]:
    try:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = int(start_sec * native_fps)
        end_f   = int(end_sec   * native_fps)
        n_frames = max(1, int((end_sec - start_sec) * fps))
        indices  = np.linspace(start_f, max(start_f, end_f - 1), n_frames, dtype=int)
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (img_size, img_size))
            out.append(torch.from_numpy(frame).permute(2,0,1).float() / 255.)
        cap.release()
        frames_t = torch.stack(out)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        return (frames_t - mean) / std
    except Exception as e:
        print(f"[WARN] cv2 failed for {video_path}: {e}")
        return None

def load_frames(video_path: str, start_sec: float, end_sec: float,
                fps: float, img_size: int) -> torch.Tensor:
    frames = None
    if DECORD_AVAILABLE:
        frames = load_frames_decord(video_path, start_sec, end_sec, fps, img_size)
    if frames is None and CV2_AVAILABLE:
        frames = load_frames_cv2(video_path, start_sec, end_sec, fps, img_size)
    if frames is None:
        n = max(1, int((end_sec - start_sec) * fps))
        frames = torch.zeros(n, 3, img_size, img_size)
    return frames

# ── frame cache ───────────────────────────────────────────────────────────────
# Each video is decoded ONCE at (fps, img_size) and stored as a uint8 .npy array
# [T,H,W,3]; windows are then read as memory-mapped slices instead of opening
# and seek-decoding the video for every window sample.

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frame_cache_name(rel_video_path: str) -> str:
    key = rel_video_path.replace("\\", "/").strip("/")
    key = os.path.splitext(key)[0].replace("/", "__")
    return key + ".npy"


@lru_cache(maxsize=None)
def open_frame_cache(path: str):
    return np.load(path, mmap_mode="r")


def extract_frames_at_fps(video_path: str, target_fps: float = 8.0,
                          img_size: int = 224) -> np.ndarray:
    """Decode a full video once at target_fps, resized. Returns uint8 [T,H,W,3]."""
    if DECORD_AVAILABLE:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=img_size, height=img_size)
            native_fps = float(vr.get_avg_fps()) or 25.0
            stride = max(1, round(native_fps / target_fps))
            indices = list(range(0, len(vr), stride))
            batch = vr.get_batch(indices)
            frames = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.numpy()
            return frames.astype(np.uint8)
        except Exception:
            pass
    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, round(native_fps / target_fps))
        frames_list, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
                frames_list.append(frame)
            idx += 1
        cap.release()
        if frames_list:
            return np.stack(frames_list).astype(np.uint8)
    raise RuntimeError("No video backend available. Install decord or opencv.")


def _cache_one_video(task):
    video_path, out_path, fps, img_size = task
    out_path = Path(out_path)
    if out_path.exists():
        return (out_path.name, "cached")
    try:
        frames = extract_frames_at_fps(video_path, fps, img_size)
        tmp = out_path.parent / (out_path.stem + f".tmp_{os.getpid()}.npy")
        np.save(str(tmp), frames)
        os.replace(str(tmp), str(out_path))  # atomic: no partial files on job kill
        return (out_path.name, "ok")
    except Exception as e:
        return (out_path.name, f"fail: {e}")


def ensure_frame_cache(video_list, video_root, cache_dir, fps=8.0, img_size=224, workers=4):
    """Pre-extract any missing videos into the cache. Skips already-extracted
    videos, so it is safe to call at the start of every run."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = {"fps": fps, "img_size": img_size}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        if existing != meta:
            raise RuntimeError(f"Frame cache {cache_dir} was built with {existing}, "
                               f"but {meta} was requested. Use a different --frame-cache-dir.")
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    video_root = Path(video_root)
    tasks = []
    for vid in video_list:
        vp = video_root / vid["video_path"]
        if not vp.exists():
            continue
        out = cache_dir / frame_cache_name(vid["video_path"])
        if not out.exists():
            tasks.append((str(vp), str(out), fps, img_size))
    if not tasks:
        print(f"[cache] Frame cache complete for {len(video_list)} videos — skipping extraction.")
        return
    print(f"[cache] Extracting {len(tasks)}/{len(video_list)} missing videos to {cache_dir} "
          f"with {workers} workers (one-time cost; future runs skip this)...", flush=True)
    t0 = time.time()
    n_fail = 0
    with Pool(processes=max(1, workers)) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(_cache_one_video, tasks), 1):
            if status.startswith("fail"):
                n_fail += 1
                print(f"[cache] {name}: {status}")
            if i % 10 == 0 or i == len(tasks):
                print(f"[cache] {i}/{len(tasks)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[cache] Finished in {(time.time() - t0) / 60:.1f} min | failed: {n_fail}")

# ── losses ────────────────────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    """Ridnik et al., ICCV 2021."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets       * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt) ** gamma
        return -loss.mean()


def diou_loss_temporal(pred_starts: torch.Tensor, pred_ends: torch.Tensor,
                        gt_starts: torch.Tensor, gt_ends: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Distance-IoU loss for temporal segments (Zheng et al., AAAI 2020).
    All inputs are absolute times in seconds, shape [N].
    """
    eps = 1e-6
    inter_s = torch.max(pred_starts, gt_starts)
    inter_e = torch.min(pred_ends,   gt_ends)
    inter   = (inter_e - inter_s).clamp(min=0)
    union   = (pred_ends - pred_starts).clamp(min=eps) + \
              (gt_ends   - gt_starts  ).clamp(min=eps) - inter
    iou     = inter / union.clamp(min=eps)

    c_s = torch.min(pred_starts, gt_starts)
    c_e = torch.max(pred_ends,   gt_ends)
    c   = (c_e - c_s).clamp(min=eps)

    pred_c = (pred_starts + pred_ends) / 2
    gt_c   = (gt_starts   + gt_ends)   / 2
    d2     = (pred_c - gt_c) ** 2
    diou   = iou - d2 / (c ** 2 + eps)

    loss = 1 - diou
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp(min=1)
        return loss.sum() / denom
    return loss.mean()


def centerness_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE for centerness score."""
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def compute_centerness_target(gt_start: torch.Tensor, gt_end: torch.Tensor,
                               pos_t: torch.Tensor) -> torch.Tensor:
    """FCOS-style centerness: sqrt(min(l,r)/max(l,r)) where l,r are distances to segment boundaries."""
    l = (pos_t - gt_start).clamp(min=0)
    r = (gt_end  - pos_t ).clamp(min=0)
    centerness = torch.sqrt(
        torch.min(l, r) / (torch.max(l, r).clamp(min=1e-6))
    ).clamp(0, 1)
    return centerness

# ── backbone ──────────────────────────────────────────────────────────────────
def build_backbone(name: str = "swin_small_patch4_window7_224") -> Tuple[nn.Module, int]:
    """Returns (backbone, feature_dim)."""
    if TIMM_AVAILABLE:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = backbone(dummy).shape[-1]
        return backbone, feat_dim
    # fallback
    import torchvision.models as tvm
    resnet = tvm.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    return backbone, 512

# ── 1D dilated convolution block ──────────────────────────────────────────────
class DilatedConv1dBlock(nn.Module):
    def __init__(self, d_model: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        residual = x
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.relu(x)
        x = self.drop(x)
        return x + residual

# ── anchor-free detection head ────────────────────────────────────────────────
class AnchorFreeHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, 2, 1)  # (dist_to_start, dist_to_end)
        )
        self.ctr_head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, 1), nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, 1)
        )

    def forward(self, feat):  # feat: [B, C, T]
        cls = self.cls_head(feat)   # [B, num_classes, T]
        reg = F.relu(self.reg_head(feat))  # [B, 2, T] — non-negative distances
        ctr = self.ctr_head(feat)   # [B, 1, T]
        return cls, reg, ctr

# ── main model ────────────────────────────────────────────────────────────────
class PyramidTransformerLocalization(nn.Module):
    """
    Swin-S backbone + position-encoded TransformerEncoder +
    1D multi-scale FPN (dilations {1,2,4,8}) + anchor-free head per scale.
    """
    def __init__(self,
                 backbone_name: str = "swin_small_patch4_window7_224",
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = NUM_CLASSES,
                 fpn_dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_dilations = fpn_dilations

        self.backbone, feat_dim = build_backbone(backbone_name)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # FPN: one dilated conv block per scale, shared input from transformer
        self.fpn_convs = nn.ModuleList([
            DilatedConv1dBlock(d_model, dil, dropout) for dil in fpn_dilations
        ])

        # One head per FPN scale
        self.heads = nn.ModuleList([
            AnchorFreeHead(d_model, num_classes, dropout) for _ in fpn_dilations
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.feat_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _positional_encoding(self, T: int, d_model: int, device) -> torch.Tensor:
        pe = torch.zeros(T, d_model, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)  # [1, T, d_model]

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] → [B, T, feat_dim]"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        with torch.cuda.amp.autocast(enabled=False):
            feats = self.backbone(flat.float())  # [B*T, feat_dim]
        feats = feats.view(B, T, -1)
        return self.feat_proj(feats)  # [B, T, d_model]

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, 3, H, W]
        Returns list of (cls, reg, ctr) per FPN scale, each:
          cls: [B, num_classes, T]
          reg: [B, 2, T]
          ctr: [B, 1, T]
        """
        feats = self.extract_frame_features(frames)  # [B, T, d_model]
        B, T, d = feats.shape

        pe = self._positional_encoding(T, d, feats.device)
        feats = feats + pe  # [B, T, d_model]

        feats = self.transformer(feats)  # [B, T, d_model]

        feats_1d = feats.transpose(1, 2)  # [B, d_model, T]

        outputs = []
        for fpn_conv, head in zip(self.fpn_convs, self.heads):
            scale_feat = fpn_conv(feats_1d)  # [B, d_model, T]
            cls, reg, ctr = head(scale_feat)
            outputs.append((cls, reg, ctr))

        return outputs  # list of (cls, reg, ctr) per scale


# ── dataset ───────────────────────────────────────────────────────────────────
class TemporalWindowDataset(Dataset):
    """
    Sliding window dataset: each sample is a window of frames with
    per-frame classification labels and segment regression targets.
    """
    def __init__(self,
                 video_list: List[Dict],
                 video_root: str,
                 fps: float = 8.0,
                 img_size: int = 224,
                 window_sec: float = 4.0,
                 stride_sec: float = 1.0,
                 split: str = "train",
                 frame_cache_dir: Optional[str] = None):
        self.video_root = video_root
        self.fps = fps
        self.img_size = img_size
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.split = split
        self.frame_cache_dir = frame_cache_dir
        self.class_to_idx = {c: i for i, c in enumerate(CANON_BUG_TYPES)}

        self.windows: List[Dict] = []
        self._build_windows(video_list)

    def _build_windows(self, video_list):
        n_cached = n_videos = 0
        for vid in video_list:
            vp = os.path.join(self.video_root, vid["video_path"])
            if not os.path.exists(vp):
                continue
            n_videos += 1
            cache_path = None
            if self.frame_cache_dir:
                cp = os.path.join(self.frame_cache_dir, frame_cache_name(vid["video_path"]))
                if os.path.exists(cp):
                    cache_path = cp
                    n_cached += 1
            dur = vid.get("duration", 60.0)
            anns = vid.get("annotations", [])

            t = 0.0
            while t + self.window_sec <= dur + 1e-3:
                win_s = t
                win_e = min(t + self.window_sec, dur)
                n_frames = max(1, int((win_e - win_s) * self.fps))

                # Per-frame classification labels and regression targets
                frame_labels = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)
                # Regression target: for each frame position, find the nearest
                # GT segment boundary (dist_to_start, dist_to_end) per class
                reg_targets = np.zeros((n_frames, NUM_CLASSES, 2), dtype=np.float32)
                # centerness targets
                ctr_targets = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)
                has_gt = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)

                for ann in anns:
                    cls_name = norm_bug_type(ann.get("bug_type", ""))
                    if cls_name not in self.class_to_idx:
                        continue
                    cls_idx = self.class_to_idx[cls_name]
                    seg_s = ann.get("start", 0.0)
                    seg_e = ann.get("end",   0.0)

                    # overlap with window
                    ov_s = max(win_s, seg_s)
                    ov_e = min(win_e, seg_e)
                    if ov_e <= ov_s:
                        continue

                    for fi in range(n_frames):
                        ft = win_s + (fi + 0.5) / self.fps  # frame center time
                        if seg_s <= ft <= seg_e:
                            frame_labels[fi, cls_idx] = 1.0
                            dist_s = ft - seg_s
                            dist_e = seg_e - ft
                            reg_targets[fi, cls_idx, 0] = dist_s
                            reg_targets[fi, cls_idx, 1] = dist_e
                            seg_len = max(seg_e - seg_s, 1e-6)
                            l = dist_s / seg_len
                            r = dist_e / seg_len
                            ctr_targets[fi, cls_idx] = math.sqrt(
                                min(l, r) / max(max(l, r), 1e-6)
                            )
                            has_gt[fi, cls_idx] = 1.0

                # window-level label (any frame positive)
                win_label = (frame_labels.sum(0) > 0).astype(np.float32)

                self.windows.append({
                    "video_path":   vp,
                    "cache_path":   cache_path,
                    "win_start":    win_s,
                    "win_end":      win_e,
                    "n_frames":     n_frames,
                    "frame_labels": frame_labels,
                    "reg_targets":  reg_targets,
                    "ctr_targets":  ctr_targets,
                    "has_gt":       has_gt,
                    "win_label":    win_label,
                })
                t += self.stride_sec
        if self.frame_cache_dir:
            print(f"[dataset:{self.split}] frame cache: {n_cached}/{n_videos} videos "
                  f"(uncached videos use slow per-window decoding)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        frames = None
        if w.get("cache_path"):
            # Fast path: memmap slice from the pre-extracted frame cache
            try:
                arr = open_frame_cache(w["cache_path"])
                f_start = int(w["win_start"] * self.fps)
                sl = np.asarray(arr[f_start: f_start + w["n_frames"]])
                frames = torch.from_numpy(np.ascontiguousarray(sl)) \
                              .permute(0, 3, 1, 2).float().div_(255.0)
                frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
            except Exception:
                frames = None
        if frames is None:
            frames = load_frames(w["video_path"], w["win_start"], w["win_end"],
                                 self.fps, self.img_size)
        # Pad/crop to n_frames
        T = w["n_frames"]
        if frames.shape[0] < T:
            pad = torch.zeros(T - frames.shape[0], 3, self.img_size, self.img_size)
            frames = torch.cat([frames, pad], dim=0)
        else:
            frames = frames[:T]

        return {
            "frames":       frames,                                      # [T,3,H,W]
            "frame_labels": torch.from_numpy(w["frame_labels"]),        # [T, C]
            "reg_targets":  torch.from_numpy(w["reg_targets"]),         # [T, C, 2]
            "ctr_targets":  torch.from_numpy(w["ctr_targets"]),         # [T, C]
            "has_gt":       torch.from_numpy(w["has_gt"]),              # [T, C]
            "win_label":    torch.from_numpy(w["win_label"]),           # [C]
            "win_start":    w["win_start"],
            "win_end":      w["win_end"],
        }


def collate_fn(batch):
    max_T = max(b["frames"].shape[0] for b in batch)
    img_size = batch[0]["frames"].shape[-1]
    frames   = torch.zeros(len(batch), max_T, 3, img_size, img_size)
    fl       = torch.zeros(len(batch), max_T, NUM_CLASSES)
    rt       = torch.zeros(len(batch), max_T, NUM_CLASSES, 2)
    ct       = torch.zeros(len(batch), max_T, NUM_CLASSES)
    hg       = torch.zeros(len(batch), max_T, NUM_CLASSES)
    wl       = torch.stack([b["win_label"] for b in batch])
    for i, b in enumerate(batch):
        T = b["frames"].shape[0]
        frames[i, :T] = b["frames"]
        fl[i, :T]     = b["frame_labels"]
        rt[i, :T]     = b["reg_targets"]
        ct[i, :T]     = b["ctr_targets"]
        hg[i, :T]     = b["has_gt"]
    return dict(frames=frames, frame_labels=fl, reg_targets=rt,
                ctr_targets=ct, has_gt=hg, win_label=wl)


# ── multi-scale loss ──────────────────────────────────────────────────────────
def compute_loss(outputs, batch, asl_fn, lambda_cls=1.0, lambda_reg=1.0):
    """
    outputs: list of (cls [B,C,T], reg [B,2,T], ctr [B,1,T]) per FPN scale
    frame_labels: [B, T, C]
    reg_targets: [B, T, C, 2]
    has_gt: [B, T, C]  — mask for regression/centerness
    """
    frame_labels = batch["frame_labels"]  # [B, T, C]
    reg_targets  = batch["reg_targets"]   # [B, T, C, 2]
    ctr_targets  = batch["ctr_targets"]   # [B, T, C]
    has_gt       = batch["has_gt"]        # [B, T, C]

    total_loss = 0.0
    n_scales   = len(outputs)

    for cls_out, reg_out, ctr_out in outputs:
        # cls_out: [B, C, T] → transpose to [B, T, C]
        cls_t = cls_out.transpose(1, 2)  # [B, T, C]
        l_cls = asl_fn(cls_t.reshape(-1, NUM_CLASSES),
                       frame_labels.reshape(-1, NUM_CLASSES))

        # regression loss per class (only where has_gt)
        # reg_out: [B, 2, T] — but we have per-class targets
        # aggregate: for each position, sum loss over active classes
        reg_out_t = reg_out.transpose(1, 2)  # [B, T, 2]
        l_reg = 0.0
        for ci in range(NUM_CLASSES):
            mask = has_gt[..., ci].reshape(-1)   # [B*T]
            if mask.sum() < 1:
                continue
            # use average reg pred across 2 offsets to match gt start/end distances
            pred_s = reg_out_t[..., 0].reshape(-1)  # dist to start
            pred_e = reg_out_t[..., 1].reshape(-1)  # dist to end
            gt_s   = reg_targets[..., ci, 0].reshape(-1)
            gt_e   = reg_targets[..., ci, 1].reshape(-1)
            # Convert dist to absolute for DIoU: pos_t - pred_s, pos_t + pred_e
            # Approximate with smooth L1 since we don't have abs pos here
            l_reg = l_reg + (F.smooth_l1_loss(pred_s * mask, gt_s * mask, reduction="sum")
                           + F.smooth_l1_loss(pred_e * mask, gt_e * mask, reduction="sum")) \
                           / (mask.sum() + 1e-6)

        # centerness loss (aggregate over classes, mask by has_gt)
        ctr_out_t = ctr_out.squeeze(1).transpose(0, 1)  # [T, B] — not ideal; use mean
        # simpler: average centerness pred for each frame vs max class centerness target
        ctr_pred  = ctr_out.squeeze(1).reshape(-1)  # [B*T]
        ctr_tgt   = ctr_targets.max(-1).values.reshape(-1)  # [B*T]
        fg_mask   = has_gt.max(-1).values.reshape(-1)
        l_ctr     = centerness_loss(ctr_pred, ctr_tgt, fg_mask)

        total_loss += lambda_cls * l_cls + lambda_reg * l_reg + l_ctr

    return total_loss / n_scales


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_temporal_metrics(pred_by_class: Dict, gt_by_class: Dict,
                              tiou_thresholds=(0.3, 0.5, 0.7)):
    results = {}
    aps_by_thresh = {t: [] for t in tiou_thresholds}
    prec_by_thresh = {t: [] for t in tiou_thresholds}
    rec_by_thresh  = {t: [] for t in tiou_thresholds}
    mean_tious, over_pred_ratios = [], []

    for cls in CANON_BUG_TYPES:
        preds = pred_by_class.get(cls, [])
        gts   = gt_by_class.get(cls, [])
        if not gts:
            for t in tiou_thresholds:
                aps_by_thresh[t].append(0.0)
                prec_by_thresh[t].append(0.0)
                rec_by_thresh[t].append(0.0)
            continue

        preds_sorted = sorted(preds, key=lambda x: -x["score"])
        n_gt = len(gts)
        tiou_mat = np.zeros((len(preds_sorted), n_gt))
        for pi, p in enumerate(preds_sorted):
            for gi, g in enumerate(gts):
                inter = max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
                union = (p["end"] - p["start"]) + (g["end"] - g["start"]) - inter
                tiou_mat[pi, gi] = inter / max(union, 1e-6)

        class_tious = tiou_mat.max(axis=1) if len(preds_sorted) > 0 else np.array([])
        mean_tious.extend(class_tious[class_tious > 0].tolist())

        if len(gts) > 0:
            over_pred_ratios.append(len(preds) / max(n_gt, 1))

        for thresh in tiou_thresholds:
            tp = np.zeros(len(preds_sorted))
            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi, p in enumerate(preds_sorted):
                best_gi, best_iou = -1, thresh
                for gi in range(n_gt):
                    if not gt_matched[gi] and tiou_mat[pi, gi] >= best_iou:
                        best_iou = tiou_mat[pi, gi]
                        best_gi  = gi
                if best_gi >= 0:
                    tp[pi] = 1
                    gt_matched[best_gi] = True

            fp = 1 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec  = tp_cum / max(n_gt, 1)
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap   = 0.0
            for r_thr in np.linspace(0, 1, 11):
                p_at_r = prec[rec >= r_thr].max() if (rec >= r_thr).any() else 0.0
                ap += p_at_r / 11
            aps_by_thresh[thresh].append(ap)
            prec_by_thresh[thresh].append(prec[-1] if len(prec) else 0.0)
            rec_by_thresh[thresh].append(rec[-1]   if len(rec)  else 0.0)

    for t in tiou_thresholds:
        results[f"mAP@{t}"]   = float(np.mean(aps_by_thresh[t]))
        results[f"prec@{t}"]  = float(np.mean(prec_by_thresh[t]))
        results[f"rec@{t}"]   = float(np.mean(rec_by_thresh[t]))
        f1 = 2 * results[f"prec@{t}"] * results[f"rec@{t}"] / \
             max(results[f"prec@{t}"] + results[f"rec@{t}"], 1e-9)
        results[f"f1@{t}"] = float(f1)
        for ci, cls in enumerate(CANON_BUG_TYPES):
            results[f"AP@{t}/{cls}"] = float(aps_by_thresh[t][ci]) if ci < len(aps_by_thresh[t]) else 0.0

    results["mean_tIoU"]              = float(np.mean(mean_tious)) if mean_tious else 0.0
    results["segment_over_pred_ratio"] = float(np.mean(over_pred_ratios)) if over_pred_ratios else 0.0
    return results


def compute_multilabel_metrics(y_pred_prob: np.ndarray, y_true: np.ndarray,
                                thresholds: np.ndarray) -> Dict:
    y_pred = (y_pred_prob >= thresholds).astype(int)
    metrics = {}
    for avg in ["micro", "macro", "weighted"]:
        metrics[f"f1_{avg}"]        = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"recall_{avg}"]    = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    metrics["hamming_loss"]   = float(hamming_loss(y_true, y_pred))
    metrics["subset_accuracy"] = float((y_pred == y_true).all(axis=1).mean())
    try:
        metrics["clip_mAP"] = float(average_precision_score(y_true, y_pred_prob, average="macro"))
    except Exception:
        metrics["clip_mAP"] = 0.0
    try:
        metrics["lrap"] = float(label_ranking_average_precision_score(y_true, y_pred_prob))
    except Exception:
        metrics["lrap"] = 0.0
    for ci, cls in enumerate(CANON_BUG_TYPES):
        if y_true[:, ci].sum() > 0:
            try:
                metrics[f"auc_roc/{cls}"] = float(roc_auc_score(y_true[:, ci], y_pred_prob[:, ci]))
            except Exception:
                metrics[f"auc_roc/{cls}"] = 0.0
    return metrics


def tune_thresholds(y_pred_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    thresholds = np.full(NUM_CLASSES, 0.5)
    for ci in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            pred = (y_pred_prob[:, ci] >= t).astype(int)
            f1 = f1_score(y_true[:, ci], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[ci] = best_t
    return thresholds


# ── inference: frames → segments ─────────────────────────────────────────────
def decode_predictions(outputs, win_start: float, fps: float,
                        min_conf: float = 0.2, min_duration: float = 0.5,
                        thresholds: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Decode multi-scale FPN outputs into temporal segments.
    outputs: list of (cls [1,C,T], reg [1,2,T], ctr [1,1,T])
    """
    if thresholds is None:
        thresholds = np.full(NUM_CLASSES, 0.5)

    all_detections = []
    for scale_idx, (cls_out, reg_out, ctr_out) in enumerate(outputs):
        T = cls_out.shape[-1]
        cls_probs = torch.sigmoid(cls_out[0]).cpu().numpy()  # [C, T]
        reg_vals  = reg_out[0].cpu().numpy()                 # [2, T]
        ctr_probs = torch.sigmoid(ctr_out[0, 0]).cpu().numpy()  # [T]

        for ti in range(T):
            t_center = win_start + (ti + 0.5) / fps
            conf = ctr_probs[ti]
            dist_s = reg_vals[0, ti]
            dist_e = reg_vals[1, ti]
            pred_s = max(0.0, t_center - dist_s)
            pred_e = t_center + dist_e

            if pred_e - pred_s < min_duration:
                continue

            for ci, cls in enumerate(CANON_BUG_TYPES):
                cls_conf = cls_probs[ci, ti]
                if cls_conf >= thresholds[ci] and cls_conf * conf >= min_conf:
                    all_detections.append({
                        "bug_type": cls,
                        "start":    pred_s,
                        "end":      pred_e,
                        "score":    float(cls_conf * conf),
                    })

    return all_detections


def soft_nms_temporal(detections: List[Dict], sigma: float = 0.5,
                       score_thr: float = 0.05) -> List[Dict]:
    """Temporal Soft-NMS per class."""
    if not detections:
        return []
    result = []
    by_class: Dict[str, List[Dict]] = {}
    for d in detections:
        by_class.setdefault(d["bug_type"], []).append(d)

    for cls, dets in by_class.items():
        dets = [dict(d) for d in sorted(dets, key=lambda x: -x["score"])]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                inter = max(0, min(dets[i]["end"], dets[j]["end"])
                              - max(dets[i]["start"], dets[j]["start"]))
                union = (dets[i]["end"] - dets[i]["start"]) + \
                        (dets[j]["end"] - dets[j]["start"]) - inter
                iou   = inter / max(union, 1e-6)
                dets[j]["score"] *= math.exp(-(iou ** 2) / sigma)
        result.extend([d for d in dets if d["score"] >= score_thr])

    return result


# ── checkpoint helpers ────────────────────────────────────────────────────────
def manage_epoch_checkpoints(save_dir: str, keep: int = 3):
    ckpts = sorted(Path(save_dir).glob("checkpoint_epoch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for p in ckpts[:-keep]:
        p.unlink()


# ── early stopping ────────────────────────────────────────────────────────────
class MultiTaskEarlyStopping:
    def __init__(self, patience: int = 15, lr_floor: float = 1e-6):
        self.patience  = patience
        self.lr_floor  = lr_floor
        self.best      = -float("inf")
        self.counter   = 0
        self.should_stop = False

    def composite(self, f1: float, map05: float) -> float:
        return 0.3 * f1 + 0.7 * map05

    def step(self, f1: float, map05: float, current_lr: float) -> bool:
        score = self.composite(f1, map05)
        if score > self.best + 1e-5:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience or current_lr <= self.lr_floor:
            self.should_stop = True
        return self.should_stop


# ── training epoch ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, asl_fn,
                    lambda_cls, lambda_reg, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        frames = batch["frames"].to(device)
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(frames)
            loss    = compute_loss(outputs, batch_dev, asl_fn, lambda_cls, lambda_reg)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, fps: float, window_sec: float,
             thresholds: Optional[np.ndarray] = None):
    model.eval()
    all_probs, all_labels = [], []
    pred_by_class: Dict[str, List] = {c: [] for c in CANON_BUG_TYPES}
    gt_by_class:   Dict[str, List] = {c: [] for c in CANON_BUG_TYPES}

    for batch in loader:
        frames = batch["frames"].to(device)
        outputs = model(frames)

        # Clip-level probs from last FPN scale, averaged over T
        cls_out = outputs[-1][0]  # [B, C, T]
        probs   = torch.sigmoid(cls_out).mean(-1).cpu().numpy()  # [B, C]
        labels  = batch["win_label"].numpy()  # [B, C]
        all_probs.append(probs)
        all_labels.append(labels)

        # Temporal predictions per window
        B = frames.shape[0]
        for bi in range(B):
            win_s = batch["win_start"][bi].item() if isinstance(batch["win_start"], torch.Tensor) \
                    else batch["win_start"][bi]
            single_out = [(c[bi:bi+1], r[bi:bi+1], ct[bi:bi+1])
                          for c, r, ct in outputs]
            dets = decode_predictions(single_out, win_s, fps,
                                      min_conf=0.1, min_duration=0.5,
                                      thresholds=thresholds)
            dets = soft_nms_temporal(dets)
            for d in dets:
                pred_by_class[d["bug_type"]].append(d)

        # GT from window
        for bi in range(B):
            win_s = batch["win_start"][bi].item() if isinstance(batch["win_start"], torch.Tensor) \
                    else batch["win_start"][bi]
            fl = batch["frame_labels"][bi].numpy()  # [T, C]
            T  = fl.shape[0]
            for ci, cls in enumerate(CANON_BUG_TYPES):
                in_seg = False
                seg_s  = 0.0
                for fi in range(T):
                    ft = win_s + (fi + 0.5) / fps
                    if fl[fi, ci] > 0.5:
                        if not in_seg:
                            in_seg = True
                            seg_s  = ft - 0.5 / fps
                    else:
                        if in_seg:
                            gt_by_class[cls].append({"start": seg_s, "end": ft, "video_id": "batch"})
                            in_seg = False
                if in_seg:
                    gt_by_class[cls].append({"start": seg_s,
                                              "end": win_s + T / fps, "video_id": "batch"})

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if thresholds is None:
        thresholds = tune_thresholds(all_probs, all_labels)

    cls_metrics  = compute_multilabel_metrics(all_probs, all_labels, thresholds)
    temp_metrics = compute_temporal_metrics(pred_by_class, gt_by_class)

    return cls_metrics, temp_metrics, thresholds


# ── visualizations ────────────────────────────────────────────────────────────
def save_visualizations(temp_metrics: Dict, cls_metrics: Dict,
                         out_dir: str, prefix: str = "test"):
    os.makedirs(out_dir, exist_ok=True)

    # mAP vs tIoU curve
    threshs = [0.3, 0.5, 0.7]
    maps    = [temp_metrics.get(f"mAP@{t}", 0) for t in threshs]
    fig, ax = plt.subplots()
    ax.plot(threshs, maps, "o-", color="steelblue")
    ax.set_xlabel("tIoU threshold")
    ax.set_ylabel("mAP")
    ax.set_title("mAP vs tIoU Threshold")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.savefig(os.path.join(out_dir, f"{prefix}_map_vs_tiou.png"), bbox_inches="tight")
    plt.close(fig)

    # Per-class AP bar chart at tIoU=0.5
    aps = [temp_metrics.get(f"AP@0.5/{c}", 0) for c in CANON_BUG_TYPES]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(CANON_BUG_TYPES, aps, color="teal")
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class AP @ tIoU=0.5")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(os.path.join(out_dir, f"{prefix}_per_class_ap.png"), bbox_inches="tight")
    plt.close(fig)

    # Segment over-prediction ratio
    opr = temp_metrics.get("segment_over_pred_ratio", 0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Over-pred ratio"], [opr], color="coral")
    ax.axhline(1.0, color="green", linestyle="--", label="ideal")
    ax.set_ylabel("pred / gt ratio")
    ax.set_title("Segment Over-Prediction")
    ax.legend()
    fig.savefig(os.path.join(out_dir, f"{prefix}_over_pred_ratio.png"), bbox_inches="tight")
    plt.close(fig)

    # Multi-label classification F1 per class (micro/macro/weighted)
    labels = ["micro", "macro", "weighted"]
    vals   = [cls_metrics.get(f"f1_{l}", 0) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, vals, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylabel("F1")
    ax.set_title("Clip-level F1 scores")
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(out_dir, f"{prefix}_f1_scores.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"[VIZ] Saved plots to {out_dir}")


# ── HPO objective ─────────────────────────────────────────────────────────────
def hpo_objective(trial, args, train_vids, val_vids, device):
    lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
    lambda_cls   = trial.suggest_float("lambda_cls",   0.5,  2.0)
    lambda_reg   = trial.suggest_float("lambda_reg",   0.5,  2.0)
    dropout      = trial.suggest_float("dropout",      0.1,  0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [2, 4, 8])
    stride_sec   = trial.suggest_categorical("stride_sec", [0.5, 1.0])

    set_seed(SEED)
    model = PyramidTransformerLocalization(
        backbone_name=args.backbone,
        dropout=dropout
    ).to(device)

    train_ds = TemporalWindowDataset(train_vids, args.video_root,
                                     fps=args.fps, img_size=args.img_size,
                                     window_sec=args.window_sec,
                                     stride_sec=stride_sec, split="train",
                                     frame_cache_dir=args.frame_cache_dir)
    val_ds   = TemporalWindowDataset(val_vids, args.video_root,
                                     fps=args.fps, img_size=args.img_size,
                                     window_sec=args.window_sec,
                                     stride_sec=stride_sec, split="val",
                                     frame_cache_dir=args.frame_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    asl_fn    = AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler()
    es        = MultiTaskEarlyStopping(patience=15)

    n_pilot_epochs = 5
    best_composite = -float("inf")

    for epoch in range(n_pilot_epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, asl_fn,
                        lambda_cls, lambda_reg, device)
        cls_m, temp_m, _ = evaluate(model, val_loader, device, args.fps, args.window_sec)
        f1   = cls_m.get("f1_micro", 0.0)
        map5 = temp_m.get("mAP@0.5", 0.0)
        comp = es.composite(f1, map5)
        best_composite = max(best_composite, comp)

        trial.report(comp, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_composite


# ── full training run ─────────────────────────────────────────────────────────
def run_train(args):
    set_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.logdir,   exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "tensorboard"), exist_ok=True)

    # W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb and args.wandb_project
    if use_wandb:
        wandb.init(project=args.wandb_project, name="m3_pyramid_transformer",
                   config=vars(args), resume="allow")

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, "tensorboard"))

    # Load dataset
    with open(args.temporal_dataset) as f:
        dataset = json.load(f)

    splits   = dataset.get("metadata", {}).get("splits", dataset.get("splits", {}))
    all_vids = {v["video_id"]: v for v in dataset.get("videos", [])}

    train_ids = splits.get("train", [])
    val_ids   = splits.get("val",   [])
    test_ids  = splits.get("test",  [])

    train_vids = [all_vids[i] for i in train_ids if i in all_vids]
    val_vids   = [all_vids[i] for i in val_ids   if i in all_vids]
    test_vids  = [all_vids[i] for i in test_ids  if i in all_vids]

    print(f"[INFO] Dataset: {len(train_vids)} train / {len(val_vids)} val / {len(test_vids)} test")

    # Build/verify the frame cache BEFORE training (no-op once complete)
    if args.frame_cache_dir:
        ensure_frame_cache(train_vids + val_vids + test_vids, args.video_root,
                           args.frame_cache_dir, fps=args.fps, img_size=args.img_size,
                           workers=max(args.num_workers, 4))
    else:
        print("[WARN] --frame-cache-dir not set — every window sample will open and "
              "seek-decode its source video (slow). Strongly consider setting it.")

    # ── pilot mode ────────────────────────────────────────────────────────────
    if args.pilot:
        print("[INFO] PILOT MODE — 5 epochs, no HPO")
        model = PyramidTransformerLocalization(backbone_name=args.backbone).to(device)
        train_ds = TemporalWindowDataset(train_vids, args.video_root, fps=args.fps,
                                         img_size=args.img_size, window_sec=args.window_sec,
                                         stride_sec=args.stride_sec, split="train",
                                         frame_cache_dir=args.frame_cache_dir)
        val_ds   = TemporalWindowDataset(val_vids, args.video_root, fps=args.fps,
                                         img_size=args.img_size, window_sec=args.window_sec,
                                         stride_sec=args.stride_sec, split="val",
                                         frame_cache_dir=args.frame_cache_dir)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=True, drop_last=True,
                                  persistent_workers=args.num_workers > 0)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=True,
                                  persistent_workers=args.num_workers > 0)
        asl_fn    = AsymmetricLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler    = torch.cuda.amp.GradScaler()
        for epoch in range(5):
            tl = train_one_epoch(model, train_loader, optimizer, scaler, asl_fn, 1.0, 1.0, device)
            cls_m, temp_m, _ = evaluate(model, val_loader, device, args.fps, args.window_sec)
            print(f"  Pilot epoch {epoch+1}/5 | loss={tl:.4f} | "
                  f"F1={cls_m.get('f1_micro',0):.3f} | mAP@0.5={temp_m.get('mAP@0.5',0):.3f}")
        print("[INFO] Pilot complete")
        writer.close()
        return

    # ── HPO phase ─────────────────────────────────────────────────────────────
    best_hparams = {
        "lr": 1e-4, "lambda_cls": 1.0, "lambda_reg": 1.0,
        "dropout": 0.1, "weight_decay": 1e-4,
        "batch_size": args.batch_size, "stride_sec": args.stride_sec
    }
    best_hparams_path = os.path.join(args.logdir, "best_hparams.json")
    study_path        = os.path.join(args.logdir, "optuna_study.db")
    hpo_log           = os.path.join(args.logdir, "hpo_progress.log")

    if args.hpo_trials > 0 and OPTUNA_AVAILABLE:
        if os.path.exists(best_hparams_path):
            print("[INFO] Loading existing best_hparams.json — skipping HPO")
            with open(best_hparams_path) as f:
                best_hparams = json.load(f)
        else:
            print(f"[INFO] HPO: {args.hpo_trials} trials (Optuna TPE + MedianPruner)")
            storage = f"sqlite:///{study_path}"
            study   = optuna.create_study(
                study_name="m3_pyramid_transformer",
                direction="maximize",
                sampler=TPESampler(seed=SEED),
                pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
                storage=storage, load_if_exists=True
            )
            n_done = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            n_remaining = max(0, args.hpo_trials - n_done)
            print(f"[INFO] Completed trials so far: {n_done}/{args.hpo_trials}")

            if n_remaining > 0:
                study.optimize(
                    lambda t: hpo_objective(t, args, train_vids, val_vids, device),
                    n_trials=n_remaining
                )

            best_hparams = study.best_params
            best_hparams["stride_sec"] = best_hparams.get("stride_sec", args.stride_sec)
            with open(best_hparams_path, "w") as f:
                json.dump(best_hparams, f, indent=2)

            with open(hpo_log, "a") as f:
                f.write(f"HPO complete — best composite: {study.best_value:.4f} "
                        f"| params: {best_hparams}\n")
            print(f"[HPO] Best composite: {study.best_value:.4f}")
            print(f"[HPO] Best params: {best_hparams}")

    # ── full training ─────────────────────────────────────────────────────────
    lr           = best_hparams.get("lr",           1e-4)
    lambda_cls   = best_hparams.get("lambda_cls",   1.0)
    lambda_reg   = best_hparams.get("lambda_reg",   1.0)
    dropout      = best_hparams.get("dropout",      0.1)
    weight_decay = best_hparams.get("weight_decay", 1e-4)
    batch_size   = int(best_hparams.get("batch_size",  args.batch_size))
    stride_sec   = float(best_hparams.get("stride_sec", args.stride_sec))

    model = PyramidTransformerLocalization(
        backbone_name=args.backbone,
        dropout=dropout
    ).to(device)

    # Architecture summary
    if TORCHINFO_AVAILABLE:
        try:
            dummy_frames = torch.zeros(1, 8, 3, args.img_size, args.img_size).to(device)
            summ = torchinfo_summary(model, input_data=dummy_frames, verbose=0)
            arch_path = os.path.join(args.logdir, "architecture_summary.txt")
            with open(arch_path, "w") as f:
                f.write(str(summ))
            print(f"[INFO] Architecture summary saved to {arch_path}")
        except Exception as e:
            print(f"[WARN] torchinfo failed: {e}")

    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO] Resumed from {args.checkpoint} (epoch {start_epoch})")

    train_ds = TemporalWindowDataset(train_vids, args.video_root, fps=args.fps,
                                     img_size=args.img_size, window_sec=args.window_sec,
                                     stride_sec=stride_sec, split="train",
                                     frame_cache_dir=args.frame_cache_dir)
    val_ds   = TemporalWindowDataset(val_vids, args.video_root, fps=args.fps,
                                     img_size=args.img_size, window_sec=args.window_sec,
                                     stride_sec=stride_sec, split="val",
                                     frame_cache_dir=args.frame_cache_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    asl_fn    = AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6, verbose=True
    )
    scaler    = torch.cuda.amp.GradScaler()
    es        = MultiTaskEarlyStopping(patience=15, lr_floor=1e-6)

    best_scores = {"composite": -1e9, "f1": -1e9, "mAP": -1e9}
    thresholds  = np.full(NUM_CLASSES, 0.5)

    print(f"[INFO] Starting full training: epochs={args.epochs}, "
          f"lr={lr:.1e}, batch={batch_size}, stride={stride_sec}s")

    for epoch in range(start_epoch, args.epochs):
        t0        = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                     asl_fn, lambda_cls, lambda_reg, device)
        cls_m, temp_m, thresholds = evaluate(model, val_loader, device,
                                              args.fps, args.window_sec)

        f1   = cls_m.get("f1_micro", 0.0)
        map5 = temp_m.get("mAP@0.5", 0.0)
        map3 = temp_m.get("mAP@0.3", 0.0)
        comp = es.composite(f1, map5)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"loss={train_loss:.4f} | F1={f1:.3f} | "
              f"mAP@0.3={map3:.3f} | mAP@0.5={map5:.3f} | "
              f"composite={comp:.4f} | lr={current_lr:.1e} | "
              f"{int(time.time()-t0)}s")

        # TensorBoard
        writer.add_scalar("train/loss",     train_loss, epoch)
        writer.add_scalar("val/f1_micro",   f1,         epoch)
        writer.add_scalar("val/mAP@0.5",    map5,       epoch)
        writer.add_scalar("val/mAP@0.3",    map3,       epoch)
        writer.add_scalar("val/composite",  comp,       epoch)
        writer.add_scalar("train/lr",       current_lr, epoch)

        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss,
                       "val/f1_micro": f1, "val/mAP@0.5": map5,
                       "val/mAP@0.3": map3, "val/composite": comp,
                       "train/lr": current_lr})

        # Save epoch checkpoint (rolling)
        epoch_ckpt = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "thresholds": thresholds.tolist(),
                    "optimizer_state": optimizer.state_dict()}, epoch_ckpt)
        manage_epoch_checkpoints(args.save_dir, keep=3)

        # Save best checkpoints
        if comp > best_scores["composite"]:
            best_scores["composite"] = comp
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "thresholds": thresholds.tolist()},
                       os.path.join(args.save_dir, "best_composite.pt"))
        if f1 > best_scores["f1"]:
            best_scores["f1"] = f1
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "thresholds": thresholds.tolist()},
                       os.path.join(args.save_dir, "best_f1.pt"))
        if map5 > best_scores["mAP"]:
            best_scores["mAP"] = map5
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "thresholds": thresholds.tolist()},
                       os.path.join(args.save_dir, "best_mAP.pt"))

        scheduler.step(comp)
        if es.step(f1, map5, current_lr):
            print(f"[INFO] Early stopping at epoch {epoch+1} "
                  f"(patience={es.patience}, counter={es.counter})")
            break

    # ── test phase ────────────────────────────────────────────────────────────
    print("\n[INFO] Running test set evaluation with all 3 best checkpoints...")
    test_ds = TemporalWindowDataset(test_vids, args.video_root, fps=args.fps,
                                    img_size=args.img_size, window_sec=args.window_sec,
                                    stride_sec=stride_sec, split="test",
                                    frame_cache_dir=args.frame_cache_dir)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    all_test_results = {}
    for ckpt_name in ["best_composite", "best_f1", "best_mAP"]:
        ckpt_path = os.path.join(args.save_dir, f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        thr  = np.array(ckpt.get("thresholds", [0.5] * NUM_CLASSES))
        cls_m, temp_m, _ = evaluate(model, test_loader, device,
                                     args.fps, args.window_sec, thr)
        all_test_results[ckpt_name] = {**cls_m, **temp_m}
        print(f"\n  [{ckpt_name}] F1={cls_m.get('f1_micro',0):.3f} | "
              f"mAP@0.3={temp_m.get('mAP@0.3',0):.3f} | "
              f"mAP@0.5={temp_m.get('mAP@0.5',0):.3f} | "
              f"mAP@0.7={temp_m.get('mAP@0.7',0):.3f} | "
              f"mean_tIoU={temp_m.get('mean_tIoU',0):.3f}")

        viz_dir = os.path.join(args.logdir, f"viz_{ckpt_name}")
        save_visualizations(temp_m, cls_m, viz_dir, prefix=ckpt_name)

        if use_wandb:
            wandb.log({f"test_{ckpt_name}/{k}": v for k, v in all_test_results[ckpt_name].items()})

    metrics_path = os.path.join(args.logdir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\n[INFO] Test metrics saved to {metrics_path}")

    writer.close()
    if use_wandb:
        wandb.finish()


# ── inference mode ────────────────────────────────────────────────────────────
def run_infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    thresholds = np.array(ckpt.get("thresholds", [0.5] * NUM_CLASSES))

    model = PyramidTransformerLocalization(backbone_name=args.backbone).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Get video duration
    dur = 600.0
    if DECORD_AVAILABLE:
        try:
            vr  = VideoReader(args.video_path, ctx=cpu(0))
            dur = len(vr) / vr.get_avg_fps()
        except Exception:
            pass
    elif CV2_AVAILABLE:
        try:
            cap = cv2.VideoCapture(args.video_path)
            dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
            cap.release()
        except Exception:
            pass

    all_dets: List[Dict] = []
    t = 0.0
    while t + args.window_sec <= dur + 1e-3:
        win_s = t
        win_e = min(t + args.window_sec, dur)
        frames = load_frames(args.video_path, win_s, win_e,
                             args.fps, args.img_size)
        frames = frames.unsqueeze(0).to(device)  # [1, T, 3, H, W]
        with torch.no_grad():
            outputs = model(frames)
        dets = decode_predictions(outputs, win_s, args.fps,
                                  min_conf=0.2, min_duration=0.5,
                                  thresholds=thresholds)
        all_dets.extend(dets)
        t += args.stride_sec

    all_dets = soft_nms_temporal(all_dets)

    # Format output (backend-compatible)
    annotations = []
    for seg_id, d in enumerate(sorted(all_dets, key=lambda x: x["start"])):
        annotations.append({
            "segment_id": seg_id,
            "bug_type":   d["bug_type"].replace("-", "_"),
            "start":      round(d["start"], 3),
            "end":        round(d["end"],   3),
            "duration":   round(d["end"] - d["start"], 3),
            "score":      round(d["score"], 4),
        })

    result = {
        "video_id":    os.path.splitext(os.path.basename(args.video_path))[0],
        "video_path":  args.video_path,
        "duration":    dur,
        "annotations": annotations,
    }

    if args.inference_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.inference_out)), exist_ok=True)
        with open(args.inference_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFER] Result saved to {args.inference_out}")
    else:
        print(json.dumps(result, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Model 3: Feature Pyramid + Transformer")
    p.add_argument("--mode",             choices=["train", "infer"], default="train")
    # data
    p.add_argument("--video-root",       default="")
    p.add_argument("--temporal-dataset", default="")
    p.add_argument("--video-path",       default="")   # infer only
    p.add_argument("--frame-cache-dir",  default=None,
                   help="Dir for pre-extracted uint8 frame .npy files. Missing videos "
                        "are extracted automatically before training (one-time), then "
                        "windows are read as memmap slices instead of decoding video.")
    # training
    p.add_argument("--save-dir",         default="checkpoints")
    p.add_argument("--logdir",           default="logs")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=4)
    p.add_argument("--num-workers",      type=int,   default=8)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--hpo-trials",       type=int,   default=0)
    p.add_argument("--pilot",            action="store_true")
    p.add_argument("--checkpoint",       default="")
    # model
    p.add_argument("--backbone",         default="swin_small_patch4_window7_224")
    p.add_argument("--window-sec",       type=float, default=4.0)
    p.add_argument("--stride-sec",       type=float, default=1.0)
    p.add_argument("--fps",              type=float, default=8.0)
    p.add_argument("--img-size",         type=int,   default=224)
    # logging
    p.add_argument("--wandb-project",    default="")
    p.add_argument("--no-wandb",         action="store_true")
    # infer
    p.add_argument("--inference-out",    default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint required for infer mode")
            sys.exit(1)
        if not args.video_path:
            print("ERROR: --video-path required for infer mode")
            sys.exit(1)
        run_infer(args)
