#!/usr/bin/env python3
"""
Model 1: BugBiLSTM + Temporal Regression Head (SSAD-inspired)
==============================================================
Architecture: ResNet18 + BiLSTM + TemporalAttention + temporal offset regression head.
Initialized from the existing anygate_wgatedreg checkpoint.

Loss:
  - ASL   (Ridnik et al., "Asymmetric Loss For Multi-Label Classification," ICCV 2021)
  - DIoU  (Zheng et al., "Distance-IoU Loss," AAAI 2020)
  - Focal (Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017)

Modes:
  --mode train   Full pipeline: [HPO →] train → val each epoch → test at end
  --mode infer   Single-video inference → result.json (for web UI backend)

Flags:
  --pilot        Run 5 epochs only (architecture sanity check), then exit
  --hpo-trials N Run N Optuna TPE trials before full training
  --checkpoint   Path to .pt for resume (training) or inference
"""

import os, sys, json, math, time, argparse, glob, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score, hamming_loss, accuracy_score, label_ranking_average_precision_score,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] optuna not installed. HPO disabled. pip install optuna")

try:
    import torchinfo
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

_BACKENDS: Dict[str, bool] = {}
try:
    import decord
    decord.bridge.set_bridge("torch")
    _BACKENDS["decord"] = True
except Exception:
    _BACKENDS["decord"] = False
try:
    import cv2
    _BACKENDS["opencv"] = True
except Exception:
    _BACKENDS["opencv"] = False

# ── Constants ─────────────────────────────────────────────────────────────────

CANON_BUG_TYPES = ["z-clipping", "corrupted_texture", "geometry_corruption", "z-fighting", "boundary_hole"]
_ALIAS = {"z_clipping": "z-clipping", "z_fighting": "z-fighting", "boundary_hole": "boundary_hole",
          "corrupted_texture": "corrupted_texture", "geometry_corruption": "geometry_corruption"}
BUG2IDX = {b: i for i, b in enumerate(CANON_BUG_TYPES)}
NUM_CLASSES = len(CANON_BUG_TYPES)

OUTPUT_NAMES = {  # canonical → output (underscore form for result.json)
    "z-clipping": "z_clipping", "z-fighting": "z_fighting",
    "corrupted_texture": "corrupted_texture",
    "geometry_corruption": "geometry_corruption", "boundary_hole": "boundary_hole",
}

def norm_bug_type(s: str) -> str:
    s2 = s.strip().lower().replace(" ", "_").replace("-", "_")
    canon = _ALIAS.get(s2, s2)
    if canon in ("z_clipping", "z-clipping"): return "z-clipping"
    if canon in ("z_fighting", "z-fighting"): return "z-fighting"
    return canon

# ── Video IO ──────────────────────────────────────────────────────────────────

def extract_frames_at_fps(video_path: str, target_fps: float = 8.0,
                          img_size: int = 224) -> Tuple[np.ndarray, float]:
    """Returns (frames_np [T,H,W,3] uint8, duration_sec). Tries decord then opencv."""
    if _BACKENDS.get("decord"):
        try:
            vr = decord.VideoReader(video_path, width=img_size, height=img_size)
            native_fps = float(vr.get_avg_fps())
            total_frames = len(vr)
            duration = total_frames / native_fps
            stride = max(1, round(native_fps / target_fps))
            indices = list(range(0, total_frames, stride))
            frames = vr.get_batch(indices).numpy()  # [T,H,W,3]
            return frames.astype(np.uint8), duration
        except Exception:
            pass
    if _BACKENDS.get("opencv"):
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps
        stride = max(1, round(native_fps / target_fps))
        frames_list = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
                frames_list.append(frame)
            idx += 1
        cap.release()
        return np.stack(frames_list).astype(np.uint8), duration
    raise RuntimeError(f"No video backend available. Install decord or opencv.")

def get_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):  # [B,T,D]
        h = torch.tanh(self.W(x))
        a = torch.softmax(self.v(h).squeeze(-1), dim=1)
        z = (x * a.unsqueeze(-1)).sum(dim=1)
        return z, a


class BugBiLSTMLocalization(nn.Module):
    """
    ResNet18 + BiLSTM + TemporalAttention for multi-label classification,
    plus a temporal regression head predicting (delta_start, delta_end)
    offsets from the window center to GT segment boundaries.

    Outputs:
      logits_types  [B, C]   — multi-label classification
      logit_any     [B]      — presence (any bug)
      reg_offsets   [B, 2]   — (delta_start_norm, delta_end_norm), normalized by window_sec
    """
    def __init__(self, num_bug_types: int = NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.cnn = nn.Sequential(*list(enc.children())[:-1])
        self.feat_dim = 512
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.4)
        self.attn = TemporalAttention(d_model=512)
        self.drop = nn.Dropout(p=dropout)
        self.fc_types = nn.Linear(512, num_bug_types)
        self.fc_any = nn.Linear(512, 1)
        self.fc_reg = nn.Linear(512, 2)  # (delta_start_norm, delta_end_norm)

    def forward(self, frames):  # [B,T,3,H,W]
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        f = self.cnn(x).flatten(1).reshape(B, T, self.feat_dim).contiguous()
        y, _ = self.lstm(f)
        z, att = self.attn(y)
        z = self.drop(z)
        logits_types = self.fc_types(z)
        logit_any = self.fc_any(z).squeeze(-1)
        reg_offsets = self.fc_reg(z)
        return logits_types, logit_any, reg_offsets, att

# ── Loss functions ────────────────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """ASL: Ridnik et al., ICCV 2021."""
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_m = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = targets * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - p_m).clamp(min=self.eps))
        loss = loss_pos * ((1 - p) ** self.gamma_pos) + loss_neg * (p_m ** self.gamma_neg)
        return -loss.mean()


def focal_loss_binary(logit: torch.Tensor, target: torch.Tensor,
                      gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Binary focal loss for presence head."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = torch.where(target == 1, p, 1 - p)
    alpha_t = torch.where(target == 1, torch.tensor(alpha, device=logit.device),
                          torch.tensor(1 - alpha, device=logit.device))
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def diou_loss_temporal(pred_offsets: torch.Tensor, gt_offsets: torch.Tensor,
                       window_sec: float = 2.0, mask: torch.Tensor = None) -> torch.Tensor:
    """
    DIoU loss for temporal regression.
    pred_offsets / gt_offsets: [B, 2] normalized by window_sec.
    Converts to absolute seconds, then computes tIoU + center distance penalty.
    Zheng et al., AAAI 2020.
    """
    # Convert normalized offsets to absolute times (center assumed at 0)
    half = window_sec / 2.0
    # pred: predicted [start, end] relative to window center
    p_s = -half + pred_offsets[:, 0] * window_sec  # delta_start_norm * window_sec
    p_e = half + pred_offsets[:, 1] * window_sec   # delta_end_norm * window_sec
    g_s = -half + gt_offsets[:, 0] * window_sec
    g_e = half + gt_offsets[:, 1] * window_sec

    inter = (torch.min(p_e, g_e) - torch.max(p_s, g_s)).clamp(min=0)
    union = (torch.max(p_e, g_e) - torch.min(p_s, g_s)).clamp(min=1e-6)
    tiou = inter / union

    # Center distance penalty
    p_c = (p_s + p_e) / 2
    g_c = (g_s + g_e) / 2
    enclosing = (torch.max(p_e, g_e) - torch.min(p_s, g_s)).clamp(min=1e-6)
    rho2 = (p_c - g_c) ** 2 / (enclosing ** 2 + 1e-6)

    loss = 1 - tiou + rho2
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp(min=1)
        return loss.sum() / denom
    return loss.mean()

# ── Dataset ───────────────────────────────────────────────────────────────────

class TemporalLocDataset(Dataset):
    """
    Generates per-window samples from full videos using temporal_bug_dataset.json.
    Each sample: (frames [T,3,H,W], cls_labels [C], any_label, reg_targets [2], has_bug)
    reg_targets: normalized (delta_start, delta_end) from window center to nearest GT boundary.
    """
    def __init__(self, video_list: List[Dict], video_root: str,
                 window_sec: float = 2.0, stride_sec: float = 1.0,
                 fps: float = 8.0, img_size: int = 224,
                 transform=None, tiou_thr: float = 0.3,
                 split: str = "train"):
        self.video_root = Path(video_root)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.fps = fps
        self.img_size = img_size
        self.transform = transform or get_transform(img_size)
        self.tiou_thr = tiou_thr
        self.split = split
        self.samples = []  # list of (video_path, annotations, window_start_sec)
        self._build_index(video_list)

    def _build_index(self, video_list):
        for vid in video_list:
            video_path = self.video_root / vid["video_path"]
            if not video_path.exists():
                continue
            duration = float(vid.get("duration", 0))
            annotations = vid.get("annotations", [])
            t = 0.0
            while t + self.window_sec <= duration + 0.5:
                self.samples.append((str(video_path), annotations, t))
                t += self.stride_sec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, annotations, t_start = self.samples[idx]
        t_end = t_start + self.window_sec

        try:
            frames_np, _ = extract_frames_at_fps(video_path, self.fps, self.img_size)
        except Exception:
            frames_np = np.zeros((int(self.fps * self.window_sec), self.img_size, self.img_size, 3), dtype=np.uint8)

        # Extract window frames
        f_start = int(t_start * self.fps)
        f_end = int(t_end * self.fps)
        f_end = min(f_end, len(frames_np))
        window_frames = frames_np[f_start:f_end]

        # Pad if short
        target_len = int(self.window_sec * self.fps)
        if len(window_frames) < target_len:
            pad = np.zeros((target_len - len(window_frames), self.img_size, self.img_size, 3), dtype=np.uint8)
            window_frames = np.concatenate([window_frames, pad], axis=0)
        window_frames = window_frames[:target_len]

        # Apply transform per frame
        frames_t = torch.stack([self.transform(f) for f in window_frames])  # [T,3,H,W]

        # Build classification labels (tIoU >= tiou_thr with any GT annotation)
        cls_labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        best_iou_per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
        best_ann_per_class = [None] * NUM_CLASSES

        for ann in annotations:
            gt_s, gt_e = float(ann["start"]), float(ann["end"])
            inter = max(0.0, min(t_end, gt_e) - max(t_start, gt_s))
            union = max(t_end, gt_e) - min(t_start, gt_s)
            iou = inter / union if union > 0 else 0.0
            bt = norm_bug_type(ann["bug_type"])
            if bt in BUG2IDX and iou >= self.tiou_thr:
                c = BUG2IDX[bt]
                cls_labels[c] = 1.0
                if iou > best_iou_per_class[c]:
                    best_iou_per_class[c] = iou
                    best_ann_per_class[c] = ann

        any_label = float(cls_labels.any())

        # Regression targets: normalized offset from window center to GT boundary
        w_center = t_start + self.window_sec / 2
        reg_targets = np.zeros(2, dtype=np.float32)
        has_reg = 0.0
        if any_label > 0:
            # Use the GT annotation with best overall tIoU
            best_iou_overall = 0.0
            best_ann = None
            for ann in annotations:
                gt_s, gt_e = float(ann["start"]), float(ann["end"])
                inter = max(0.0, min(t_end, gt_e) - max(t_start, gt_s))
                union = max(t_end, gt_e) - min(t_start, gt_s)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou_overall:
                    best_iou_overall = iou
                    best_ann = ann
            if best_ann is not None:
                gt_s, gt_e = float(best_ann["start"]), float(best_ann["end"])
                # Normalize by window_sec: delta from center to boundary / window_sec
                reg_targets[0] = (gt_s - w_center) / self.window_sec  # delta_start
                reg_targets[1] = (gt_e - w_center) / self.window_sec  # delta_end
                has_reg = 1.0

        return (frames_t,
                torch.from_numpy(cls_labels),
                torch.tensor(any_label, dtype=torch.float32),
                torch.from_numpy(reg_targets),
                torch.tensor(has_reg, dtype=torch.float32))


def collate_fn(batch):
    frames, cls, any_l, reg, has_reg = zip(*batch)
    return (torch.stack(frames), torch.stack(cls), torch.stack(any_l),
            torch.stack(reg), torch.stack(has_reg))

# ── Training utilities ────────────────────────────────────────────────────────

class MultiTaskEarlyStopping:
    """
    Stops training when composite score (0.3*F1 + 0.7*mAP) hasn't improved
    for `patience` epochs. Also stops when LR hits `lr_floor`.
    Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks," TPAMI 2022.
    """
    def __init__(self, patience: int = 15, lr_floor: float = 1e-6, w_f1: float = 0.3, w_map: float = 0.7):
        self.patience = patience
        self.lr_floor = lr_floor
        self.w_f1 = w_f1
        self.w_map = w_map
        self.best_score = -1.0
        self.counter = 0

    def composite(self, val_f1: float, val_map: float) -> float:
        return self.w_f1 * val_f1 + self.w_map * val_map

    def step(self, val_f1: float, val_map: float, current_lr: float) -> Tuple[bool, float]:
        score = self.composite(val_f1, val_map)
        stop = False
        if score > self.best_score + 1e-4:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience or current_lr <= self.lr_floor:
                stop = True
        return stop, score


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)


def load_checkpoint_for_resume(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def manage_epoch_checkpoints(checkpoint_dir: Path, keep: int = 3):
    """Keep only the last `keep` epoch checkpoints."""
    epoch_ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    for old in epoch_ckpts[:-keep]:
        old.unlink(missing_ok=True)

# ── Sliding-window inference (for eval and infer modes) ──────────────────────

@torch.no_grad()
def sliding_window_inference(frames_np, model, transform, thr_any, thr_map,
                              target_fps=8.0, window_sec=2.0, stride_sec=1.0,
                              batch_size=16, device="cuda"):
    model.eval()
    window_frames = int(window_sec * target_fps)
    total_frames = frames_np.shape[0]
    duration = total_frames / target_fps
    window_starts = list(range(0, max(1, total_frames - window_frames + 1),
                               max(1, int(stride_sec * target_fps))))
    if not window_starts or window_starts[-1] + window_frames > total_frames:
        window_starts = [max(0, total_frames - window_frames)]

    thr_vec = np.array([thr_map.get(n, 0.5) for n in CANON_BUG_TYPES], dtype=np.float32)
    predictions = []

    for i in range(0, len(window_starts), batch_size):
        batch_ws = window_starts[i: i + batch_size]
        batch_frames = []
        for ws in batch_ws:
            wf = frames_np[ws: ws + window_frames]
            if len(wf) < window_frames:
                pad = np.zeros((window_frames - len(wf), *wf.shape[1:]), dtype=np.uint8)
                wf = np.concatenate([wf, pad], axis=0)
            batch_frames.append(torch.stack([transform(f) for f in wf]))
        batch_t = torch.stack(batch_frames).to(device)

        logits_types, logit_any, reg_offsets, _ = model(batch_t)
        prob_types = torch.sigmoid(logits_types).cpu().numpy()
        prob_any = torch.sigmoid(logit_any).cpu().numpy()

        for j, ws in enumerate(batch_ws):
            t_start = ws / target_fps
            t_end = min((ws + window_frames) / target_fps, duration)
            if prob_any[j] >= thr_any:
                pred_types = (prob_types[j] >= thr_vec).astype(int).tolist()
            else:
                pred_types = [0] * NUM_CLASSES
            predictions.append({
                "t_start": t_start, "t_end": t_end,
                "pred_types": pred_types, "prob_types": prob_types[j].tolist(),
                "prob_any": float(prob_any[j]),
            })
    return predictions


def merge_window_predictions(window_preds, class_names=CANON_BUG_TYPES,
                              min_conf=0.0, min_duration=0.0, max_gap_sec=float("inf")):
    segments = []
    for c_idx, class_name in enumerate(class_names):
        active = [(wp["t_start"], wp["t_end"], wp["prob_types"][c_idx])
                  for wp in window_preds if wp["pred_types"][c_idx] == 1]
        if not active:
            continue
        active.sort(key=lambda x: x[0])
        cur_s, cur_e = active[0][0], active[0][1]
        conf_acc = [active[0][2]]
        for t_s, t_e, conf in active[1:]:
            if t_s - cur_e <= max_gap_sec + 1e-6:
                cur_e = max(cur_e, t_e)
                conf_acc.append(conf)
            else:
                mc = float(np.mean(conf_acc))
                dur = cur_e - cur_s
                if mc >= min_conf and dur >= min_duration:
                    segments.append({"bug_type": OUTPUT_NAMES.get(class_name, class_name),
                                     "start": round(cur_s, 3), "end": round(cur_e, 3),
                                     "duration": round(dur, 3), "confidence": round(mc, 4)})
                cur_s, cur_e, conf_acc = t_s, t_e, [conf]
        mc = float(np.mean(conf_acc))
        dur = cur_e - cur_s
        if mc >= min_conf and dur >= min_duration:
            segments.append({"bug_type": OUTPUT_NAMES.get(class_name, class_name),
                             "start": round(cur_s, 3), "end": round(cur_e, 3),
                             "duration": round(dur, 3), "confidence": round(mc, 4)})
    segments.sort(key=lambda s: (s["start"], s["bug_type"]))
    for i, s in enumerate(segments):
        s["segment_id"] = i
    return segments

# ── Temporal metrics ──────────────────────────────────────────────────────────

def temporal_iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_temporal_metrics(pred_by_class, gt_by_class, tiou_thresholds=(0.3, 0.5, 0.7)):
    per_class = {}
    map_per_thr = {}
    all_ious = []
    total_gt = sum(len(v) for v in gt_by_class.values())
    segment_recall = {}

    for cls in CANON_BUG_TYPES:
        preds = sorted(pred_by_class.get(cls, []), key=lambda x: -x.get("confidence", 0.5))
        gts = gt_by_class.get(cls, [])
        cls_ap = {}
        for thr in tiou_thresholds:
            if not gts or not preds:
                cls_ap[f"AP_{thr}"] = 0.0
                continue
            gt_matched = [False] * len(gts)
            tp_arr, fp_arr = [], []
            for pred in preds:
                best_iou, best_idx = 0.0, -1
                for j, gt in enumerate(gts):
                    iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
                if best_iou >= thr and best_idx >= 0 and not gt_matched[best_idx]:
                    tp_arr.append(1); fp_arr.append(0); gt_matched[best_idx] = True
                else:
                    tp_arr.append(0); fp_arr.append(1)
            tp_c = np.cumsum(tp_arr); fp_c = np.cumsum(fp_arr)
            rec = np.concatenate([[0.0], tp_c / max(len(gts), 1)])
            prec = np.concatenate([[1.0], tp_c / (tp_c + fp_c + 1e-8)])
            cls_ap[f"AP_{thr}"] = round(float(np.sum((rec[1:] - rec[:-1]) * prec[1:])), 4)
        per_class[cls] = cls_ap

        # Collect matched IoUs for mean_tIoU
        for pred in preds:
            best_iou = max((temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                           for gt in gts), default=0.0)
            all_ious.append(best_iou)

    for thr_key, thr in [("mAP_0.3", 0.3), ("mAP_0.5", 0.5), ("mAP_0.7", 0.7)]:
        ap_vals = [per_class[c][f"AP_{thr}"] for c in CANON_BUG_TYPES]
        map_per_thr[thr_key] = round(float(np.mean(ap_vals)), 4)

    for thr in tiou_thresholds:
        matched = 0
        for cls in CANON_BUG_TYPES:
            preds = sorted(pred_by_class.get(cls, []), key=lambda x: -x.get("confidence", 0.5))
            gts = gt_by_class.get(cls, [])
            gt_matched = [False] * len(gts)
            for pred in preds:
                for j, gt in enumerate(gts):
                    if not gt_matched[j] and temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"]) >= thr:
                        gt_matched[j] = True; matched += 1; break
        segment_recall[f"tIoU_{thr}"] = round(matched / total_gt, 4) if total_gt > 0 else 0.0

    # Temporal precision
    temporal_precision = {}
    for thr in tiou_thresholds:
        total_pred = sum(len(pred_by_class.get(c, [])) for c in CANON_BUG_TYPES)
        matched = 0
        for cls in CANON_BUG_TYPES:
            preds = pred_by_class.get(cls, [])
            gts = gt_by_class.get(cls, [])
            gt_matched = [False] * len(gts)
            for pred in preds:
                for j, gt in enumerate(gts):
                    if not gt_matched[j] and temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"]) >= thr:
                        gt_matched[j] = True; matched += 1; break
        temporal_precision[f"tIoU_{thr}"] = round(matched / total_pred, 4) if total_pred > 0 else 0.0

    # Temporal F1
    temporal_f1 = {}
    for thr in tiou_thresholds:
        p = temporal_precision[f"tIoU_{thr}"]
        r = segment_recall[f"tIoU_{thr}"]
        temporal_f1[f"tIoU_{thr}"] = round(2 * p * r / (p + r + 1e-8), 4)

    # Segment over-prediction ratio (global)
    total_pred_global = sum(len(pred_by_class.get(c, [])) for c in CANON_BUG_TYPES)
    over_pred_ratio = round(total_pred_global / max(total_gt, 1), 3)

    return {
        **map_per_thr,
        "mean_tIoU": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        "per_class": per_class,
        "segment_recall": segment_recall,
        "temporal_precision": temporal_precision,
        "temporal_f1": temporal_f1,
        "over_prediction_ratio": over_pred_ratio,
    }

# ── Clip-level multi-label metrics ────────────────────────────────────────────

def compute_multilabel_metrics(y_pred_prob: np.ndarray, y_true: np.ndarray,
                                thresholds: np.ndarray = None) -> Dict:
    """Full suite of multi-label classification metrics."""
    if thresholds is None:
        thresholds = np.full(NUM_CLASSES, 0.5)
    y_pred = (y_pred_prob >= thresholds).astype(int)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)[2]
    per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    h_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)

    clip_map_per_cls = []
    roc_auc_per_cls = []
    for c in range(NUM_CLASSES):
        if y_true[:, c].sum() > 0:
            clip_map_per_cls.append(average_precision_score(y_true[:, c], y_pred_prob[:, c]))
            try:
                roc_auc_per_cls.append(roc_auc_score(y_true[:, c], y_pred_prob[:, c]))
            except Exception:
                roc_auc_per_cls.append(0.0)
        else:
            clip_map_per_cls.append(0.0)
            roc_auc_per_cls.append(0.0)

    try:
        lrap = label_ranking_average_precision_score(y_true, y_pred_prob)
    except Exception:
        lrap = 0.0

    metrics = {
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "micro_precision": round(float(micro_p), 4),
        "micro_recall": round(float(micro_r), 4),
        "hamming_loss": round(float(h_loss), 4),
        "subset_accuracy": round(float(subset_acc), 4),
        "clip_mAP": round(float(np.mean(clip_map_per_cls)), 4),
        "mean_roc_auc": round(float(np.mean(roc_auc_per_cls)), 4),
        "lrap": round(float(lrap), 4),
        "per_class": {},
    }
    for c, name in enumerate(CANON_BUG_TYPES):
        metrics["per_class"][name] = {
            "precision": round(float(per_cls_p[c]), 4),
            "recall": round(float(per_cls_r[c]), 4),
            "f1": round(float(per_cls_f1[c]), 4),
            "clip_AP": round(float(clip_map_per_cls[c]), 4),
            "roc_auc": round(float(roc_auc_per_cls[c]), 4),
        }
    return metrics


def tune_thresholds(y_pred_prob: np.ndarray, y_true: np.ndarray,
                    thr_range=np.arange(0.1, 0.91, 0.05)) -> Tuple[np.ndarray, float]:
    """Per-class threshold sweep to maximize micro F1 on val set."""
    best_thrs = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in thr_range:
            pred = (y_pred_prob[:, c] >= t).astype(int)
            tp = ((pred == 1) & (y_true[:, c] == 1)).sum()
            fp = ((pred == 1) & (y_true[:, c] == 0)).sum()
            fn = ((pred == 0) & (y_true[:, c] == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thrs[c] = best_t
    any_best_f1, any_best_t = 0.0, 0.5
    y_any_true = (y_true.sum(axis=1) > 0).astype(int)
    y_any_prob = y_pred_prob.max(axis=1)
    for t in thr_range:
        pred = (y_any_prob >= t).astype(int)
        tp = ((pred == 1) & (y_any_true == 1)).sum()
        fp = ((pred == 1) & (y_any_true == 0)).sum()
        fn = ((pred == 0) & (y_any_true == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        if f1 > any_best_f1:
            any_best_f1, any_best_t = f1, t
    return best_thrs, any_best_t

# ── Visualizations ────────────────────────────────────────────────────────────

def save_visualizations(test_metrics: Dict, pred_by_class: Dict, gt_by_class: Dict,
                        per_video_results: List, out_dir: Path, tb_writer=None, wandb_run=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. mAP vs tIoU smooth curve
    thr_range = np.arange(0.1, 0.91, 0.05)
    map_vals = []
    for thr in thr_range:
        ap_per_cls = []
        for cls in CANON_BUG_TYPES:
            preds = sorted(pred_by_class.get(cls, []), key=lambda x: -x.get("confidence", 0.5))
            gts = gt_by_class.get(cls, [])
            if not gts or not preds:
                ap_per_cls.append(0.0); continue
            gt_matched = [False] * len(gts)
            tp_arr, fp_arr = [], []
            for pred in preds:
                best_iou, best_idx = 0.0, -1
                for j, gt in enumerate(gts):
                    iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
                if best_iou >= thr and best_idx >= 0 and not gt_matched[best_idx]:
                    tp_arr.append(1); fp_arr.append(0); gt_matched[best_idx] = True
                else:
                    tp_arr.append(0); fp_arr.append(1)
            tp_c = np.cumsum(tp_arr); fp_c = np.cumsum(fp_arr)
            rec = np.concatenate([[0.0], tp_c / max(len(gts), 1)])
            prec = np.concatenate([[1.0], tp_c / (tp_c + fp_c + 1e-8)])
            ap_per_cls.append(float(np.sum((rec[1:] - rec[:-1]) * prec[1:])))
        map_vals.append(float(np.mean(ap_per_cls)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr_range, map_vals, marker="o", linewidth=2, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="tIoU=0.5")
    ax.set_xlabel("tIoU Threshold"); ax.set_ylabel("mAP"); ax.set_title("mAP vs tIoU Threshold")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "map_vs_tiou.png"
    fig.savefig(path, dpi=120); plt.close(fig)
    if tb_writer: tb_writer.add_figure("test/map_vs_tiou", plt.figure(figsize=(8, 5)))
    if wandb_run: wandb_run.log({"test/map_vs_tiou": wandb.Image(str(path))})

    # 2. tIoU histogram
    all_ious = []
    for cls in CANON_BUG_TYPES:
        preds = pred_by_class.get(cls, [])
        gts = gt_by_class.get(cls, [])
        for pred in preds:
            best = max((temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"]) for gt in gts), default=0.0)
            all_ious.append(best)
    if all_ious:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_ious, bins=20, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="tIoU=0.5")
        ax.set_xlabel("tIoU"); ax.set_ylabel("Count"); ax.set_title("tIoU Distribution (pred vs GT)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "tiou_histogram.png"
        fig.savefig(path, dpi=120); plt.close(fig)
        if wandb_run: wandb_run.log({"test/tiou_histogram": wandb.Image(str(path))})

    # 3. Per-video over-prediction ratio
    if per_video_results:
        vids = [v["video_id"] for v in per_video_results]
        ratios = [v["num_pred_segments"] / max(v["num_gt_annotations"], 1) for v in per_video_results]
        vids_sorted = [v for _, v in sorted(zip(ratios, vids), reverse=True)]
        ratios_sorted = sorted(ratios, reverse=True)
        fig, ax = plt.subplots(figsize=(max(10, len(vids) * 0.3), 5))
        colors = ["tomato" if r > 1.5 else "steelblue" for r in ratios_sorted]
        ax.bar(range(len(vids_sorted)), ratios_sorted, color=colors)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="1:1 ratio")
        ax.set_xticks(range(len(vids_sorted)))
        ax.set_xticklabels([v.split("_")[-1] for v in vids_sorted], rotation=90, fontsize=7)
        ax.set_ylabel("Pred / GT segments"); ax.set_title("Segment Over-Prediction Ratio (per video)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = out_dir / "over_prediction_ratio.png"
        fig.savefig(path, dpi=120); plt.close(fig)
        if wandb_run: wandb_run.log({"test/over_prediction_ratio": wandb.Image(str(path))})

    # 4. Per-video F1 scatter
    if per_video_results:
        x = [v["num_gt_annotations"] for v in per_video_results]
        y = [v["clip_f1_types"] for v in per_video_results]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.7, color="steelblue")
        ax.set_xlabel("Num GT Annotations"); ax.set_ylabel("Clip F1 (types)")
        ax.set_title("Per-Video: Clip F1 vs GT Density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "per_video_f1_scatter.png"
        fig.savefig(path, dpi=120); plt.close(fig)
        if wandb_run: wandb_run.log({"test/per_video_f1_scatter": wandb.Image(str(path))})

    # 5. Per-class AP bar chart
    ap_vals_03 = [test_metrics["temporal"]["per_class"][c]["AP_0.3"] for c in CANON_BUG_TYPES]
    ap_vals_05 = [test_metrics["temporal"]["per_class"][c]["AP_0.5"] for c in CANON_BUG_TYPES]
    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, ap_vals_03, 0.4, label="AP@0.3", color="steelblue")
    ax.bar(x + 0.2, ap_vals_05, 0.4, label="AP@0.5", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(CANON_BUG_TYPES, rotation=15)
    ax.set_ylabel("AP"); ax.set_title("Per-Class AP@0.3 and AP@0.5")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "per_class_ap.png"
    fig.savefig(path, dpi=120); plt.close(fig)
    if wandb_run: wandb_run.log({"test/per_class_ap": wandb.Image(str(path))})

# ── One epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, asl_loss, device, window_sec,
                    lambda_cls, lambda_reg, lambda_any, epoch, tb_writer=None):
    model.train()
    total_loss = total_cls = total_reg = total_any = 0.0
    for frames, cls_labels, any_labels, reg_targets, has_reg in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        frames = frames.to(device)
        cls_labels = cls_labels.to(device)
        any_labels = any_labels.to(device)
        reg_targets = reg_targets.to(device)
        has_reg = has_reg.to(device)

        logits_types, logit_any, reg_offsets, _ = model(frames)

        l_cls = asl_loss(logits_types, cls_labels)
        l_any = focal_loss_binary(logit_any, any_labels)
        l_reg = diou_loss_temporal(reg_offsets, reg_targets, window_sec=window_sec, mask=has_reg)

        loss = lambda_cls * l_cls + lambda_reg * l_reg + lambda_any * l_any
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += l_cls.item()
        total_reg += l_reg.item()
        total_any += l_any.item()

    n = max(len(loader), 1)
    metrics = {"loss": total_loss / n, "cls": total_cls / n,
               "reg": total_reg / n, "any": total_any / n}
    if tb_writer:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"train/{k}", v, epoch)
    return metrics


@torch.no_grad()
def validate(model, loader, asl_loss, device, window_sec, lambda_cls, lambda_reg, lambda_any,
             epoch, tb_writer=None):
    model.eval()
    total_loss = 0.0
    all_prob = []
    all_gt = []
    for frames, cls_labels, any_labels, reg_targets, has_reg in loader:
        frames = frames.to(device)
        cls_labels = cls_labels.to(device)
        any_labels = any_labels.to(device)
        reg_targets = reg_targets.to(device)
        has_reg = has_reg.to(device)

        logits_types, logit_any, reg_offsets, _ = model(frames)
        l_cls = asl_loss(logits_types, cls_labels)
        l_any = focal_loss_binary(logit_any, any_labels)
        l_reg = diou_loss_temporal(reg_offsets, reg_targets, window_sec=window_sec, mask=has_reg)
        loss = lambda_cls * l_cls + lambda_reg * l_reg + lambda_any * l_any
        total_loss += loss.item()
        all_prob.append(torch.sigmoid(logits_types).cpu().numpy())
        all_gt.append(cls_labels.cpu().numpy())

    all_prob = np.concatenate(all_prob, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    thrs, _ = tune_thresholds(all_prob, all_gt)
    y_pred = (all_prob >= thrs).astype(int)
    tp = ((y_pred == 1) & (all_gt == 1)).sum()
    fp = ((y_pred == 1) & (all_gt == 0)).sum()
    fn = ((y_pred == 0) & (all_gt == 1)).sum()
    micro_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))

    # Simplified val mAP (clip-level AP as proxy for temporal mAP during training)
    clip_aps = []
    for c in range(NUM_CLASSES):
        if all_gt[:, c].sum() > 0:
            clip_aps.append(average_precision_score(all_gt[:, c], all_prob[:, c]))
        else:
            clip_aps.append(0.0)
    val_map_proxy = float(np.mean(clip_aps))

    avg_loss = total_loss / max(len(loader), 1)
    if tb_writer:
        tb_writer.add_scalar("val/loss", avg_loss, epoch)
        tb_writer.add_scalar("val/micro_f1", micro_f1, epoch)
        tb_writer.add_scalar("val/clip_mAP_proxy", val_map_proxy, epoch)

    return {"loss": avg_loss, "micro_f1": micro_f1, "clip_mAP_proxy": val_map_proxy,
            "thresholds": thrs}

# ── Test evaluation ───────────────────────────────────────────────────────────

def run_test(model, args, device, thresholds, out_dir: Path,
             tb_writer=None, wandb_run=None):
    """Full test-set evaluation: sliding window → segment metrics + clip metrics."""
    with open(args.temporal_dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    splits = dataset.get("metadata", {}).get("splits", dataset.get("splits", {}))
    test_ids = set(splits.get("test", []))
    test_videos = [v for v in dataset["videos"] if v["video_id"] in test_ids] if test_ids else \
                  [v for v in dataset["videos"] if v.get("split") == "test"]

    print(f"[test] Evaluating {len(test_videos)} videos...")
    transform = get_transform(args.img_size)
    video_root = Path(args.video_root)

    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    per_video_results = []
    all_prob, all_gt_cls = [], []

    for vid in tqdm(test_videos, desc="[test]"):
        video_path = video_root / vid["video_path"]
        if not video_path.exists():
            continue
        annotations = vid.get("annotations", [])
        try:
            frames_np, duration = extract_frames_at_fps(str(video_path), args.fps, args.img_size)
        except Exception as e:
            print(f"[WARN] {vid['video_id']}: {e}")
            continue

        window_preds = sliding_window_inference(
            frames_np, model, transform,
            thr_any=float(thresholds[-1]) if len(thresholds) > NUM_CLASSES else 0.5,
            thr_map={n: float(thresholds[i]) for i, n in enumerate(CANON_BUG_TYPES)},
            target_fps=args.fps, window_sec=args.window_sec, stride_sec=args.stride_sec,
            batch_size=args.batch_size, device=device,
        )

        # Clip-level labels for multi-label metrics
        for wp in window_preds:
            t_s, t_e = wp["t_start"], wp["t_end"]
            gt_types = np.zeros(NUM_CLASSES, dtype=int)
            for ann in annotations:
                if max(0.0, min(t_e, ann["end"]) - max(t_s, ann["start"])) > 0:
                    bt = norm_bug_type(ann["bug_type"])
                    if bt in BUG2IDX:
                        gt_types[BUG2IDX[bt]] = 1
            all_prob.append(np.array(wp["prob_types"]))
            all_gt_cls.append(gt_types)

        pred_segments = merge_window_predictions(
            window_preds, CANON_BUG_TYPES,
            min_conf=getattr(args, "min_conf", 0.0),
            min_duration=getattr(args, "min_duration", 0.0),
            max_gap_sec=getattr(args, "max_gap", float("inf")),
        )

        for seg in pred_segments:
            c_key = norm_bug_type(seg["bug_type"])
            pred_by_class[c_key].append({
                "start": seg["start"], "end": seg["end"],
                "confidence": seg.get("confidence", 0.5),
            })
        for ann in annotations:
            c_key = norm_bug_type(ann["bug_type"])
            gt_by_class[c_key].append({"start": ann["start"], "end": ann["end"]})

        # Per-video clip F1
        pv_pred = np.stack(all_prob[-len(window_preds):])
        pv_gt = np.stack(all_gt_cls[-len(window_preds):])
        y_pred_pv = (pv_pred >= thresholds[:NUM_CLASSES]).astype(int)
        tp = ((y_pred_pv == 1) & (pv_gt == 1)).sum()
        fp = ((y_pred_pv == 1) & (pv_gt == 0)).sum()
        fn = ((y_pred_pv == 0) & (pv_gt == 1)).sum()
        per_video_results.append({
            "video_id": vid["video_id"],
            "duration": round(float(vid.get("duration", 0)), 3),
            "num_windows": len(window_preds),
            "num_pred_segments": len(pred_segments),
            "num_gt_annotations": len(annotations),
            "clip_f1_types": round(float(2 * tp / (2 * tp + fp + fn + 1e-8)), 4),
        })

    all_prob_np = np.stack(all_prob) if all_prob else np.zeros((1, NUM_CLASSES))
    all_gt_np = np.stack(all_gt_cls) if all_gt_cls else np.zeros((1, NUM_CLASSES), dtype=int)

    clip_metrics = compute_multilabel_metrics(all_prob_np, all_gt_np, thresholds[:NUM_CLASSES])
    temporal_metrics = compute_temporal_metrics(dict(pred_by_class), dict(gt_by_class))

    test_metrics = {
        "clip_level": clip_metrics,
        "temporal": temporal_metrics,
        "per_video": per_video_results,
    }

    # Save metrics
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Log to TensorBoard / W&B
    flat = {
        "test/clip_micro_f1": clip_metrics["micro_f1"],
        "test/clip_mAP": clip_metrics["clip_mAP"],
        "test/mAP_0.3": temporal_metrics["mAP_0.3"],
        "test/mAP_0.5": temporal_metrics["mAP_0.5"],
        "test/mAP_0.7": temporal_metrics["mAP_0.7"],
        "test/mean_tIoU": temporal_metrics["mean_tIoU"],
        "test/over_prediction_ratio": temporal_metrics["over_prediction_ratio"],
    }
    if tb_writer:
        for k, v in flat.items():
            tb_writer.add_scalar(k, v, 0)
    if wandb_run:
        wandb_run.log(flat)

    save_visualizations(test_metrics, dict(pred_by_class), dict(gt_by_class),
                        per_video_results, out_dir / "visualizations",
                        tb_writer=tb_writer, wandb_run=wandb_run)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Clip micro F1       : {clip_metrics['micro_f1']:.4f}")
    print(f"  Clip mAP (clip)     : {clip_metrics['clip_mAP']:.4f}")
    print(f"  mAP@0.3             : {temporal_metrics['mAP_0.3']:.4f}")
    print(f"  mAP@0.5             : {temporal_metrics['mAP_0.5']:.4f}")
    print(f"  mAP@0.7             : {temporal_metrics['mAP_0.7']:.4f}")
    print(f"  Mean tIoU           : {temporal_metrics['mean_tIoU']:.4f}")
    print(f"  Over-pred ratio     : {temporal_metrics['over_prediction_ratio']:.3f}")
    print(f"  Hamming loss        : {clip_metrics['hamming_loss']:.4f}")
    print(f"  Subset accuracy     : {clip_metrics['subset_accuracy']:.4f}")
    print(f"  LRAP                : {clip_metrics['lrap']:.4f}")
    print("=" * 60)
    return test_metrics

# ── Build model from hparams ──────────────────────────────────────────────────

def build_model_and_optim(hparams: Dict, device: torch.device, pretrain_ckpt: Optional[str] = None):
    model = BugBiLSTMLocalization(dropout=hparams.get("dropout", 0.3)).to(device)
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        # Load compatible keys only (CNN + LSTM layers)
        own_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
        print(f"[init] Loaded {loaded}/{len(own_state)} layers from pretrain checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=hparams.get("lr", 1e-4),
                                   weight_decay=hparams.get("weight_decay", 1e-4))
    return model, optimizer

# ── HPO objective ─────────────────────────────────────────────────────────────

def hpo_objective(trial, args, device, train_loader, val_loader):
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "lambda_cls": trial.suggest_float("lambda_cls", 0.5, 2.0),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.5, 2.0),
        "lambda_any": trial.suggest_float("lambda_any", 0.2, 1.0),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "stride_sec": trial.suggest_categorical("stride_sec", [0.5, 1.0]),
    }

    model, optimizer = build_model_and_optim(hparams, device, args.pretrain_checkpoint)
    asl_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    es = MultiTaskEarlyStopping(patience=5)  # short patience for HPO trials

    for epoch in range(1, 11):  # max 10 epochs per trial
        train_one_epoch(model, train_loader, optimizer, asl_loss, device,
                        args.window_sec, hparams["lambda_cls"], hparams["lambda_reg"],
                        hparams["lambda_any"], epoch)
        val_m = validate(model, val_loader, asl_loss, device, args.window_sec,
                         hparams["lambda_cls"], hparams["lambda_reg"], hparams["lambda_any"], epoch)
        composite = 0.3 * val_m["micro_f1"] + 0.7 * val_m["clip_mAP_proxy"]
        trial.report(composite, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        stop, _ = es.step(val_m["micro_f1"], val_m["clip_mAP_proxy"], optimizer.param_groups[0]["lr"])
        if stop:
            break

    val_m = validate(model, val_loader, asl_loss, device, args.window_sec,
                     hparams["lambda_cls"], hparams["lambda_reg"], hparams["lambda_any"], 0)
    return 0.3 * val_m["micro_f1"] + 0.7 * val_m["clip_mAP_proxy"]

# ── Main train loop ───────────────────────────────────────────────────────────

def run_train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    checkpoint_dir = Path(args.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logdir)
    (log_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    hpo_log = log_dir / "hpo_progress.log"

    # W&B
    use_wandb = (not getattr(args, "no_wandb", False)) and WANDB_AVAILABLE
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(project=getattr(args, "wandb_project", "localization-bugs"),
                               name=f"model1_bilstm_{time.strftime('%m%d_%H%M')}",
                               config=vars(args), tags=["model1", "bilstm", "localization"])

    # Load dataset
    with open(args.temporal_dataset, "r") as f:
        dataset = json.load(f)
    splits = dataset.get("metadata", {}).get("splits", dataset.get("splits", {}))
    all_videos = {v["video_id"]: v for v in dataset["videos"]}
    train_ids = set(splits.get("train", []))
    val_ids = set(splits.get("val", []))
    train_vids = [all_videos[i] for i in train_ids if i in all_videos]
    val_vids = [all_videos[i] for i in val_ids if i in all_videos]
    print(f"[train] Train: {len(train_vids)} | Val: {len(val_vids)} videos")

    hparams = {
        "lr": args.lr, "lambda_cls": args.lambda_cls, "lambda_reg": args.lambda_reg,
        "lambda_any": args.lambda_any, "dropout": args.dropout,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "stride_sec": args.stride_sec,
    }

    # HPO phase
    if not args.pilot and args.hpo_trials > 0 and OPTUNA_AVAILABLE:
        print(f"\n[HPO] Starting {args.hpo_trials} Optuna TPE trials...")
        study_path = str(log_dir / "optuna_study.db")
        study = optuna.create_study(
            study_name="model1_bilstm",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
        )
        remaining = args.hpo_trials - len([t for t in study.trials
                                           if t.state == optuna.trial.TrialState.COMPLETE])
        if remaining > 0:
            # Quick loaders for HPO (smaller batch for speed)
            hpo_stride = 1.0
            hpo_train_ds = TemporalLocDataset(train_vids[:50], args.video_root,
                                              window_sec=args.window_sec, stride_sec=hpo_stride,
                                              fps=args.fps, img_size=args.img_size, split="train")
            hpo_val_ds = TemporalLocDataset(val_vids[:20], args.video_root,
                                            window_sec=args.window_sec, stride_sec=hpo_stride,
                                            fps=args.fps, img_size=args.img_size, split="val")
            hpo_train_loader = DataLoader(hpo_train_ds, batch_size=8, shuffle=True,
                                          num_workers=args.num_workers, collate_fn=collate_fn)
            hpo_val_loader = DataLoader(hpo_val_ds, batch_size=8, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=collate_fn)
            for trial_num in range(remaining):
                try:
                    study.optimize(
                        lambda trial: hpo_objective(trial, args, device, hpo_train_loader, hpo_val_loader),
                        n_trials=1, gc_after_trial=True,
                    )
                    best = study.best_trial
                    msg = f"Trial {len(study.trials)}/{args.hpo_trials} done, best composite: {best.value:.4f}\n"
                    print(f"[HPO] {msg.strip()}")
                    with open(hpo_log, "a") as f:
                        f.write(msg)
                except Exception as e:
                    print(f"[HPO] Trial failed: {e}")

        best_params = study.best_params
        hparams.update(best_params)
        with open(log_dir / "best_hparams.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"[HPO] Best hparams: {best_params}")
        if wandb_run:
            wandb_run.config.update({"hpo_best": best_params})

    # Build full loaders
    train_ds = TemporalLocDataset(train_vids, args.video_root,
                                  window_sec=args.window_sec, stride_sec=hparams["stride_sec"],
                                  fps=args.fps, img_size=args.img_size, split="train")
    val_ds = TemporalLocDataset(val_vids, args.video_root,
                                window_sec=args.window_sec, stride_sec=hparams["stride_sec"],
                                fps=args.fps, img_size=args.img_size, split="val")
    train_loader = DataLoader(train_ds, batch_size=int(hparams["batch_size"]), shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(hparams["batch_size"]), shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model, optimizer = build_model_and_optim(hparams, device, args.pretrain_checkpoint)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6)
    asl_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    es = MultiTaskEarlyStopping(patience=15, lr_floor=1e-6)

    start_epoch = 1
    best_checkpoints = {k: {"score": -1.0, "path": checkpoint_dir / f"{k}.pt"}
                        for k in ("best_composite", "best_f1", "best_mAP")}

    # Resume if checkpoint provided
    if args.checkpoint and Path(args.checkpoint).exists():
        start_epoch, _ = load_checkpoint_for_resume(
            args.checkpoint, model, optimizer, scheduler, str(device))
        start_epoch += 1
        print(f"[train] Resumed from epoch {start_epoch - 1}")

    # Architecture summary
    if TORCHINFO_AVAILABLE:
        dummy = torch.zeros(1, int(args.window_sec * args.fps), 3, args.img_size, args.img_size).to(device)
        summary_str = str(torchinfo.summary(model, input_data=dummy, verbose=0))
        with open(log_dir / "architecture_summary.txt", "w") as f:
            f.write(summary_str)
        print(f"[train] Architecture summary → {log_dir}/architecture_summary.txt")

    max_epochs = 5 if args.pilot else args.epochs
    best_val_thresholds = np.full(NUM_CLASSES + 1, 0.5)  # last entry = any_thr

    for epoch in range(start_epoch, max_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, asl_loss, device,
                                  args.window_sec, hparams["lambda_cls"],
                                  hparams["lambda_reg"], hparams["lambda_any"],
                                  epoch, tb_writer)
        val_m = validate(model, val_loader, asl_loss, device, args.window_sec,
                         hparams["lambda_cls"], hparams["lambda_reg"], hparams["lambda_any"],
                         epoch, tb_writer)

        composite = 0.3 * val_m["micro_f1"] + 0.7 * val_m["clip_mAP_proxy"]
        scheduler.step(composite)
        current_lr = optimizer.param_groups[0]["lr"]
        tb_writer.add_scalar("train/lr", current_lr, epoch)

        print(f"Epoch {epoch:3d} | loss={train_m['loss']:.4f} | "
              f"val_F1={val_m['micro_f1']:.4f} | val_cMAP={val_m['clip_mAP_proxy']:.4f} | "
              f"composite={composite:.4f} | lr={current_lr:.2e}")

        if wandb_run:
            wandb_run.log({"epoch": epoch, "train/loss": train_m["loss"],
                           "val/micro_f1": val_m["micro_f1"],
                           "val/clip_mAP_proxy": val_m["clip_mAP_proxy"],
                           "val/composite": composite, "train/lr": current_lr})

        # Save best checkpoints
        for key, score in [("best_composite", composite),
                            ("best_f1", val_m["micro_f1"]),
                            ("best_mAP", val_m["clip_mAP_proxy"])]:
            if score > best_checkpoints[key]["score"]:
                best_checkpoints[key]["score"] = score
                save_checkpoint(model, optimizer, scheduler, epoch,
                                {"composite": composite, "f1": val_m["micro_f1"],
                                 "mAP": val_m["clip_mAP_proxy"]},
                                best_checkpoints[key]["path"])
                if key == "best_composite":
                    best_val_thresholds = np.append(val_m["thresholds"], 0.5)

        # Save epoch checkpoint (rolling last 3)
        epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch,
                        {"composite": composite}, epoch_ckpt)
        manage_epoch_checkpoints(checkpoint_dir, keep=3)

        if args.pilot and epoch >= 5:
            print(f"[pilot] 5 epochs done. composite={composite:.4f} | Exiting pilot mode.")
            tb_writer.close()
            if wandb_run:
                wandb_run.finish()
            return

        stop, _ = es.step(val_m["micro_f1"], val_m["clip_mAP_proxy"], current_lr)
        if stop:
            print(f"[train] Early stopping at epoch {epoch} (composite patience={es.patience}, lr={current_lr:.2e})")
            break

    tb_writer.close()

    # Final test evaluation with best_composite checkpoint
    if not args.pilot:
        print("\n[test] Loading best_composite checkpoint for test evaluation...")
        best_ckpt_path = best_checkpoints["best_composite"]["path"]
        load_checkpoint_for_resume(str(best_ckpt_path), model, device=str(device))
        out_dir = Path(args.logdir)
        test_m = run_test(model, args, device, best_val_thresholds, out_dir,
                          tb_writer=None, wandb_run=wandb_run)
        if wandb_run:
            wandb_run.finish()


# ── Inference mode ────────────────────────────────────────────────────────────

def run_infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BugBiLSTMLocalization().to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    thr_map = ckpt.get("thr_map", {n: 0.5 for n in CANON_BUG_TYPES})
    thr_any = float(ckpt.get("thr_any", 0.5))

    transform = get_transform(args.img_size)
    frames_np, duration = extract_frames_at_fps(args.video_path, args.fps, args.img_size)
    window_preds = sliding_window_inference(
        frames_np, model, transform, thr_any=thr_any, thr_map=thr_map,
        target_fps=args.fps, window_sec=args.window_sec, stride_sec=args.stride_sec,
        batch_size=args.batch_size, device=device,
    )
    segments = merge_window_predictions(
        window_preds, CANON_BUG_TYPES,
        min_conf=getattr(args, "min_conf", 0.0),
        min_duration=getattr(args, "min_duration", 0.0),
        max_gap_sec=getattr(args, "max_gap", float("inf")),
    )
    result = {
        "video_id": Path(args.video_path).stem,
        "video_path": args.video_path,
        "duration": round(duration, 3),
        "annotations": [{"bug_type": s["bug_type"], "start": s["start"],
                          "end": s["end"], "duration": s["duration"],
                          "segment_id": s["segment_id"]} for s in segments],
    }
    out_path = args.inference_out or "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[infer] {len(segments)} segments → {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Model 1: BugBiLSTM + Temporal Regression",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["train", "infer"], default="train")

    # Paths
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--temporal-dataset", type=str, default="../localization/temporal_bug_dataset.json")
    ap.add_argument("--video-path", type=str, default=None, help="infer mode only")
    ap.add_argument("--save-dir", type=str, default="checkpoints/")
    ap.add_argument("--logdir", type=str, default="logs/")
    ap.add_argument("--inference-out", type=str, default="result.json")
    ap.add_argument("--checkpoint", type=str, default=None, help="Resume or infer")
    ap.add_argument("--pretrain-checkpoint", type=str, default=None,
                    help="Existing BugBiLSTM checkpoint for weight initialization")

    # Sliding window
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--img-size", type=int, default=224)

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-any", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")

    # HPO / pilot
    ap.add_argument("--pilot", action="store_true", help="Run 5 epochs only for architecture sanity check")
    ap.add_argument("--hpo-trials", type=int, default=0, help="Optuna HPO trials (0 = skip HPO)")

    # Segment post-processing (also used during test eval)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max-gap", type=float, default=float("inf"))

    # Tracking
    ap.add_argument("--wandb-project", type=str, default="localization-bugs")
    ap.add_argument("--no-wandb", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        if not args.video_path:
            raise ValueError("--video-path required for --mode infer")
        run_infer(args)


if __name__ == "__main__":
    main()
