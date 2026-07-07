#!/usr/bin/env python3
"""
Model 2: MS-TCN Multi-Label Temporal Segmentation
==================================================
Architecture: ResNet18 (per-frame features) + 4-stage MS-TCN (dilated 1D conv).
Treats temporal localization as dense frame-level segmentation — predicts
multi-label bug presence for every frame. Segments are extracted via run-length
encoding of the per-frame predictions.

Loss:
  - ASL   per frame, per stage (Ridnik et al., ICCV 2021)
  - Truncated MSE Smoothing — penalizes rapid label changes between adjacent frames
    (Abu Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for
     Action Segmentation," CVPR 2019)

Modes:
  --mode train   [HPO →] train → val → test (one go)
  --mode infer   Single-video inference → result.json
"""

import os, sys, json, math, time, argparse, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool

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
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score, hamming_loss, accuracy_score, label_ranking_average_precision_score,
)

try:
    import wandb; WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torchinfo; TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

_BACKENDS: Dict[str, bool] = {}
try:
    import decord; decord.bridge.set_bridge("torch"); _BACKENDS["decord"] = True
except Exception: _BACKENDS["decord"] = False
try:
    import cv2; _BACKENDS["opencv"] = True
except Exception: _BACKENDS["opencv"] = False

# ── Constants ─────────────────────────────────────────────────────────────────

CANON_BUG_TYPES = ["z-clipping", "corrupted_texture", "geometry_corruption", "z-fighting", "boundary_hole"]
_ALIAS = {"z_clipping": "z-clipping", "z_fighting": "z-fighting", "boundary_hole": "boundary_hole",
          "corrupted_texture": "corrupted_texture", "geometry_corruption": "geometry_corruption"}
BUG2IDX = {b: i for i, b in enumerate(CANON_BUG_TYPES)}
NUM_CLASSES = len(CANON_BUG_TYPES)
OUTPUT_NAMES = {"z-clipping": "z_clipping", "z-fighting": "z_fighting",
                "corrupted_texture": "corrupted_texture",
                "geometry_corruption": "geometry_corruption", "boundary_hole": "boundary_hole"}

def norm_bug_type(s: str) -> str:
    s2 = s.strip().lower().replace(" ", "_").replace("-", "_")
    canon = _ALIAS.get(s2, s2)
    if canon in ("z_clipping", "z-clipping"): return "z-clipping"
    if canon in ("z_fighting", "z-fighting"): return "z-fighting"
    return canon

# ── Video IO ──────────────────────────────────────────────────────────────────

def extract_frames_at_fps(video_path: str, target_fps: float = 8.0,
                          img_size: int = 224) -> Tuple[np.ndarray, float]:
    if _BACKENDS.get("decord"):
        try:
            vr = decord.VideoReader(video_path, width=img_size, height=img_size)
            native_fps = float(vr.get_avg_fps())
            total = len(vr); duration = total / native_fps
            stride = max(1, round(native_fps / target_fps))
            indices = list(range(0, total, stride))
            return vr.get_batch(indices).numpy().astype(np.uint8), duration
        except Exception:
            pass
    if _BACKENDS.get("opencv"):
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration = total / native_fps
        stride = max(1, round(native_fps / target_fps))
        frames_list = []; idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % stride == 0:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
                frames_list.append(frame)
            idx += 1
        cap.release()
        return np.stack(frames_list).astype(np.uint8), duration
    raise RuntimeError("No video backend. Install decord or opencv.")

def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ── Frame cache ───────────────────────────────────────────────────────────────
# Each video is decoded ONCE at (fps, img_size) and stored as a uint8 .npy array
# [T,H,W,3]; chunks are then read as memory-mapped slices instead of re-decoding
# the full video for every chunk sample.

def frame_cache_name(rel_video_path: str) -> str:
    key = rel_video_path.replace("\\", "/").strip("/")
    key = os.path.splitext(key)[0].replace("/", "__")
    return key + ".npy"


@lru_cache(maxsize=None)
def open_frame_cache(path: str):
    return np.load(path, mmap_mode="r")


def _cache_one_video(task):
    video_path, out_path, fps, img_size = task
    out_path = Path(out_path)
    if out_path.exists():
        return (out_path.name, "cached")
    try:
        frames, _ = extract_frames_at_fps(video_path, fps, img_size)
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


def load_video_frames_for_eval(video_path: str, rel_path: str, args):
    """Full-video frames for eval: memmapped from the frame cache when available,
    otherwise decoded once. Returns (frames [T,H,W,3] uint8, duration_sec)."""
    cache_dir = getattr(args, "frame_cache_dir", None)
    if cache_dir:
        cp = Path(cache_dir) / frame_cache_name(rel_path)
        if cp.exists():
            arr = open_frame_cache(str(cp))
            return arr, arr.shape[0] / args.fps
    return extract_frames_at_fps(video_path, args.fps, args.img_size)


# ── GPU-side normalization ────────────────────────────────────────────────────

_norm_stats = {}

def normalize_frames_gpu(frames_u8: torch.Tensor, device) -> torch.Tensor:
    """uint8 [B,T,H,W,3] on CPU → normalized float32 [B,T,3,H,W] on device."""
    key = str(device)
    if key not in _norm_stats:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        _norm_stats[key] = (mean, std)
    mean, std = _norm_stats[key]
    x = frames_u8.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)
    return x.sub_(mean).div_(std)

# ── Model ─────────────────────────────────────────────────────────────────────

class DilatedResidualLayer(nn.Module):
    """Single dilated causal conv layer with residual connection."""
    def __init__(self, num_f_maps: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(num_f_maps, num_f_maps, 3,
                              padding=dilation, dilation=dilation)
        self.norm = nn.InstanceNorm1d(num_f_maps, track_running_stats=False)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x))) + x


class SingleStageModel(nn.Module):
    """One MS-TCN refinement stage: dilations 1,2,4,8,16."""
    def __init__(self, num_layers: int, num_f_maps: int, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_f_maps, 2 ** i) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

    def forward(self, x):
        f = self.conv_in(x)
        for layer in self.layers:
            f = layer(f)
        return self.conv_out(f)  # [B, C, T]


class MSTCNLocalization(nn.Module):
    """
    ResNet18 frame feature extractor + 4-stage MS-TCN for per-frame multi-label
    bug segmentation.

    MS-TCN: Abu Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network
    for Action Segmentation," CVPR 2019.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, num_stages: int = 4,
                 num_layers: int = 10, num_f_maps: int = 64,
                 feat_dim: int = 512, pretrained: bool = True):
        super().__init__()
        enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.cnn = nn.Sequential(*list(enc.children())[:-1])
        self.feat_dim = feat_dim
        self.num_stages = num_stages

        # Stage 1: from raw features
        self.stage1 = SingleStageModel(num_layers, num_f_maps, feat_dim, num_classes)
        # Stages 2–N: refine from previous stage output + original features
        self.stages = nn.ModuleList([
            SingleStageModel(num_layers, num_f_maps, num_classes + feat_dim, num_classes)
            for _ in range(num_stages - 1)
        ])

    def forward(self, frames_seq):
        """
        frames_seq: [B, T, 3, H, W]
        Returns: list of [B, C, T] per-frame logits — one per stage.
        """
        B, T, C, H, W = frames_seq.shape
        x = frames_seq.reshape(B * T, C, H, W)
        feats = self.cnn(x).flatten(1).reshape(B, T, self.feat_dim)  # [B,T,D]
        feats_t = feats.permute(0, 2, 1)  # [B,D,T]

        out1 = self.stage1(feats_t)  # [B,C,T]
        outputs = [out1]
        prev = out1
        for stage in self.stages:
            inp = torch.cat([F.softmax(prev, dim=1), feats_t], dim=1)  # [B,C+D,T]
            out = stage(inp)
            outputs.append(out)
            prev = out
        return outputs  # list of [B,C,T]

# ── Loss ──────────────────────────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """ASL: Ridnik et al., ICCV 2021."""
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos
        self.clip = clip; self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_m = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss = (targets * ((1-p)**self.gamma_pos) * torch.log(p.clamp(min=self.eps)) +
                (1-targets) * (p_m**self.gamma_neg) * torch.log((1-p_m).clamp(min=self.eps)))
        return -loss.mean()


def smoothing_loss(logits_seq, targets_seq):
    """
    Truncated MSE smoothing loss — penalizes rapid frame-to-frame label changes.
    L_smooth = (1/T) * sum_t sum_c min(|ŷ_{t,c} - ŷ_{t-1,c}|^2, 4)
    Abu Farha & Gall, MS-TCN, CVPR 2019.
    logits_seq: [B, C, T]; targets_seq not used here (unsupervised smoothness).
    """
    prob = torch.sigmoid(logits_seq)  # [B,C,T]
    delta = prob[:, :, 1:] - prob[:, :, :-1]  # [B,C,T-1]
    return torch.clamp(delta ** 2, max=4.0).mean()

# ── Dataset: full-video per-frame labels ─────────────────────────────────────

class FrameSegDataset(Dataset):
    """
    Loads full videos and generates per-frame multi-label GT sequences.
    Each sample: (frame_sequence [T,3,H,W], label_sequence [T,C])
    Videos are chunked to max_frames to fit in GPU memory.
    """
    def __init__(self, video_list, video_root, fps=8.0, img_size=224,
                 max_frames=512, transform=None, frame_cache_dir=None,
                 min_chunk_frames=16):
        self.video_root = Path(video_root)
        self.fps = fps; self.img_size = img_size; self.max_frames = max_frames
        self.transform = transform or get_transform(img_size)
        self.frame_cache_dir = frame_cache_dir
        self.min_chunk_frames = min_chunk_frames
        self.samples = []  # (video_path, annotations, chunk_start_frame, chunk_end_frame, cache_path)
        self._build_index(video_list)

    def _build_index(self, video_list):
        n_cached = n_videos = 0
        for vid in video_list:
            vp = self.video_root / vid["video_path"]
            if not vp.exists(): continue
            n_videos += 1
            cache_path = None
            if self.frame_cache_dir:
                cp = Path(self.frame_cache_dir) / frame_cache_name(vid["video_path"])
                if cp.exists():
                    cache_path = str(cp)
                    n_cached += 1
            duration = float(vid.get("duration", 0))
            total_frames = int(duration * self.fps)
            anns = vid.get("annotations", [])
            for start_f in range(0, max(1, total_frames), self.max_frames):
                end_f = min(start_f + self.max_frames, total_frames)
                # InstanceNorm1d needs >1 temporal element: a tiny trailing
                # chunk (e.g. 1 frame when total % max_frames == 1) crashes the
                # model, so extend it backwards to overlap the previous chunk.
                if end_f - start_f < self.min_chunk_frames and start_f > 0:
                    start_f = max(0, end_f - self.min_chunk_frames)
                self.samples.append((str(vp), anns, start_f, end_f, cache_path))
        if self.frame_cache_dir:
            print(f"[dataset] frame cache: {n_cached}/{n_videos} videos "
                  f"(uncached videos decode the full video per chunk — slow)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vp, anns, start_f, end_f, cache_path = self.samples[idx]

        # Fast path: memory-mapped slice from the pre-extracted frame cache
        chunk = None
        if cache_path:
            try:
                arr = open_frame_cache(cache_path)
                chunk = np.asarray(arr[start_f:end_f])
            except Exception:
                chunk = None
        if chunk is None:
            try:
                frames_np, _ = extract_frames_at_fps(vp, self.fps, self.img_size)
                chunk = frames_np[start_f:end_f]
            except Exception:
                T = max(2, end_f - start_f)
                return (torch.zeros(T, self.img_size, self.img_size, 3, dtype=torch.uint8),
                        torch.zeros(T, NUM_CLASSES))

        T = len(chunk)
        if T < 2:  # InstanceNorm needs >1 temporal element
            pad = np.zeros((2 - T, self.img_size, self.img_size, 3), dtype=np.uint8)
            chunk = np.concatenate([chunk, pad], axis=0) if T else pad
            T = 2
        # uint8 [T,H,W,3] — normalized batched on GPU via normalize_frames_gpu()
        frames_t = torch.from_numpy(np.ascontiguousarray(chunk))

        # Per-frame multi-label GT
        labels = np.zeros((T, NUM_CLASSES), dtype=np.float32)
        fps = self.fps
        for ann in anns:
            gt_s_f = int(ann["start"] * fps) - start_f
            gt_e_f = int(ann["end"] * fps) - start_f
            bt = norm_bug_type(ann["bug_type"])
            if bt not in BUG2IDX: continue
            c = BUG2IDX[bt]
            f_start = max(0, gt_s_f)
            f_end = min(T, gt_e_f)
            if f_start < f_end:
                labels[f_start:f_end, c] = 1.0

        return frames_t, torch.from_numpy(labels)


def collate_variable_length(batch):
    """Pad uint8 [T,H,W,3] sequences in batch to the same length."""
    frames_list, labels_list = zip(*batch)
    max_T = max(f.shape[0] for f in frames_list)
    H, W = frames_list[0].shape[1], frames_list[0].shape[2]
    padded_frames = torch.zeros(len(batch), max_T, H, W, 3, dtype=torch.uint8)
    padded_labels = torch.zeros(len(batch), max_T, NUM_CLASSES)
    masks = torch.zeros(len(batch), max_T, dtype=torch.bool)
    for i, (f, l) in enumerate(zip(frames_list, labels_list)):
        T = f.shape[0]
        padded_frames[i, :T] = f
        padded_labels[i, :T] = l
        masks[i, :T] = True
    return padded_frames, padded_labels, masks

# ── Utility: frame predictions → temporal segments ────────────────────────────

def frames_to_segments(frame_probs, fps, min_conf=0.0, min_duration=0.0):
    """
    Convert per-frame probability array [T, C] to temporal segments.
    Uses run-length encoding on thresholded predictions.
    """
    T, C = frame_probs.shape
    segments = []
    seg_id = 0
    for c in range(C):
        class_name = CANON_BUG_TYPES[c]
        active = frame_probs[:, c] >= 0.5  # default threshold; override with tuned thr
        in_seg = False
        start_f = 0
        conf_acc = []
        for t in range(T):
            if active[t] and not in_seg:
                in_seg = True; start_f = t; conf_acc = [float(frame_probs[t, c])]
            elif active[t] and in_seg:
                conf_acc.append(float(frame_probs[t, c]))
            elif not active[t] and in_seg:
                mean_conf = float(np.mean(conf_acc))
                dur = (t - start_f) / fps
                if mean_conf >= min_conf and dur >= min_duration:
                    segments.append({
                        "bug_type": OUTPUT_NAMES.get(class_name, class_name),
                        "start": round(start_f / fps, 3),
                        "end": round(t / fps, 3),
                        "duration": round(dur, 3),
                        "confidence": round(mean_conf, 4),
                        "segment_id": seg_id,
                    })
                    seg_id += 1
                in_seg = False; conf_acc = []
        if in_seg:
            mean_conf = float(np.mean(conf_acc))
            dur = (T - start_f) / fps
            if mean_conf >= min_conf and dur >= min_duration:
                segments.append({
                    "bug_type": OUTPUT_NAMES.get(class_name, class_name),
                    "start": round(start_f / fps, 3),
                    "end": round(T / fps, 3),
                    "duration": round(dur, 3),
                    "confidence": round(mean_conf, 4),
                    "segment_id": seg_id,
                })
                seg_id += 1
    segments.sort(key=lambda s: (s["start"], s["bug_type"]))
    for i, s in enumerate(segments): s["segment_id"] = i
    return segments

# ── Temporal metrics (shared with Model 1) ────────────────────────────────────

def temporal_iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_temporal_metrics(pred_by_class, gt_by_class, tiou_thresholds=(0.3, 0.5, 0.7)):
    per_class = {}
    map_per_thr = {}
    all_ious = []
    total_gt = sum(len(v) for v in gt_by_class.values())

    for cls in CANON_BUG_TYPES:
        preds = sorted(pred_by_class.get(cls, []), key=lambda x: -x.get("confidence", 0.5))
        gts = gt_by_class.get(cls, [])
        cls_ap = {}
        for thr in tiou_thresholds:
            if not gts or not preds:
                cls_ap[f"AP_{thr}"] = 0.0; continue
            gt_matched = [False] * len(gts)
            tp_arr, fp_arr = [], []
            for pred in preds:
                best_iou, best_idx = 0.0, -1
                for j, gt in enumerate(gts):
                    iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                    if iou > best_iou: best_iou, best_idx = iou, j
                if best_iou >= thr and best_idx >= 0 and not gt_matched[best_idx]:
                    tp_arr.append(1); fp_arr.append(0); gt_matched[best_idx] = True
                else:
                    tp_arr.append(0); fp_arr.append(1)
            tp_c = np.cumsum(tp_arr); fp_c = np.cumsum(fp_arr)
            rec = np.concatenate([[0.0], tp_c / max(len(gts), 1)])
            prec = np.concatenate([[1.0], tp_c / (tp_c + fp_c + 1e-8)])
            cls_ap[f"AP_{thr}"] = round(float(np.sum((rec[1:] - rec[:-1]) * prec[1:])), 4)
        per_class[cls] = cls_ap
        for pred in preds:
            best = max((temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                        for gt in gts), default=0.0)
            all_ious.append(best)

    for thr_key, thr in [("mAP_0.3", 0.3), ("mAP_0.5", 0.5), ("mAP_0.7", 0.7)]:
        map_per_thr[thr_key] = round(float(np.mean([per_class[c][f"AP_{thr}"] for c in CANON_BUG_TYPES])), 4)

    segment_recall, temporal_precision, temporal_f1 = {}, {}, {}
    for thr in tiou_thresholds:
        matched_gt, matched_pred = 0, 0
        total_pred = sum(len(pred_by_class.get(c, [])) for c in CANON_BUG_TYPES)
        for cls in CANON_BUG_TYPES:
            preds = pred_by_class.get(cls, []); gts = gt_by_class.get(cls, [])
            gt_matched = [False] * len(gts)
            for pred in preds:
                for j, gt in enumerate(gts):
                    if not gt_matched[j] and temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"]) >= thr:
                        gt_matched[j] = True; matched_gt += 1; matched_pred += 1; break
        segment_recall[f"tIoU_{thr}"] = round(matched_gt / max(total_gt, 1), 4)
        temporal_precision[f"tIoU_{thr}"] = round(matched_pred / max(total_pred, 1), 4)
        p = temporal_precision[f"tIoU_{thr}"]; r = segment_recall[f"tIoU_{thr}"]
        temporal_f1[f"tIoU_{thr}"] = round(2 * p * r / (p + r + 1e-8), 4)

    total_pred_global = sum(len(pred_by_class.get(c, [])) for c in CANON_BUG_TYPES)
    return {**map_per_thr,
            "mean_tIoU": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
            "per_class": per_class, "segment_recall": segment_recall,
            "temporal_precision": temporal_precision, "temporal_f1": temporal_f1,
            "over_prediction_ratio": round(total_pred_global / max(total_gt, 1), 3)}


def compute_multilabel_metrics(y_pred_prob, y_true, thresholds=None):
    if thresholds is None: thresholds = np.full(NUM_CLASSES, 0.5)
    y_pred = (y_pred_prob >= thresholds).astype(int)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    clip_ap = [average_precision_score(y_true[:, c], y_pred_prob[:, c]) if y_true[:, c].sum() > 0 else 0.0
               for c in range(NUM_CLASSES)]
    roc_auc = []
    for c in range(NUM_CLASSES):
        try: roc_auc.append(roc_auc_score(y_true[:, c], y_pred_prob[:, c]) if y_true[:, c].sum() > 0 else 0.0)
        except: roc_auc.append(0.0)
    try: lrap = label_ranking_average_precision_score(y_true, y_pred_prob)
    except: lrap = 0.0
    metrics = {
        "micro_f1": round(float(micro_f1), 4), "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4), "micro_precision": round(float(micro_p), 4),
        "micro_recall": round(float(micro_r), 4), "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "subset_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "clip_mAP": round(float(np.mean(clip_ap)), 4), "mean_roc_auc": round(float(np.mean(roc_auc)), 4),
        "lrap": round(float(lrap), 4), "per_class": {
            CANON_BUG_TYPES[c]: {"precision": round(float(per_cls_p[c]), 4),
                                  "recall": round(float(per_cls_r[c]), 4),
                                  "f1": round(float(per_cls_f1[c]), 4),
                                  "clip_AP": round(float(clip_ap[c]), 4),
                                  "roc_auc": round(float(roc_auc[c]), 4)}
            for c in range(NUM_CLASSES)
        }
    }
    return metrics


def tune_thresholds(y_pred_prob, y_true, thr_range=np.arange(0.1, 0.91, 0.05)):
    best_thrs = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in thr_range:
            pred = (y_pred_prob[:, c] >= t).astype(int)
            tp = ((pred == 1) & (y_true[:, c] == 1)).sum()
            fp = ((pred == 1) & (y_true[:, c] == 0)).sum()
            fn = ((pred == 0) & (y_true[:, c] == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1: best_f1, best_t = f1, t
        best_thrs[c] = best_t
    return best_thrs

# ── Training utilities ────────────────────────────────────────────────────────

class MultiTaskEarlyStopping:
    def __init__(self, patience=15, lr_floor=1e-6):
        self.patience = patience; self.lr_floor = lr_floor
        self.best = -1.0; self.counter = 0

    def composite(self, f1, map_val): return 0.3 * f1 + 0.7 * map_val

    def step(self, f1, map_val, lr):
        score = self.composite(f1, map_val)
        if score > self.best + 1e-4:
            self.best = score; self.counter = 0
        else:
            self.counter += 1
        return (self.counter >= self.patience or lr <= self.lr_floor), score


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "metrics": metrics}, path)


def load_for_resume(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"): scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def manage_epoch_checkpoints(d: Path, keep=3):
    ckpts = sorted(d.glob("checkpoint_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in ckpts[:-keep]: old.unlink(missing_ok=True)

# ── Train / val one epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, asl_loss, device,
                    lambda_smooth, epoch, tb_writer=None, use_amp=False):
    model.train()
    total_loss = total_cls = total_smooth = 0.0
    for frames, labels, masks in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        frames = normalize_frames_gpu(frames, device)
        labels = labels.to(device, non_blocking=True)   # [B,T,C]
        masks = masks.to(device, non_blocking=True)     # [B,T]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(frames)  # list of [B,C,T]
        outputs = [o.float() for o in outputs]  # losses in fp32

        loss = torch.tensor(0.0, device=device)
        for out in outputs:
            out_t = out.permute(0, 2, 1)  # [B,T,C]
            valid_out = out_t[masks]; valid_lbl = labels[masks]
            l_cls = asl_loss(valid_out, valid_lbl)
            l_sm = smoothing_loss(out, labels.permute(0, 2, 1))
            loss = loss + l_cls + lambda_smooth * l_sm
            total_cls += l_cls.item(); total_smooth += l_sm.item()

        total_loss += loss.item()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    n = max(len(loader), 1)
    m = {"loss": total_loss / n, "cls": total_cls / n, "smooth": total_smooth / n}
    if tb_writer:
        for k, v in m.items(): tb_writer.add_scalar(f"train/{k}", v, epoch)
    return m


@torch.no_grad()
def validate(model, loader, asl_loss, device, lambda_smooth, epoch, tb_writer=None,
             use_amp=False):
    model.eval()
    total_loss = 0.0; all_prob = []; all_gt = []
    for frames, labels, masks in loader:
        frames = normalize_frames_gpu(frames, device)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(frames)
        outputs = [o.float() for o in outputs]
        out = outputs[-1]  # use last stage for metrics
        out_t = out.permute(0, 2, 1)
        valid_out = out_t[masks]; valid_lbl = labels[masks]
        l_cls = asl_loss(valid_out, valid_lbl)
        l_sm = smoothing_loss(out, labels.permute(0, 2, 1))
        total_loss += (l_cls + lambda_smooth * l_sm).item()
        all_prob.append(torch.sigmoid(valid_out).cpu().numpy())
        all_gt.append(valid_lbl.cpu().numpy())

    all_prob = np.concatenate(all_prob, 0); all_gt = np.concatenate(all_gt, 0)
    thrs = tune_thresholds(all_prob, all_gt)
    y_pred = (all_prob >= thrs).astype(int)
    tp = ((y_pred == 1) & (all_gt == 1)).sum()
    fp = ((y_pred == 1) & (all_gt == 0)).sum()
    fn = ((y_pred == 0) & (all_gt == 1)).sum()
    micro_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    clip_aps = [average_precision_score(all_gt[:, c], all_prob[:, c]) if all_gt[:, c].sum() > 0 else 0.0
                for c in range(NUM_CLASSES)]
    val_map_proxy = float(np.mean(clip_aps))
    avg_loss = total_loss / max(len(loader), 1)
    if tb_writer:
        tb_writer.add_scalar("val/loss", avg_loss, epoch)
        tb_writer.add_scalar("val/micro_f1", micro_f1, epoch)
        tb_writer.add_scalar("val/clip_mAP_proxy", val_map_proxy, epoch)
    return {"loss": avg_loss, "micro_f1": micro_f1, "clip_mAP_proxy": val_map_proxy,
            "thresholds": thrs}

# ── Test evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_test(model, args, device, thresholds, out_dir: Path, tb_writer=None, wandb_run=None):
    with open(args.temporal_dataset) as f: dataset = json.load(f)
    splits = dataset.get("metadata", {}).get("splits", dataset.get("splits", {}))
    test_ids = set(splits.get("test", []))
    test_videos = [v for v in dataset["videos"] if v["video_id"] in test_ids]
    print(f"[test] {len(test_videos)} test videos")

    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    use_amp = (dev_type == "cuda" and not getattr(args, "no_amp", False)
               and torch.cuda.is_bf16_supported())
    model.eval()
    pred_by_class = defaultdict(list); gt_by_class = defaultdict(list)
    all_prob = []; all_gt_cls = []; per_video_results = []

    for vid in tqdm(test_videos, desc="[test]"):
        vp = Path(args.video_root) / vid["video_path"]
        if not vp.exists(): continue
        anns = vid.get("annotations", [])
        try: frames_np, duration = load_video_frames_for_eval(str(vp), vid["video_path"], args)
        except Exception as e: print(f"[WARN] {vid['video_id']}: {e}"); continue

        T = len(frames_np)

        # Process in chunks (kept on CPU as uint8; each chunk normalized on GPU)
        chunk = 512
        frame_probs = np.zeros((T, NUM_CLASSES), dtype=np.float32)
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            if t1 - t0 < 2:  # InstanceNorm needs >1 temporal element
                if t1 < 2:
                    break
                t0 = t1 - 2  # extend tiny tail chunk backwards
            chunk_u8 = torch.from_numpy(np.ascontiguousarray(frames_np[t0:t1])).unsqueeze(0)
            chunk_t = normalize_frames_gpu(chunk_u8, device)  # [1,t,3,H,W]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(chunk_t)
            probs = torch.sigmoid(outputs[-1].float()).squeeze(0).permute(1, 0).cpu().numpy()  # [t,C]
            frame_probs[t0:t1] = probs

        # Per-frame GT for clip metrics
        for t in range(T):
            t_s = t / args.fps; t_e = (t + 1) / args.fps
            gt_types = np.zeros(NUM_CLASSES, dtype=int)
            for ann in anns:
                if max(0.0, min(t_e, ann["end"]) - max(t_s, ann["start"])) > 0:
                    bt = norm_bug_type(ann["bug_type"])
                    if bt in BUG2IDX: gt_types[BUG2IDX[bt]] = 1
            all_prob.append(frame_probs[t])
            all_gt_cls.append(gt_types)

        # Apply tuned thresholds to frame_probs
        adjusted_probs = frame_probs.copy()
        for c in range(NUM_CLASSES):
            adjusted_probs[:, c] = (frame_probs[:, c] >= thresholds[c]).astype(float)

        segments = frames_to_segments(frame_probs, args.fps,
                                       min_conf=getattr(args, "min_conf", 0.0),
                                       min_duration=getattr(args, "min_duration", 0.0))
        for seg in segments:
            c_key = norm_bug_type(seg["bug_type"])
            pred_by_class[c_key].append({"start": seg["start"], "end": seg["end"],
                                          "confidence": seg.get("confidence", 0.5)})
        for ann in anns:
            c_key = norm_bug_type(ann["bug_type"])
            gt_by_class[c_key].append({"start": ann["start"], "end": ann["end"]})

        pv_pred = (frame_probs >= thresholds[:NUM_CLASSES]).astype(int)
        pv_gt = np.stack(all_gt_cls[-T:])
        tp = ((pv_pred == 1) & (pv_gt == 1)).sum()
        fp = ((pv_pred == 1) & (pv_gt == 0)).sum()
        fn = ((pv_pred == 0) & (pv_gt == 1)).sum()
        per_video_results.append({
            "video_id": vid["video_id"], "duration": round(float(vid.get("duration", 0)), 3),
            "num_frames": T, "num_pred_segments": len(segments),
            "num_gt_annotations": len(anns),
            "clip_f1_types": round(float(2 * tp / (2 * tp + fp + fn + 1e-8)), 4),
        })

    all_prob_np = np.stack(all_prob) if all_prob else np.zeros((1, NUM_CLASSES))
    all_gt_np = np.stack(all_gt_cls) if all_gt_cls else np.zeros((1, NUM_CLASSES), dtype=int)
    clip_metrics = compute_multilabel_metrics(all_prob_np, all_gt_np, thresholds[:NUM_CLASSES])
    temporal_metrics = compute_temporal_metrics(dict(pred_by_class), dict(gt_by_class))

    test_metrics = {"clip_level": clip_metrics, "temporal": temporal_metrics,
                    "per_video": per_video_results}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    flat = {"test/clip_micro_f1": clip_metrics["micro_f1"],
            "test/mAP_0.3": temporal_metrics["mAP_0.3"],
            "test/mAP_0.5": temporal_metrics["mAP_0.5"],
            "test/mAP_0.7": temporal_metrics["mAP_0.7"],
            "test/mean_tIoU": temporal_metrics["mean_tIoU"],
            "test/over_prediction_ratio": temporal_metrics["over_prediction_ratio"]}
    if tb_writer:
        for k, v in flat.items(): tb_writer.add_scalar(k, v, 0)
    if wandb_run: wandb_run.log(flat)

    print("\n" + "=" * 60 + "\nTEST RESULTS\n" + "=" * 60)
    for k, v in flat.items(): print(f"  {k.split('/')[-1]:30s}: {v:.4f}")
    print("=" * 60)
    return test_metrics

# ── Main train ────────────────────────────────────────────────────────────────

def run_train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = (device.type == "cuda" and not args.no_amp
               and torch.cuda.is_bf16_supported())
    print(f"[train] Mixed precision (bf16): {'enabled' if use_amp else 'disabled'}")
    checkpoint_dir = Path(args.save_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logdir); (log_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tensorboard"))
    hpo_log = log_dir / "hpo_progress.log"

    use_wandb = not getattr(args, "no_wandb", False) and WANDB_AVAILABLE
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(project=getattr(args, "wandb_project", "localization-bugs"),
                               name=f"model2_mstcn_{time.strftime('%m%d_%H%M')}",
                               config=vars(args), tags=["model2", "mstcn", "localization"])

    with open(args.temporal_dataset) as f: dataset = json.load(f)
    splits = dataset.get("metadata", {}).get("splits", dataset.get("splits", {}))
    all_videos = {v["video_id"]: v for v in dataset["videos"]}
    train_vids = [all_videos[i] for i in splits.get("train", []) if i in all_videos]
    val_vids = [all_videos[i] for i in splits.get("val", []) if i in all_videos]
    print(f"[train] Train: {len(train_vids)} | Val: {len(val_vids)}")

    # Build/verify the frame cache BEFORE training (no-op once complete)
    if args.frame_cache_dir:
        test_vids_cache = [all_videos[i] for i in splits.get("test", []) if i in all_videos]
        ensure_frame_cache(train_vids + val_vids + test_vids_cache, args.video_root,
                           args.frame_cache_dir, fps=args.fps, img_size=args.img_size,
                           workers=max(args.num_workers, 4))
    else:
        print("[train] WARNING: --frame-cache-dir not set — every chunk sample will "
              "decode its full source video (extremely slow). Strongly consider setting it.")

    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate_variable_length,
                         pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    hparams = {"lr": args.lr, "lambda_smooth": args.lambda_smooth,
               "dropout": 0.0, "weight_decay": args.weight_decay,
               "batch_size": args.batch_size, "num_f_maps": args.num_f_maps}

    # HPO
    if not args.pilot and args.hpo_trials > 0 and OPTUNA_AVAILABLE:
        print(f"\n[HPO] {args.hpo_trials} Optuna TPE trials...")
        study_path = str(log_dir / "optuna_study.db")
        study = optuna.create_study(study_name="model2_mstcn", direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
                                    storage=f"sqlite:///{study_path}", load_if_exists=True)
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = args.hpo_trials - completed

        def hpo_obj(trial):
            hp = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "lambda_smooth": trial.suggest_float("lambda_smooth", 0.05, 0.5),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
                "num_f_maps": trial.suggest_categorical("num_f_maps", [32, 64]),
            }
            m = MSTCNLocalization(num_f_maps=int(hp["num_f_maps"])).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            asl = AsymmetricLoss()
            ds_t = FrameSegDataset(train_vids[:30], args.video_root, fps=args.fps,
                                   img_size=args.img_size, frame_cache_dir=args.frame_cache_dir)
            ds_v = FrameSegDataset(val_vids[:10], args.video_root, fps=args.fps,
                                   img_size=args.img_size, frame_cache_dir=args.frame_cache_dir)
            tl = DataLoader(ds_t, batch_size=int(hp["batch_size"]), shuffle=True,
                            drop_last=True, **loader_kwargs)
            vl = DataLoader(ds_v, batch_size=int(hp["batch_size"]), shuffle=False, **loader_kwargs)
            es = MultiTaskEarlyStopping(patience=4)
            for ep in range(1, 9):
                train_one_epoch(m, tl, opt, asl, device, hp["lambda_smooth"], ep, use_amp=use_amp)
                vm = validate(m, vl, asl, device, hp["lambda_smooth"], ep, use_amp=use_amp)
                comp = 0.3 * vm["micro_f1"] + 0.7 * vm["clip_mAP_proxy"]
                trial.report(comp, ep)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                stop, _ = es.step(vm["micro_f1"], vm["clip_mAP_proxy"], opt.param_groups[0]["lr"])
                if stop: break
            vm = validate(m, vl, asl, device, hp["lambda_smooth"], 0, use_amp=use_amp)
            return 0.3 * vm["micro_f1"] + 0.7 * vm["clip_mAP_proxy"]

        for _ in range(remaining):
            try:
                study.optimize(hpo_obj, n_trials=1, gc_after_trial=True)
                msg = f"Trial {len(study.trials)}/{args.hpo_trials} done, best: {study.best_value:.4f}\n"
                print(f"[HPO] {msg.strip()}")
                with open(hpo_log, "a") as f: f.write(msg)
            except Exception as e:
                print(f"[HPO] Trial failed: {e}")

        hparams.update(study.best_params)
        with open(log_dir / "best_hparams.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

    # Build full dataset / model
    train_ds = FrameSegDataset(train_vids, args.video_root, fps=args.fps, img_size=args.img_size,
                               frame_cache_dir=args.frame_cache_dir)
    val_ds = FrameSegDataset(val_vids, args.video_root, fps=args.fps, img_size=args.img_size,
                             frame_cache_dir=args.frame_cache_dir)
    train_loader = DataLoader(train_ds, batch_size=int(hparams["batch_size"]), shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(hparams["batch_size"]), shuffle=False,
                            **loader_kwargs)

    model = MSTCNLocalization(num_f_maps=int(hparams.get("num_f_maps", 64))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"],
                                   weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6)
    asl_loss = AsymmetricLoss(); es = MultiTaskEarlyStopping(patience=15)

    start_epoch = 1
    best_ckpts = {k: {"score": -1.0, "path": checkpoint_dir / f"{k}.pt"}
                  for k in ("best_composite", "best_f1", "best_mAP")}
    if args.checkpoint and Path(args.checkpoint).exists():
        start_epoch, _ = load_for_resume(args.checkpoint, model, optimizer, scheduler, str(device))
        start_epoch += 1

    if TORCHINFO_AVAILABLE:
        dummy = torch.zeros(1, 16, 3, args.img_size, args.img_size).to(device)
        with open(log_dir / "architecture_summary.txt", "w") as f:
            f.write(str(torchinfo.summary(model, input_data=dummy, verbose=0)))

    best_thresholds = np.full(NUM_CLASSES, 0.5)
    max_epochs = 5 if args.pilot else args.epochs

    for epoch in range(start_epoch, max_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, asl_loss, device,
                                   hparams.get("lambda_smooth", 0.15), epoch, tb,
                                   use_amp=use_amp)
        val_m = validate(model, val_loader, asl_loss, device,
                         hparams.get("lambda_smooth", 0.15), epoch, tb, use_amp=use_amp)
        composite = 0.3 * val_m["micro_f1"] + 0.7 * val_m["clip_mAP_proxy"]
        scheduler.step(composite)
        lr = optimizer.param_groups[0]["lr"]
        tb.add_scalar("train/lr", lr, epoch)
        print(f"Epoch {epoch:3d} | loss={train_m['loss']:.4f} | F1={val_m['micro_f1']:.4f} | "
              f"cMAP={val_m['clip_mAP_proxy']:.4f} | comp={composite:.4f} | lr={lr:.2e}")
        if wandb_run:
            wandb_run.log({"epoch": epoch, "train/loss": train_m["loss"],
                           "val/micro_f1": val_m["micro_f1"],
                           "val/clip_mAP_proxy": val_m["clip_mAP_proxy"],
                           "val/composite": composite, "train/lr": lr})

        for key, score in [("best_composite", composite), ("best_f1", val_m["micro_f1"]),
                            ("best_mAP", val_m["clip_mAP_proxy"])]:
            if score > best_ckpts[key]["score"]:
                best_ckpts[key]["score"] = score
                save_checkpoint(model, optimizer, scheduler, epoch,
                                {"composite": composite}, best_ckpts[key]["path"])
                if key == "best_composite":
                    best_thresholds = val_m["thresholds"]

        epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, {"composite": composite}, epoch_ckpt)
        manage_epoch_checkpoints(checkpoint_dir, keep=3)

        if args.pilot and epoch >= 5:
            print(f"[pilot] Done. composite={composite:.4f}")
            tb.close()
            if wandb_run: wandb_run.finish()
            return

        stop, _ = es.step(val_m["micro_f1"], val_m["clip_mAP_proxy"], lr)
        if stop:
            print(f"[train] Early stopping at epoch {epoch}")
            break

    tb.close()
    if not args.pilot:
        load_for_resume(str(best_ckpts["best_composite"]["path"]), model, device=str(device))
        run_test(model, args, device, best_thresholds, Path(args.logdir), wandb_run=wandb_run)
    if wandb_run: wandb_run.finish()

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MSTCNLocalization().to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt)); model.eval()

    use_amp = (device.type == "cuda" and not getattr(args, "no_amp", False)
               and torch.cuda.is_bf16_supported())
    frames_np, duration = extract_frames_at_fps(args.video_path, args.fps, args.img_size)
    T = len(frames_np)

    frame_probs = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    chunk = 512
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        if t1 - t0 < 2:  # InstanceNorm needs >1 temporal element
            if t1 < 2:
                break
            t0 = t1 - 2  # extend tiny tail chunk backwards
        chunk_u8 = torch.from_numpy(np.ascontiguousarray(frames_np[t0:t1])).unsqueeze(0)
        chunk_t = normalize_frames_gpu(chunk_u8, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(chunk_t)
        frame_probs[t0:t1] = torch.sigmoid(outputs[-1].float()).squeeze(0).permute(1, 0).cpu().numpy()

    segments = frames_to_segments(frame_probs, args.fps,
                                   min_conf=getattr(args, "min_conf", 0.0),
                                   min_duration=getattr(args, "min_duration", 0.0))
    result = {"video_id": Path(args.video_path).stem, "video_path": args.video_path,
              "duration": round(duration, 3),
              "annotations": [{"bug_type": s["bug_type"], "start": s["start"],
                                "end": s["end"], "duration": s["duration"],
                                "segment_id": s["segment_id"]} for s in segments]}
    out_path = args.inference_out or "result.json"
    with open(out_path, "w") as f: json.dump(result, f, indent=2)
    print(f"[infer] {len(segments)} segments → {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Model 2: MS-TCN Temporal Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--temporal-dataset", type=str, default="../localization/temporal_bug_dataset.json")
    ap.add_argument("--video-path", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default="checkpoints/")
    ap.add_argument("--logdir", type=str, default="logs/")
    ap.add_argument("--inference-out", type=str, default="result.json")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-smooth", type=float, default=0.15)
    ap.add_argument("--num-f-maps", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--frame-cache-dir", type=str, default=None,
                    help="Dir for pre-extracted uint8 frame .npy files. Missing videos "
                         "are extracted automatically before training (one-time), then "
                         "chunks are read as memmap slices instead of decoding video.")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable bf16 mixed precision (enabled by default on CUDA)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--hpo-trials", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--wandb-project", type=str, default="localization-bugs")
    ap.add_argument("--no-wandb", action="store_true")
    # window_sec / stride_sec kept for API consistency with other models
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "train": run_train(args)
    elif args.mode == "infer":
        if not args.video_path: raise ValueError("--video-path required")
        run_infer(args)


if __name__ == "__main__":
    main()
