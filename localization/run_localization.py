#!/usr/bin/env python3
"""
run_localization.py — Multi-label bug detection + temporal localization

Uses the pretrained BugBiLSTM checkpoint (ResNet18 + BiLSTM + TemporalAttention)
to detect and localize visual bugs in full-length gameplay videos via a sliding
window approach. No retraining needed — the model was trained on 2-second clips
from the same 263 videos at 224x224 @ 8fps.

Modes:
  eval   — Run sliding-window inference over all test-split videos in
            temporal_bug_dataset.json. Computes clip-level F1 + temporal
            localization metrics (mAP @ tIoU 0.3/0.5/0.7, segment recall,
            per-class AP). Saves localization_test_metrics.json.

  infer  — Run sliding-window inference on a single uploaded video and write
            result.json in the format expected by the backend API.

Usage:
  # Evaluation on full test set
  python run_localization.py \\
      --mode eval \\
      --checkpoint /path/to/best_checkpoint.pt \\
      --temporal-dataset ./temporal_bug_dataset.json \\
      --video-root /path/to/videos \\
      --out-dir ./localization_outputs

  # Single-video inference (backend)
  python run_localization.py \\
      --mode infer \\
      --checkpoint /path/to/best_checkpoint.pt \\
      --video-path /path/to/upload.mp4 \\
      --inference-out /path/to/result.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# ── Video backend availability (mirrors train_FINAL_ALL_METRICS.py) ───────────
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

# torchvision.io is intentionally not used for full-video reading: read_video()
# loads the entire file into RAM, which is impractical for 20-minute videos.

# ── Optional experiment tracking ──────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── Canonical bug types (must match training script) ──────────────────────────
CANON_BUG_TYPES = [
    "z-clipping",
    "corrupted_texture",
    "geometry_corruption",
    "z-fighting",
    "boundary_hole",
]
BUG2IDX = {b: i for i, b in enumerate(CANON_BUG_TYPES)}

# temporal_bug_dataset.json uses underscores; training used hyphens for two types
_ALIAS = {
    "z_clipping": "z-clipping",
    "z-clipping": "z-clipping",
    "z_fighting": "z-fighting",
    "z-fighting": "z-fighting",
    "boundary_hole": "boundary_hole",
    "corrupted_texture": "corrupted_texture",
    "geometry_corruption": "geometry_corruption",
}


def norm_bug_type(s: str) -> str:
    s2 = s.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIAS.get(s2, _ALIAS.get(s.strip().lower(), s.strip().lower()))


def canon_to_output(s: str) -> str:
    """Return underscore form suitable for backend output / temporal_bug_dataset keys."""
    return s.replace("-", "_")


# ── Model (identical to train_FINAL_ALL_METRICS.py) ──────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.W(x))
        a = torch.softmax(self.v(h).squeeze(-1), dim=1)
        z = (x * a.unsqueeze(-1)).sum(dim=1)
        return z, a


class BugBiLSTM(nn.Module):
    """ResNet18 + BiLSTM + TemporalAttention with three prediction heads."""

    def __init__(self, num_bug_types: int = len(CANON_BUG_TYPES), pretrained: bool = False):
        super().__init__()
        enc = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(enc.children())[:-1])
        self.feat_dim = 512
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )
        self.attn = TemporalAttention(d_model=512)
        self.head_drop = nn.Dropout(p=0.3)
        self.fc_types = nn.Linear(512, num_bug_types)
        self.fc_count = nn.Linear(512, 4)
        self.fc_any = nn.Linear(512, 1)

    def forward(self, frames: torch.Tensor):
        """frames: [B, T, 3, H, W]"""
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        f = self.cnn(x).flatten(1).reshape(B, T, self.feat_dim).contiguous()
        y, _ = self.lstm(f)
        z, att = self.attn(y)
        z = self.head_drop(z)
        return self.fc_types(z), self.fc_count(z), self.fc_any(z).squeeze(-1), att


# ── Preprocessing transform (inference = center crop + normalize) ─────────────

def get_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[BugBiLSTM, float, Dict[str, float]]:
    """
    Load BugBiLSTM checkpoint saved by train_FINAL_ALL_METRICS.py.
    Returns (model, thr_any, thr_map).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = BugBiLSTM(num_bug_types=len(CANON_BUG_TYPES), pretrained=False)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # checkpoint IS the state dict

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    thr_any = float(ckpt.get("thr_any", 0.5))
    thr_map = ckpt.get("thr_map", {name: 0.5 for name in CANON_BUG_TYPES})
    # Ensure all canonical types have a threshold
    for name in CANON_BUG_TYPES:
        thr_map.setdefault(name, 0.5)

    return model, thr_any, thr_map


# ── Video frame extraction ────────────────────────────────────────────────────

def _extract_decord(
    video_path: str,
    target_fps: float,
    img_size: int,
    progress_callback,
) -> Tuple[np.ndarray, float]:
    """Decord backend — fastest on HPC; uses chunked random access."""
    vr = decord.VideoReader(str(video_path), width=img_size, height=img_size)
    native_fps = vr.get_avg_fps() or 30.0
    total_native = len(vr)
    duration = total_native / native_fps

    # Build array of target frame indices (uniform subsampling)
    indices = np.arange(0, total_native, native_fps / target_fps).astype(np.int64)
    indices = np.clip(indices, 0, total_native - 1)

    # Read in chunks to keep peak memory manageable
    CHUNK = 500
    out_frames: List[np.ndarray] = []
    for chunk_start in range(0, len(indices), CHUNK):
        chunk_idx = indices[chunk_start: chunk_start + CHUNK]
        batch = vr.get_batch(chunk_idx)  # torch tensor [T, H, W, C] uint8
        out_frames.append(batch.numpy())
        if progress_callback:
            progress_callback(int(chunk_idx[-1]), total_native)

    return np.concatenate(out_frames, axis=0), duration


def _extract_opencv(
    video_path: str,
    target_fps: float,
    img_size: int,
    progress_callback,
) -> Tuple[np.ndarray, float]:
    """OpenCV backend — sequential read with grab() skipping for efficiency."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_native / native_fps if total_native > 0 else 0.0

    sample_interval = native_fps / target_fps  # native frames per target frame
    frames: List[np.ndarray] = []
    frame_idx = 0
    next_sample = 0.0

    while True:
        if frame_idx >= int(next_sample):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
            frames.append(frame_resized)
            next_sample += sample_interval
            if progress_callback and len(frames) % 100 == 0:
                progress_callback(frame_idx, total_native)
        else:
            if not cap.grab():
                break
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    if duration <= 0:
        duration = len(frames) / target_fps

    return np.stack(frames, axis=0), duration


def extract_frames_at_fps(
    video_path: str,
    target_fps: float = 8.0,
    img_size: int = 224,
    progress_callback=None,
) -> Tuple[np.ndarray, float]:
    """
    Extract frames from a full-length video at target_fps, resized to img_size².

    Backend priority: decord (fastest, HPC-friendly) → opencv (fallback).
    torchvision.io is intentionally skipped — read_video() loads the whole file
    into RAM, which is impractical for 20-minute videos.

    Returns:
        frames_uint8 : np.ndarray  [N, img_size, img_size, 3]  uint8
        duration_sec : float
    """
    errors: List[str] = []

    if _BACKENDS.get("decord"):
        try:
            return _extract_decord(video_path, target_fps, img_size, progress_callback)
        except Exception as exc:
            errors.append(f"decord: {exc}")
            print(f"[WARN] decord failed, falling back to opencv — {exc}")

    if _BACKENDS.get("opencv"):
        try:
            return _extract_opencv(video_path, target_fps, img_size, progress_callback)
        except Exception as exc:
            errors.append(f"opencv: {exc}")

    raise RuntimeError(
        f"All video backends failed for {video_path}.\n"
        + "\n".join(errors)
        + "\nInstall at least one backend: pip install decord  or  pip install opencv-python-headless"
    )


# ── Sliding window inference ──────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(
    frames_np: np.ndarray,
    model: BugBiLSTM,
    transform: transforms.Compose,
    thr_any: float,
    thr_map: Dict[str, float],
    target_fps: float = 8.0,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> List[Dict]:
    """
    Slide a window over pre-extracted frames and run BugBiLSTM on each window.

    Returns list of per-window predictions:
    [
        {
            "t_start": float,
            "t_end": float,
            "pred_types": [0|1, ...] (len = num_classes),
            "prob_any": float,
            "prob_types": [float, ...],
        },
        ...
    ]
    """
    model.eval()
    N = frames_np.shape[0]
    window_frames = max(1, int(round(window_sec * target_fps)))   # 16
    stride_frames = max(1, int(round(stride_sec * target_fps)))   # 8

    # Pad with last frame if video is shorter than one window
    if N < window_frames:
        pad_n = window_frames - N
        pad = np.repeat(frames_np[-1:], pad_n, axis=0)
        frames_np = np.concatenate([frames_np, pad], axis=0)
        N = frames_np.shape[0]

    window_starts = list(range(0, N - window_frames + 1, stride_frames))
    if not window_starts:
        window_starts = [0]

    thr_vec = np.array(
        [thr_map.get(name, 0.5) for name in CANON_BUG_TYPES], dtype=np.float32
    )

    predictions: List[Dict] = []

    for batch_start in range(0, len(window_starts), batch_size):
        batch_ws = window_starts[batch_start: batch_start + batch_size]

        batch_tensors = []
        for ws in batch_ws:
            window = frames_np[ws: ws + window_frames]  # [T, H, W, 3] uint8
            imgs = [transform(window[t]) for t in range(window_frames)]
            batch_tensors.append(torch.stack(imgs, dim=0))  # [T, 3, H, W]

        x = torch.stack(batch_tensors, dim=0).to(device)  # [B, T, 3, H, W]
        logits_types, _, logit_any, _ = model(x)

        prob_types = torch.sigmoid(logits_types).cpu().numpy()   # [B, C]
        prob_any = torch.sigmoid(logit_any).cpu().numpy()        # [B]

        for i, ws in enumerate(batch_ws):
            t_start = ws / target_fps
            t_end = min((ws + window_frames) / target_fps, frames_np.shape[0] / target_fps)

            if prob_any[i] >= thr_any:
                pred_types = (prob_types[i] >= thr_vec).astype(int).tolist()
            else:
                pred_types = [0] * len(CANON_BUG_TYPES)

            predictions.append({
                "t_start": t_start,
                "t_end": t_end,
                "pred_types": pred_types,
                "prob_any": float(prob_any[i]),
                "prob_types": prob_types[i].tolist(),
            })

    return predictions


# ── Segment merging ───────────────────────────────────────────────────────────

def merge_window_predictions(
    window_preds: List[Dict],
    class_names: List[str] = CANON_BUG_TYPES,
    min_conf: float = 0.0,
    min_duration: float = 0.0,
    max_gap_sec: float = float("inf"),
) -> List[Dict]:
    """
    For each bug class, merge overlapping / adjacent windows with positive
    predictions into contiguous temporal segments.

    Args:
        min_conf: Discard merged segments whose mean window confidence is below
            this value. Reduces over-prediction (default 0.0 = keep all).
        min_duration: Discard merged segments shorter than this many seconds
            (default 0.0 = keep all).
        max_gap_sec: Only merge consecutive positive windows when the gap
            between them is <= this value in seconds. Prevents merging across
            long clean sections (default inf = always merge if overlapping or
            adjacent, matching legacy behaviour).

    Output format (one entry per bug_type per segment) is compatible with the
    backend result.json schema — the frontend groups same-(start,end) pairs
    into multi-label display segments.

    Returns list of dicts:
        {"bug_type": str, "start": float, "end": float,
         "duration": float, "confidence": float, "segment_id": int}
    """
    segments: List[Dict] = []

    for c_idx, class_name in enumerate(class_names):
        # Collect windows where this class is active
        active: List[Tuple[float, float, float]] = []  # (t_start, t_end, conf)
        for wp in window_preds:
            if wp["pred_types"][c_idx] == 1:
                active.append((wp["t_start"], wp["t_end"], wp["prob_types"][c_idx]))

        if not active:
            continue

        # Merge overlapping / touching intervals that are within max_gap_sec
        active.sort(key=lambda x: x[0])
        cur_start, cur_end = active[0][0], active[0][1]
        conf_acc = [active[0][2]]

        for t_s, t_e, conf in active[1:]:
            gap = t_s - cur_end
            if gap <= max_gap_sec + 1e-6:  # overlapping, adjacent, or within allowed gap
                cur_end = max(cur_end, t_e)
                conf_acc.append(conf)
            else:
                mean_conf = float(np.mean(conf_acc))
                dur = cur_end - cur_start
                if mean_conf >= min_conf and dur >= min_duration:
                    segments.append({
                        "bug_type": canon_to_output(class_name),
                        "start": round(cur_start, 3),
                        "end": round(cur_end, 3),
                        "duration": round(dur, 3),
                        "confidence": round(mean_conf, 4),
                    })
                cur_start, cur_end = t_s, t_e
                conf_acc = [conf]

        mean_conf = float(np.mean(conf_acc))
        dur = cur_end - cur_start
        if mean_conf >= min_conf and dur >= min_duration:
            segments.append({
                "bug_type": canon_to_output(class_name),
                "start": round(cur_start, 3),
                "end": round(cur_end, 3),
                "duration": round(dur, 3),
                "confidence": round(mean_conf, 4),
            })

    # Sort by start time; assign sequential segment IDs
    segments.sort(key=lambda s: (s["start"], s["bug_type"]))
    for i, seg in enumerate(segments):
        seg["segment_id"] = i

    return segments


# ── Temporal localization metrics ─────────────────────────────────────────────

def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_ap_at_tiou(
    pred_segments: List[Dict],   # sorted by confidence descending
    gt_segments: List[Dict],
    tiou_thr: float,
) -> float:
    """Average Precision at one tIoU threshold for a single class."""
    if not gt_segments or not pred_segments:
        return 0.0

    num_gt = len(gt_segments)
    gt_matched = [False] * num_gt
    tp_arr, fp_arr = [], []

    for pred in pred_segments:
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gt_segments):
            iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou, best_idx = iou, j

        if best_iou >= tiou_thr and best_idx >= 0 and not gt_matched[best_idx]:
            tp_arr.append(1)
            fp_arr.append(0)
            gt_matched[best_idx] = True
        else:
            tp_arr.append(0)
            fp_arr.append(1)

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    rec = np.concatenate([[0.0], tp_cum / num_gt])
    prec = np.concatenate([[1.0], tp_cum / (tp_cum + fp_cum)])

    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    return round(ap, 4)


def compute_temporal_metrics(
    pred_by_class: Dict[str, List[Dict]],
    gt_by_class: Dict[str, List[Dict]],
    tiou_thresholds: List[float] = (0.3, 0.5, 0.7),
) -> Dict:
    """
    Compute mAP, per-class AP, segment recall, and mean tIoU.

    pred_by_class / gt_by_class: canonical class name → list of segment dicts
    """
    per_class: Dict[str, Dict] = {}

    for class_name in CANON_BUG_TYPES:
        out_name = canon_to_output(class_name)
        preds = sorted(
            pred_by_class.get(class_name, []),
            key=lambda x: x.get("confidence", 0.5),
            reverse=True,
        )
        gts = gt_by_class.get(class_name, [])
        per_class[out_name] = {
            f"AP_{thr}": compute_ap_at_tiou(preds, gts, thr)
            for thr in tiou_thresholds
        }

    map_per_thr = {
        f"mAP_{thr}": round(
            float(np.mean([per_class[canon_to_output(c)][f"AP_{thr}"] for c in CANON_BUG_TYPES])), 4
        )
        for thr in tiou_thresholds
    }

    # Segment recall — fraction of GT segments matched at each tIoU
    segment_recall: Dict[str, float] = {}
    total_gt = sum(len(v) for v in gt_by_class.values())
    for thr in tiou_thresholds:
        matched = 0
        for class_name in CANON_BUG_TYPES:
            preds = sorted(pred_by_class.get(class_name, []), key=lambda x: x.get("confidence", 0.5), reverse=True)
            gts = gt_by_class.get(class_name, [])
            gt_matched = [False] * len(gts)
            for pred in preds:
                for j, gt in enumerate(gts):
                    if not gt_matched[j] and temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"]) >= thr:
                        gt_matched[j] = True
                        matched += 1
                        break
        segment_recall[f"tIoU_{thr}"] = round(matched / total_gt, 4) if total_gt > 0 else 0.0

    # Mean tIoU across best-matched (pred, GT) pairs
    all_ious: List[float] = []
    for class_name in CANON_BUG_TYPES:
        preds = sorted(pred_by_class.get(class_name, []), key=lambda x: x.get("confidence", 0.5), reverse=True)
        gts = gt_by_class.get(class_name, [])
        gt_matched = [False] * len(gts)
        for pred in preds:
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(gts):
                if not gt_matched[j]:
                    iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
            if best_idx >= 0:
                all_ious.append(best_iou)
                gt_matched[best_idx] = True

    mean_tiou = round(float(np.mean(all_ious)), 4) if all_ious else 0.0

    return {**map_per_thr, "mean_tIoU": mean_tiou, "per_class": per_class, "segment_recall": segment_recall}


# ── Clip-level metrics (matches outer-folder evaluation) ─────────────────────

def compute_clip_level_metrics(
    window_preds: List[Dict],
    gt_annotations: List[Dict],
) -> Dict:
    """
    Label each sliding window with GT from temporal_bug_dataset.json annotations
    (any overlap counts), then compute micro-F1 for types and presence F1.
    """
    all_pred_t, all_gt_t = [], []
    all_pred_a, all_gt_a = [], []

    for wp in window_preds:
        t_s, t_e = wp["t_start"], wp["t_end"]
        gt_types = np.zeros(len(CANON_BUG_TYPES), dtype=int)
        for ann in gt_annotations:
            inter = max(0.0, min(t_e, ann["end"]) - max(t_s, ann["start"]))
            if inter > 0.0:
                bt = norm_bug_type(ann["bug_type"])
                if bt in BUG2IDX:
                    gt_types[BUG2IDX[bt]] = 1

        pred_types = np.array(wp["pred_types"], dtype=int)
        all_pred_t.append(pred_types)
        all_gt_t.append(gt_types)
        all_pred_a.append(int(pred_types.any()))
        all_gt_a.append(int(gt_types.any()))

    pred_t = np.stack(all_pred_t)
    gt_t = np.stack(all_gt_t)
    pred_a = np.array(all_pred_a)
    gt_a = np.array(all_gt_a)

    tp = ((pred_t == 1) & (gt_t == 1)).sum()
    fp = ((pred_t == 1) & (gt_t == 0)).sum()
    fn = ((pred_t == 0) & (gt_t == 1)).sum()
    f1_types = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    tp_a = ((pred_a == 1) & (gt_a == 1)).sum()
    fp_a = ((pred_a == 1) & (gt_a == 0)).sum()
    fn_a = ((pred_a == 0) & (gt_a == 1)).sum()
    f1_any = float(2 * tp_a / (2 * tp_a + fp_a + fn_a)) if (2 * tp_a + fp_a + fn_a) > 0 else 0.0

    return {
        "f1_types": round(f1_types, 4),
        "f1_any": round(f1_any, 4),
        "num_windows": len(window_preds),
    }


# ── Eval mode ─────────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if (args.device == "cpu" or not torch.cuda.is_available()) else args.device
    )
    print(f"[eval] Device: {device}")

    model, thr_any, thr_map = load_checkpoint(args.checkpoint, device)
    if args.thr_any is not None:
        thr_any = args.thr_any
        print(f"[eval] Overriding thr_any → {thr_any}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] thr_any={thr_any:.3f}")
    print(f"[eval] thr_map={thr_map}")

    transform = get_transform(args.img_size)

    with open(args.temporal_dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    videos = dataset["videos"]
    splits_info = dataset.get("splits", {})
    test_ids = set(splits_info.get("test", []))

    if test_ids:
        test_videos = [v for v in videos if v["video_id"] in test_ids]
    else:
        test_videos = [v for v in videos if v.get("split") == "test"]

    print(f"[eval] Test videos: {len(test_videos)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class accumulators (canonical keys)
    pred_by_class: Dict[str, List[Dict]] = defaultdict(list)
    gt_by_class: Dict[str, List[Dict]] = defaultdict(list)
    clip_f1_types_all: List[float] = []
    clip_f1_any_all: List[float] = []
    per_video_results: List[Dict] = []

    for video_info in tqdm(test_videos, desc="Evaluating test videos", disable=not sys.stderr.isatty()):
        video_id = video_info["video_id"]
        video_path = Path(args.video_root) / video_info["video_path"]
        annotations = video_info.get("annotations", [])

        if not video_path.exists():
            print(f"[WARN] Missing: {video_path}")
            continue

        try:
            frames_np, duration = extract_frames_at_fps(
                str(video_path), target_fps=args.fps, img_size=args.img_size
            )
        except Exception as exc:
            print(f"[WARN] Frame extraction failed for {video_id}: {exc}")
            continue

        window_preds = sliding_window_inference(
            frames_np, model, transform,
            thr_any=thr_any, thr_map=thr_map,
            target_fps=args.fps,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            batch_size=args.batch_size,
            device=device,
        )

        clip_m = compute_clip_level_metrics(window_preds, annotations)
        clip_f1_types_all.append(clip_m["f1_types"])
        clip_f1_any_all.append(clip_m["f1_any"])

        pred_segments = merge_window_predictions(
            window_preds, CANON_BUG_TYPES,
            min_conf=args.min_conf,
            min_duration=args.min_duration,
            max_gap_sec=args.max_gap,
        )

        # Accumulate per-class predictions (with confidence for AP ranking)
        for seg in pred_segments:
            c_key = norm_bug_type(seg["bug_type"])
            pred_by_class[c_key].append({
                "start": seg["start"],
                "end": seg["end"],
                "confidence": seg.get("confidence", 0.5),
            })

        # Accumulate per-class GT annotations
        for ann in annotations:
            c_key = norm_bug_type(ann["bug_type"])
            gt_by_class[c_key].append({"start": ann["start"], "end": ann["end"]})

        per_video_results.append({
            "video_id": video_id,
            "duration": round(duration, 3),
            "num_windows": clip_m["num_windows"],
            "num_pred_segments": len(pred_segments),
            "num_gt_annotations": len(annotations),
            "clip_f1_types": clip_m["f1_types"],
            "clip_f1_any": clip_m["f1_any"],
        })

    temporal_m = compute_temporal_metrics(
        dict(pred_by_class), dict(gt_by_class), tiou_thresholds=[0.3, 0.5, 0.7]
    )

    results = {
        "clip_level": {
            "f1_types": round(float(np.mean(clip_f1_types_all)), 4) if clip_f1_types_all else 0.0,
            "f1_any": round(float(np.mean(clip_f1_any_all)), 4) if clip_f1_any_all else 0.0,
            "num_videos_evaluated": len(per_video_results),
        },
        "temporal_localization": temporal_m,
        "per_video": per_video_results,
        "config": {
            "checkpoint": str(args.checkpoint),
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "fps": args.fps,
            "img_size": args.img_size,
            "thr_any": thr_any,
            "thr_map": thr_map,
        },
    }

    out_path = out_dir / "localization_test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── W&B logging ──────────────────────────────────────────────────────────
    use_wandb = (not getattr(args, "no_wandb", False)) and WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(
                project=getattr(args, "wandb_project", "localization-bugs"),
                name=f"eval_{Path(args.checkpoint).stem}",
                config=results["config"],
                tags=["eval", "localization"],
            )
            flat_metrics = {
                "clip/f1_types": results["clip_level"]["f1_types"],
                "clip/f1_any": results["clip_level"]["f1_any"],
                "temporal/mAP_0.3": temporal_m["mAP_0.3"],
                "temporal/mAP_0.5": temporal_m["mAP_0.5"],
                "temporal/mAP_0.7": temporal_m["mAP_0.7"],
                "temporal/mean_tIoU": temporal_m["mean_tIoU"],
                "temporal/recall_tIoU_0.3": temporal_m["segment_recall"]["tIoU_0.3"],
                "temporal/recall_tIoU_0.5": temporal_m["segment_recall"]["tIoU_0.5"],
                "temporal/recall_tIoU_0.7": temporal_m["segment_recall"]["tIoU_0.7"],
            }
            for cls, m in temporal_m["per_class"].items():
                for thr_key, ap_val in m.items():
                    flat_metrics[f"per_class/{cls}/{thr_key}"] = ap_val
            wandb.log(flat_metrics)
            wandb.finish()
            print("[eval] Metrics logged to W&B.")
        except Exception as exc:
            print(f"[WARN] W&B logging failed: {exc}")
    elif not WANDB_AVAILABLE:
        print("[eval] wandb not installed — skipping experiment tracking. Install: pip install wandb")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CLIP-LEVEL METRICS (averaged over test videos)")
    print(f"  F1 Types (micro) : {results['clip_level']['f1_types']:.4f}")
    print(f"  F1 Presence      : {results['clip_level']['f1_any']:.4f}")
    print(f"\nTEMPORAL LOCALIZATION METRICS")
    print(f"  mAP @ tIoU=0.3   : {temporal_m['mAP_0.3']:.4f}")
    print(f"  mAP @ tIoU=0.5   : {temporal_m['mAP_0.5']:.4f}")
    print(f"  mAP @ tIoU=0.7   : {temporal_m['mAP_0.7']:.4f}")
    print(f"  mean tIoU        : {temporal_m['mean_tIoU']:.4f}")
    print(f"\nPER-CLASS AP @ tIoU=0.5:")
    for cls, m in temporal_m["per_class"].items():
        print(f"  {cls:25s}: {m['AP_0.5']:.4f}")
    print(f"\nSEGMENT RECALL:")
    for k, v in temporal_m["segment_recall"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults saved → {out_path}")


# ── Infer mode (called by backend) ───────────────────────────────────────────

def run_infer(args: argparse.Namespace) -> str:
    """
    Run inference on a single video. Returns path to result.json.
    Compatible with backend/inference.py job result schema.
    """
    device = torch.device(
        args.device if (args.device == "cpu" or not torch.cuda.is_available()) else args.device
    )

    model, thr_any, thr_map = load_checkpoint(args.checkpoint, device)
    if args.thr_any is not None:
        thr_any = args.thr_any

    transform = get_transform(args.img_size)

    video_path = str(args.video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[infer] Extracting frames from: {video_path}")
    frames_np, duration = extract_frames_at_fps(
        video_path, target_fps=args.fps, img_size=args.img_size
    )
    print(f"[infer] Duration: {duration:.1f}s | Frames: {frames_np.shape[0]}")

    print("[infer] Running sliding-window inference...")
    window_preds = sliding_window_inference(
        frames_np, model, transform,
        thr_any=thr_any, thr_map=thr_map,
        target_fps=args.fps,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        batch_size=args.batch_size,
        device=device,
    )

    segments = merge_window_predictions(
        window_preds, CANON_BUG_TYPES,
        min_conf=args.min_conf,
        min_duration=args.min_duration,
        max_gap_sec=args.max_gap,
    )
    print(f"[infer] Detected {len(segments)} bug segments")

    # Backend schema: one entry per bug_type per segment (same start/end for multi-label)
    # The frontend groups entries with the same (start, end) into multi-label display segments.
    result = {
        "video_id": Path(video_path).stem,
        "video_path": video_path,
        "duration": round(duration, 3),
        "annotations": [
            {
                "bug_type": seg["bug_type"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "segment_id": seg["segment_id"],
            }
            for seg in segments
        ],
    }

    out_path = args.inference_out
    if not out_path:
        out_dir = Path(args.out_dir) if args.out_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "result.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[infer] Result → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Multi-label bug detection + temporal localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--mode", choices=["eval", "infer"], default="eval",
                    help="eval: run on test set + compute metrics; infer: single video for backend")

    # Paths
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to BugBiLSTM .pt checkpoint from train_FINAL_ALL_METRICS.py")
    ap.add_argument("--temporal-dataset", type=str, default="temporal_bug_dataset.json",
                    help="Path to temporal_bug_dataset.json (eval mode)")
    ap.add_argument("--video-root", type=str, default=".",
                    help="Root folder containing game subfolders with full-resolution videos (eval mode)")
    ap.add_argument("--video-path", type=str, default=None,
                    help="Path to single video file (infer mode)")
    ap.add_argument("--out-dir", type=str, default="./localization_outputs",
                    help="Output directory for metrics JSON / result.json")
    ap.add_argument("--inference-out", type=str, default=None,
                    help="Exact output path for result.json (infer mode); overrides --out-dir")

    # Sliding window
    ap.add_argument("--window-sec", type=float, default=2.0,
                    help="Window duration in seconds (must match training clip length)")
    ap.add_argument("--stride-sec", type=float, default=1.0,
                    help="Stride between windows in seconds")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="Frame sampling rate (must match training FPS)")
    ap.add_argument("--img-size", type=int, default=224,
                    help="Frame resize dimension (must match training resolution)")

    # Thresholds
    ap.add_argument("--thr-any", type=float, default=None,
                    help="Override presence threshold from checkpoint (default: use saved value)")

    # Segment merging / post-processing
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Discard merged segments whose mean window confidence is below this value "
                         "(0.0 = keep all; recommended: 0.35 to reduce over-prediction)")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="Discard merged segments shorter than this many seconds "
                         "(0.0 = keep all; recommended: 1.0)")
    ap.add_argument("--max-gap", type=float, default=float("inf"),
                    help="Maximum gap in seconds between consecutive positive windows to still merge "
                         "(inf = always merge if overlapping/adjacent; recommended: 1.0)")

    # Runtime
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Number of windows per model forward pass")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers (unused for sliding window; reserved)")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Compute device: cuda or cpu")

    # Experiment tracking
    ap.add_argument("--wandb-project", type=str, default="localization-bugs",
                    help="W&B project name for logging eval metrics")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging even if wandb is installed")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "infer":
        if not args.video_path:
            raise ValueError("--video-path is required for --mode infer")
        run_infer(args)


if __name__ == "__main__":
    main()
