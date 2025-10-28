#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TimeSformer fine-tuning for multi-task bug detection:
 - Backbone: facebook/timesformer-base-finetuned-k400 (Divided Space–Time attention)
 - Heads: types (multi-label), any (binary), count (0/1/2/3+)
 - Losses: BCEWithLogits (types + any) + CE/focal (count) + consistency (any ≈ max(types))
 - Val-time threshold tuning (τ_any and per-class τ) and gated metrics
 - TensorBoard logging compatible with your previous setup

Requires:
  pip install transformers timm decord torchvision torch tensorboard
"""

import os, sys, json, math, time, argparse, glob, gzip, io, random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data._utils.collate import default_collate

from torchvision import transforms
try:
    from torchvision.io import read_video
    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

# Optional fast video backend
try:
    import decord
    decord.bridge.set_bridge('torch')
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

# OpenCV last-resort
try:
    import cv2  # noqa
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Hugging Face TimeSformer
from transformers import TimesformerModel, AutoConfig

# ---------------- Canonical bug types ----------------
CANON_BUG_TYPES = ["z-clipping","corrupted_texture","geometry_corruption","z-fighting","boundary_hole"]
_ALIAS = {
    "z_clipping":"z-clipping","z_fighting":"z-fighting","boundary_hole":"boundary_hole",
    "corrupted_texture":"corrupted_texture","geometry_corruption":"geometry_corruption",
}
BUG2IDX = {b:i for i,b in enumerate(CANON_BUG_TYPES)}
def norm_bug_type(s:str)->str:
    s2 = s.strip().lower().replace(" ","_").replace("-","_")
    canon = _ALIAS.get(s2, s2)
    if canon in ("z_clipping","z-clipping"): return "z-clipping"
    if canon in ("z_fighting","z-fighting"): return "z-fighting"
    return canon

# ---------------- JSONL helpers ----------------
def open_maybe_gzip(path: Path):
    return io.TextIOWrapper(gzip.open(path,"rb"),encoding="utf-8") if str(path).endswith(".gz") else path.open("r",encoding="utf-8")

def discover_manifests(root: Path, pattern: str) -> List[Path]:
    if root.is_file(): return [root]
    paths = [Path(p) for p in glob.glob(str(root/pattern), recursive=True)]
    return sorted(p for p in paths if p.suffix in (".jsonl",".json",".gz"))

def load_all_records(manifest_paths):
    recs, bad = [], 0
    for mp in manifest_paths:
        base_dir = str(Path(mp).parent)
        with open_maybe_gzip(Path(mp)) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1; continue
                obj["_base_dir"] = base_dir
                recs.append(obj)
    if bad: print(f"[WARN] Skipped {bad} malformed JSON lines.")
    return recs

# ---------------- Video IO ----------------
def load_video_tensor(path: str, num_frames=16) -> torch.Tensor:
    path = str(path)

    # --- 1) OpenCV first (most reliable on Windows)
    if _HAS_CV2:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cv2.VideoCapture failed")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total == 0:
                cap.release()
                raise RuntimeError("Empty video (cv2)")
            idx = np.linspace(0, max(0, total - 1), num_frames).astype(np.int64)
            frames_list = []
            for target in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                ret, frame = cap.read()
                if not ret or frame is None:
                    frames_list.append(torch.zeros((224, 224, 3), dtype=torch.uint8))
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame))
            cap.release()
            return torch.stack(frames_list, dim=0).to(torch.uint8)
        except Exception as e:
            print(f"[opencv fallback] {e}")

    # --- 2) torchvision.io (PyAV)
    if _HAS_TVIO:
        try:
            video, _, _ = read_video(path, pts_unit='sec')  # [T,H,W,C] uint8/float
            if video.numel() == 0:
                raise RuntimeError("Empty video (torchvision)")
            if video.dtype != torch.uint8:
                video = torch.clamp(video, 0, 255).to(torch.uint8)
            t = video.shape[0]
            idx = torch.linspace(0, max(0, t - 1), steps=num_frames).round().to(torch.long)
            return video.index_select(0, idx)
        except Exception as e:
            print(f"[torchvision.io fallback] {e}")

    # --- 3) decord LAST (but we just uninstalled it)
    if _HAS_DECORD:
        try:
            vr = decord.VideoReader(path)
            n = len(vr)
            if n == 0:
                raise RuntimeError("Empty video (decord)")
            idx = np.linspace(0, max(0, n - 1), num_frames, dtype=np.int64)
            frames = vr.get_batch(idx)  # [T,H,W,C] uint8 as torch via bridge
            return frames
        except Exception as e:
            print(f"[decord fallback] {e}")

    raise RuntimeError("No working video backend (cv2/torchvision/decord all failed).")


def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# ---------------- Dataset ----------------
class BugClipDataset(Dataset):
    """
    Expects each record with fields:
      - relpath / clip_path (video file)
      - bug_types: list[str]
      - num_bugs: int (optional; defaults to len(bug_types), clamped 0..3)
    """
    def __init__(self, data_root: Path, records: List[Dict[str,Any]], split: str,
                 num_frames=16, resize_hw=224):
        self.data_root=Path(data_root)
        self.num_frames=num_frames; self.resize=resize_hw; self.split=split
        self.records=records

        self.targets_multi=[]; self.targets_count=[]; self.targets_any=[]
        for r in self.records:
            y=torch.zeros(len(CANON_BUG_TYPES),dtype=torch.float32)
            types=r.get("bug_types") or r.get("categories") or []
            if isinstance(types,list):
                for t in types:
                    t2=norm_bug_type(str(t))
                    if t2 in BUG2IDX: y[BUG2IDX[t2]]=1.0
            nb=int(r.get("num_bugs", int(y.sum().item())))
            nb=max(0,min(3,nb))
            self.targets_multi.append(y)
            self.targets_count.append(torch.tensor(nb,dtype=torch.long))
            self.targets_any.append(torch.tensor(1 if y.sum()>0 else 0, dtype=torch.float32))

        # Normalization to ImageNet stats (works fine for TimeSformer)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize,self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.tf_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(self.resize*1.14)),
            transforms.RandomCrop((self.resize,self.resize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rel = r.get("relpath") or r.get("clip_path")
        if rel is None:
            return None
        rel_norm = str(rel).replace("\\", os.sep).replace("/", os.sep)
        base_dir = Path(r.get("_base_dir", self.data_root))
        clip_path = Path(rel_norm) if os.path.isabs(rel_norm) else (base_dir / rel_norm)

        try:
            if not clip_path.exists():
                print(f"[WARN] Missing file: {clip_path}"); return None
            frames = load_video_tensor(str(clip_path), num_frames=self.num_frames)  # [T,H,W,C] uint8
            tfm = self.tf_train if self.split == "train" else self.tf
            imgs = [tfm(frames[t].numpy()) for t in range(frames.shape[0])]  # each [3,H,W]
            x = torch.stack(imgs, dim=0)  # [T,3,H,W]  (TimeSformer wants B,T,3,H,W later)

            y_multi = self.targets_multi[idx]
            y_count = self.targets_count[idx]
            y_any   = self.targets_any[idx]
            return x, y_multi, y_count, y_any, str(clip_path)
        except Exception as e:
            print(f"[WARN] Failed to load clip ({self.split}): {clip_path} -> {e}")
            return None

# ---------------- Model: TimeSformer + heads ----------------
class TimeSformerWithHeads(nn.Module):
    def __init__(self, num_bug_types: int, pretrained_name: str = "facebook/timesformer-base-finetuned-k400",
                 frozen_stages: int = 0):
        super().__init__()
        # Load config/model; keep hidden size for heads
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.backbone = TimesformerModel.from_pretrained(pretrained_name, config=self.config)
        self.hidden_dim = self.config.hidden_size  # typically 768 for base

        # Optionally freeze some encoder blocks (if memory/overfit)
        if frozen_stages > 0:
            # Freeze patch/embed + first N blocks
            for p in self.backbone.parameters():
                p.requires_grad = True  # set all trainable first
            # Freeze embeddings
            if frozen_stages >= 1:
                for p in self.backbone.embeddings.parameters():
                    p.requires_grad = False
            # Freeze first few encoder layers
            num_layers = len(self.backbone.encoder.layer)
            k = min(frozen_stages-1, num_layers)  # stages>1 start freezing layers
            for i in range(k):
                for p in self.backbone.encoder.layer[i].parameters():
                    p.requires_grad = False

        # Three heads
        self.fc_types = nn.Linear(self.hidden_dim, num_bug_types)
        self.fc_count = nn.Linear(self.hidden_dim, 4)
        self.fc_any   = nn.Linear(self.hidden_dim, 1)

    def forward(self, frames: torch.Tensor):
        """
        frames: [B,T,3,H,W] float32 normalized
        Returns: logits_types [B,C], logits_count [B,4], logit_any [B]
        """
        # HF TimeSformer expects pixel_values [B, T, C, H, W]
        out = self.backbone(pixel_values=frames, output_hidden_states=False, return_dict=True)
        # CLS token as global representation
        cls = out.last_hidden_state[:, 0]  # [B, hidden]
        logits_types = self.fc_types(cls)
        logits_count = self.fc_count(cls)
        logit_any    = self.fc_any(cls).squeeze(-1)
        return logits_types, logits_count, logit_any

# ---------------- Metrics ----------------
@torch.no_grad()
def multilabel_micro_f1(logits, targets, thr=0.5):
    preds=(torch.sigmoid(logits)>=thr).to(torch.int); t=targets.to(torch.int)
    tp=(preds & t).sum().item(); fp=(preds & (~t.bool())).sum().item()
    fn=((~preds.bool()) & t.bool()).sum().item()
    denom=(2*tp+fp+fn); return (2*tp/denom) if denom>0 else 0.0

@torch.no_grad()
def accuracy_count(logits, targets):
    pred=logits.argmax(dim=1); return (pred==targets).float().mean().item()

@torch.no_grad()
def presence_f1(any_logits, any_targets, thr=0.5):
    p=(torch.sigmoid(any_logits)>=thr).to(torch.int)
    y=any_targets.to(torch.int)
    tp=(p & y).sum().item(); fp=(p & (~y.bool())).sum().item(); fn=((~p.bool()) & y.bool()).sum().item()
    denom=(2*tp+fp+fn); return (2*tp/denom) if denom>0 else 0.0

# ---------------- Train / Val epoch ----------------
def run_epoch(model, loader, opt, scaler, device, *,
              amp=True, train=True,
              bce_multi=None, count_criterion=None, count_lambda=0.25,
              any_lambda=1.0, cons_lambda=0.1, sched=None):
    model.train(train)
    totL = tot = totF1_types = totAcc_cnt = totF1_any = 0.0

    for batch in loader:
        if batch is None: continue
        x, y_multi, y_count, y_any, _paths = batch
        # x: [T,3,H,W] -> model expects [B,T,3,H,W]
        x = x.to(device, non_blocking=True)                      # [B,T,3,H,W] already (via collate)
        y_multi = y_multi.to(device, non_blocking=True)
        y_count = y_count.to(device, non_blocking=True)
        y_any   = y_any.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type=='cuda')):
            logits_types, logits_count, logit_any = model(x)
            # types loss
            loss_types = (F.binary_cross_entropy_with_logits(logits_types, y_multi)
                          if bce_multi is None else bce_multi(logits_types, y_multi))
            # count loss
            loss_count = (F.cross_entropy(logits_count, y_count)
                          if count_criterion is None else count_criterion(logits_count, y_count))
            # presence loss
            loss_any = F.binary_cross_entropy_with_logits(logit_any, y_any)
            # consistency: any ≈ max over types
            with torch.no_grad():
                max_type_prob = torch.sigmoid(logits_types).amax(dim=1)
            any_prob = torch.sigmoid(logit_any)
            loss_cons = F.mse_loss(any_prob, max_type_prob)

            loss = loss_types + count_lambda*loss_count + any_lambda*loss_any + cons_lambda*loss_cons

        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if sched is not None:
                sched.step()

        with torch.no_grad():
            f1_types = multilabel_micro_f1(logits_types, y_multi)
            acc_cnt  = accuracy_count(logits_count, y_count)
            f1_any   = presence_f1(logit_any, y_any)

        bs = x.size(0)
        tot += bs
        totL += loss.item()*bs
        totF1_types += f1_types*bs
        totAcc_cnt  += acc_cnt*bs
        totF1_any   += f1_any*bs

    if tot == 0: return (0.0,)*4
    return totL/tot, totF1_types/tot, totF1_any/tot, totAcc_cnt/tot

# ---------------- Utils ----------------
def seed_everything(seed:int=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def build_label_matrix(records, min_combo_support=5):
    N = len(records); C = len(CANON_BUG_TYPES)
    Y_bug = np.zeros((N, C), dtype=np.int64)
    counts = np.zeros(N, dtype=np.int64)
    combos = []
    for i, r in enumerate(records):
        types = r.get("bug_types") or r.get("categories") or []
        present = []
        for t in types if isinstance(types, list) else []:
            t2 = norm_bug_type(str(t))
            if t2 in BUG2IDX:
                Y_bug[i, BUG2IDX[t2]] = 1; present.append(t2)
        c = int(r.get("num_bugs", len(present))); c = max(0, min(3, c)); counts[i]=c
        combos.append("NONE" if len(present)==0 else "|".join(sorted(set(present))))
    any_bug = (Y_bug.sum(1) > 0).astype(np.int64).reshape(-1, 1)
    count_bucket = np.zeros((N, 4), dtype=np.int64)
    for i, c in enumerate(counts): count_bucket[i, min(c,3)] = 1
    combo_counts = Counter(combos)
    frequent = [k for k, v in combo_counts.items() if v >= min_combo_support and k != "NONE"]
    combo_to_col = {name: idx for idx, name in enumerate(frequent)}
    Y_combo = np.zeros((N, len(frequent)), dtype=np.int64)
    if frequent:
        for i, name in enumerate(combos):
            if name in combo_to_col: Y_combo[i, combo_to_col[name]] = 1
    Y = np.concatenate([Y_bug, any_bug, count_bucket, Y_combo], axis=1)
    label_names = list(CANON_BUG_TYPES) + ["any_bug"] + ["count_0","count_1","count_2","count_3+"] + [f"combo:{c}" for c in frequent]
    return Y, label_names

def stratified_splits(records, train_ratio=0.8, val_ratio=0.1, seed=1337, min_combo_support=5):
    N = len(records)
    idx = np.arange(N)
    Y, _ = build_label_matrix(records, min_combo_support=min_combo_support)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and test_ratio >= 0
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
        msss = MSSS(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
        train_idx, temp_idx = next(msss.split(idx, Y))
        temp_Y = Y[temp_idx]
        if (val_ratio + test_ratio) == 0:
            val_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
        elif test_ratio == 0:
            val_idx, test_idx = temp_idx, np.array([], dtype=int)
        else:
            rel = test_ratio / (val_ratio + test_ratio)
            msss2 = MSSS(n_splits=1, test_size=rel, random_state=seed+1)
            val_rel_idx, test_rel_idx = next(msss2.split(temp_idx, temp_Y))
            val_idx, test_idx = temp_idx[val_rel_idx], temp_idx[test_rel_idx]
    except Exception:
        rng = np.random.RandomState(seed); rng.shuffle(idx)
        n_train = int(N * train_ratio); n_val = int(N * val_ratio)
        train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

@torch.no_grad()
def collect_logits_targets(model, loader, device):
    model.eval()
    S_types, Y_types = [], []
    S_any, Y_any = [], []
    for batch in loader:
        if batch is None: continue
        x, y_multi, _y_count, y_any, _paths = batch
        x = x.to(device, non_blocking=True)
        logits_types, _count, logit_any = model(x)
        S_types.append(torch.sigmoid(logits_types).cpu())
        Y_types.append(y_multi.cpu())
        S_any.append(torch.sigmoid(logit_any).cpu())
        Y_any.append(y_any.cpu())
    if not S_types: return None, None, None, None
    return torch.cat(S_types,0), torch.cat(Y_types,0), torch.cat(S_any,0), torch.cat(Y_any,0)

def tune_thresholds_np(S, Y, class_names, grid=None):
    import numpy as np
    if grid is None: grid = np.linspace(0.05,0.95,19)
    thr_map = {}
    S = S.numpy(); Y = Y.numpy().astype(int)
    for k,name in enumerate(class_names):
        s = S[:,k]; y = Y[:,k]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            p = (s >= t).astype(int)
            tp = (p & y).sum(); fp = (p & (1-y)).sum(); fn = ((1-p) & y).sum()
            f1 = 2*tp / (2*tp + fp + fn + 1e-9)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        thr_map[name] = best_t
    return thr_map

def tune_presence_threshold(S_any, Y_any, grid=None, target_recall=None):
    import numpy as np
    if grid is None: grid = np.linspace(0.01,0.99,99)
    s = S_any.numpy(); y = Y_any.numpy().astype(int)
    best = {"f1":-1.0, "thr":0.5, "prec":0.0, "rec":0.0}
    for t in grid:
        p = (s >= t).astype(int)
        tp = (p & y).sum(); fp=(p & (1-y)).sum(); fn=((1-p) & y).sum()
        prec = tp / (tp+fp+1e-9); rec = tp / (tp+fn+1e-9)
        f1 = 2*prec*rec / (prec+rec+1e-9)
        if target_recall is not None:
            if rec >= target_recall and (t < best["thr"] or best["f1"]<0):
                best = {"f1":f1, "thr":float(t), "prec":prec, "rec":rec}
        else:
            if f1 > best["f1"]:
                best = {"f1":f1, "thr":float(t), "prec":prec, "rec":rec}
    return best

@torch.no_grad()
def gated_types_micro_f1(S_types, Y_types, S_any, thr_any, thr_map, class_names):
    import numpy as np
    s_types = S_types.numpy()
    y = Y_types.numpy().astype(int)
    s_any = S_any.numpy()
    pred_any = (s_any >= float(thr_any)).astype(int)[:, None]
    thr_vec = np.array([thr_map.get(n, 0.5) for n in class_names])[None, :]
    p_types = (s_types >= thr_vec).astype(int) * pred_any
    tp = np.logical_and(p_types == 1, y == 1).sum()
    fp = np.logical_and(p_types == 1, y == 0).sum()
    fn = np.logical_and(p_types == 0, y == 1).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return float(f1)

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--manifests", type=str, nargs="*", default=None)
    ap.add_argument("--manifests-glob", type=str, default="**/*.jsonl")
    # splits
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    # train
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--logdir", type=str, default="runs/bugdetector_timesformer")
    ap.add_argument("--save-dir", type=str, default="checkpoints")
    # count head options
    ap.add_argument("--count-weighting", type=str, default="inv", choices=["none","inv","effective"])
    ap.add_argument("--count-focal", action="store_true")
    ap.add_argument("--count-gamma", type=float, default=2.0)
    ap.add_argument("--count-label-smooth", type=float, default=0.05)
    ap.add_argument("--count-lambda", type=float, default=0.25)
    # any head + consistency
    ap.add_argument("--any-lambda", type=float, default=1.0)
    ap.add_argument("--cons-lambda", type=float, default=0.1)
    # LR schedule
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--use-cosine", action="store_true")
    # EMA freeze stages (optional tiny regularization)
    ap.add_argument("--frozen-stages", type=int, default=0, help="0=train all, 1=freeze embeddings, >1 also freeze first N-1 encoder blocks")
    # presence recall tuning
    ap.add_argument("--presence-recall", type=float, default=None, help="If set (e.g. 0.95), pick smallest τ_any achieving ≥ recall on val")
    # resume
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-reset-lr", type=float, default=None)
    args=ap.parse_args()

    seed_everything(args.seed)
    data_root=Path(args.data_root)
    manifests=[Path(m) for m in args.manifests] if args.manifests else discover_manifests(data_root, args.manifests_glob)
    if not manifests: print("No manifests found.", file=sys.stderr); sys.exit(1)
    all_records=load_all_records(manifests)
    if not all_records: print("No records loaded.", file=sys.stderr); sys.exit(1)

    tr_idx, va_idx, _ = stratified_splits(all_records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed, min_combo_support=5)
    train_recs=[all_records[i] for i in tr_idx]
    val_recs  =[all_records[i] for i in va_idx]
    print(f"Discovered manifests: {len(manifests)} | Total clips: {len(all_records)}")
    print(f"Split -> train: {len(train_recs)} | val: {len(val_recs)}")

    # Datasets
    ds_train=BugClipDataset(data_root, train_recs, split="train", num_frames=16, resize_hw=224)
    ds_val  =BugClipDataset(data_root, val_recs,   split="val",   num_frames=16, resize_hw=224)

    # Count class weights
    count_hist = torch.zeros(4, dtype=torch.float32)
    for y in ds_train.targets_count: count_hist[y.item()] += 1
    Ncnt = count_hist.sum().clamp_min(1.0)
    mode = args.count_weighting
    if mode == "none":
        count_weights = torch.ones_like(count_hist)
    elif mode == "inv":
        count_weights = (Ncnt / (len(count_hist) * count_hist.clamp_min(1.0)))
    elif mode == "effective":
        beta=0.999
        eff_num = (1.0 - beta**count_hist) / (1.0 - beta)
        count_weights = (1.0 / eff_num).float()
    else:
        count_weights = torch.ones_like(count_hist)
    count_weights = count_weights / count_weights.mean()
    print("Count histogram (train) [0,1,2,3+]:", count_hist.tolist())
    print("Count class weights     [0,1,2,3+]:", count_weights.tolist())

    # pos_weight for BCE types
    with torch.no_grad():
        Ytrain = torch.stack(ds_train.targets_multi)
        pos = Ytrain.sum(0).clamp_min(1.0)
        N = torch.tensor(len(ds_train), dtype=torch.float32)
        pos_weight_types = ((N - pos) / pos).float()

    def bce_with_pos_weight(logits, targets, device, pos_w):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w.to(device))

    # Loaders
    train_loader=DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        collate_fn=collate_drop_none
    )
    val_loader  =DataLoader(
        ds_val, batch_size=max(1,args.batch_size//2), shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        collate_fn=collate_drop_none
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:" , (torch.cuda.get_device_name(0) if device.type=='cuda' else "CPU"))

    model=TimeSformerWithHeads(num_bug_types=len(CANON_BUG_TYPES),
                               pretrained_name="facebook/timesformer-base-finetuned-k400",
                               frozen_stages=args.frozen_stages).to(device)
    
    
    try:
        model.backbone.gradient_checkpointing_enable()
        print("[info] HF gradient checkpointing enabled")
    except Exception as e:
        print("[info] gradient checkpointing not available:", e)


    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type=='cuda'))

    # Cosine LR with warmup (by steps)
    sched = None
    if args.use_cosine:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * max(1, args.epochs)
        warmup_steps = int(steps_per_epoch * max(0.0, args.warmup_epochs))
        eta_min = 1e-6
        base_lr = args.lr
        floor_ratio = eta_min / base_lr
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Loss fns
    bce_multi = lambda logits, targets: bce_with_pos_weight(logits, targets, device, pos_weight_types)
    cw = count_weights.to(device)
    if args.count_focal:
        def focal_ce(logits, targets, alpha=None, gamma=2.0, label_smoothing=0.0):
            ce = F.cross_entropy(logits, targets, reduction='none', weight=alpha, label_smoothing=label_smoothing)
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** gamma) * ce
            return loss.mean()
        def count_criterion(logits, targets):
            return focal_ce(logits, targets, alpha=cw, gamma=args.count_gamma, label_smoothing=args.count_label_smooth)
    else:
        def count_criterion(logits, targets):
            return F.cross_entropy(logits, targets, weight=cw, label_smoothing=args.count_label_smooth)

    # Resume (optional)
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "opt_state" in ckpt and ckpt["opt_state"] is not None:
            opt.load_state_dict(ckpt["opt_state"])
        if "scaler_state" in ckpt and ckpt["scaler_state"] is not None and scaler is not None:
            try: scaler.load_state_dict(ckpt["scaler_state"])
            except Exception: print("[Resume] scaler_state incompatible; skipping.")
        if "sched_state" in ckpt and ckpt["sched_state"] is not None and sched is not None:
            try: sched.load_state_dict(ckpt["sched_state"])
            except Exception: print("[Resume] sched_state incompatible; skipping.")
        if args.resume_reset_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(args.resume_reset_lr)
            print(f"[Resume] LR reset to {args.resume_reset_lr}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] Starting at epoch {start_epoch}")

    # TensorBoard
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    for i,v in enumerate(count_hist.tolist()): writer.add_scalar(f"train_count_hist/class_{i}", v, 0)
    for i,wv in enumerate(count_weights.tolist()): writer.add_scalar(f"train_count_weight/class_{i}", wv, 0)

    os.makedirs(args.save_dir,exist_ok=True)
    best_f1=-1.0; best_path=None

    global_step = 0
    for epoch in range(start_epoch, args.epochs+1):
        t0=time.time()
        trL,trF1_types,trF1_any,trAcc_cnt = run_epoch(
            model,train_loader,opt,scaler,device,
            amp=(not args.no_amp),train=True,
            bce_multi=bce_multi, count_criterion=count_criterion,
            count_lambda=args.count_lambda, any_lambda=args.any_lambda,
            cons_lambda=args.cons_lambda, sched=sched
        )
        global_step += len(train_loader)

        # ---- Tune thresholds on val set ----
        S_types, Y_types, S_any, Y_any = collect_logits_targets(model, val_loader, device)
        thr_any_obj = {"thr": 0.5, "f1": 0.0, "prec": 0.0, "rec": 0.0}
        thr_map = {name: 0.5 for name in CANON_BUG_TYPES}
        f1_types_gated = None

        if (S_types is not None) and (S_any is not None):
            target_recall = args.presence_recall if args.presence_recall is not None else 0.95
            thr_any_obj = tune_presence_threshold(S_any, Y_any, target_recall=target_recall)
            tau_any = float(thr_any_obj["thr"])
            mask = (S_any.numpy() >= tau_any)
            MIN_GATED = 100
            if mask.sum() >= MIN_GATED:
                grid = np.linspace(0.2, 0.8, 13)
                thr_map = tune_thresholds_np(S_types[mask], Y_types[mask], CANON_BUG_TYPES, grid=grid)
                thr_map = {k: float(np.clip(v, 0.2, 0.8)) for k, v in thr_map.items()}
            f1_types_gated = gated_types_micro_f1(S_types, Y_types, S_any, tau_any, thr_map, CANON_BUG_TYPES)

        # ---- Ungated val pass (for direct comparability) ----
        vaL, vaF1_types, vaF1_any, vaAcc_cnt = run_epoch(
            model, val_loader, opt, scaler, device,
            amp=False, train=False,
            bce_multi=bce_multi, count_criterion=count_criterion,
            count_lambda=args.count_lambda, any_lambda=args.any_lambda,
            cons_lambda=args.cons_lambda
        )
        dt = time.time() - t0

        # ---- Logging ----
        step = epoch
        writer.add_scalar("loss/train", float(trL), step)
        writer.add_scalar("loss/val",   float(vaL), step)
        writer.add_scalar("f1_types/train", float(trF1_types), step)
        writer.add_scalar("f1_types/val",   float(vaF1_types), step)
        writer.add_scalar("f1_any/train", float(trF1_any), step)
        writer.add_scalar("f1_any/val",   float(vaF1_any), step)
        writer.add_scalar("acc_count/train", float(trAcc_cnt), step)
        writer.add_scalar("acc_count/val",   float(vaAcc_cnt), step)
        writer.add_scalar("f1_types_gated/val", float(f1_types_gated or 0.0), step)
        writer.add_scalar("lr", float(opt.param_groups[0]['lr']), step)

        print(f"[Epoch {epoch:02d}] loss_tr={trL:.4f} loss_va={vaL:.4f} | "
              f"typesF1_tr={trF1_types:.3f} typesF1_va={vaF1_types:.3f} | "
              f"anyF1_tr={trF1_any:.3f} anyF1_va={vaF1_any:.3f} | "
              f"cntAcc_tr={trAcc_cnt:.3f} cntAcc_va={vaAcc_cnt:.3f} ({dt/60:.1f} min)")
        if S_any is not None:
            print(f"  tuned τ_any={thr_any_obj['thr']:.2f} (val prec={thr_any_obj['prec']:.2f}, rec={thr_any_obj['rec']:.2f})")
        if f1_types_gated is not None:
            print(f"  gated_typesF1_va={f1_types_gated:.3f}")

        # ---- Save best on ungated types micro-F1 (same criterion as before) ----
        score = vaF1_types
        if score > best_f1:
            best_f1 = score
            best_path = Path(args.save_dir)/f"timesformer_anygate_best_epoch{epoch:02d}_typesF1{vaF1_types:.3f}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": (sched.state_dict() if sched is not None else None),
                "scaler_state": (scaler.state_dict() if scaler is not None else None),
                "val_types_f1": vaF1_types,
                "val_any_f1": vaF1_any,
                "val_count_acc": vaAcc_cnt,
                "val_loss": vaL,
                "config": vars(args),
                "bug_types": CANON_BUG_TYPES,
                "thr_map": thr_map,
                "thr_any": thr_any_obj["thr"],
            }, best_path)
            print(f"  ↳ Saved new best: {best_path}")

    writer.close()
    print("Done. Best val types micro-F1:", f"{best_f1:.3f}" if best_f1>=0 else "N/A")
    if best_path: print("Best checkpoint:", str(best_path))

if __name__=="__main__":
    main()
