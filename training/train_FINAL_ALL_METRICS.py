#!/usr/bin/env python3
# ==============================================================================
# ORIGINAL CONFIG VERSION - No Early Stopping, No ReduceLROnPlateau
# ==============================================================================
# This version runs your original 60-epoch configuration with:
# - All W&B logging and monitoring
# - Checkpoint every epoch (keeps last 3)
# - Data loading from my_new_splits.jsonl
# - TensorBoard logging
# - BUT: No early stopping, runs full 60 epochs
# - AND: Uses --use-cosine flag controls scheduler (as in your original command)
# ==============================================================================
import os, sys, json, math, time, argparse, glob, gzip, io, random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from torch import amp as torch_amp

# Weights & Biases for remote monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Install with: pip install wandb")

# Visualization and detailed metrics
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ---------- Backends ----------
_BACKENDS = {}
try:
    import decord
    decord.bridge.set_bridge('torch'); _BACKENDS['decord'] = True
except Exception: _BACKENDS['decord'] = False
try:
    from torchvision.io import read_video; _BACKENDS['tvio'] = True
except Exception: _BACKENDS['tvio'] = False
try:
    import cv2; _BACKENDS['opencv'] = True
except Exception: _BACKENDS['opencv'] = False

# ---------- Canonical bug types ----------
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

# ---------- JSONL ----------
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
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                obj["_base_dir"] = base_dir
                recs.append(obj)
    if bad:
        print(f"[WARN] Skipped {bad} malformed JSON lines.")
    return recs

# ---------- Video IO ----------
def load_video_tensor(path: str, num_frames=16) -> torch.Tensor:
    path=str(path)
    if _BACKENDS.get('tvio'):
        try:
            video,_,_ = read_video(path, pts_unit='sec')  # [T,H,W,C]
            if video.dtype!=torch.uint8:
                video=torch.clamp(video,0,255).to(torch.uint8)
            t=video.shape[0]; 
            if t==0: raise RuntimeError("Empty video")
            idx=torch.linspace(0,max(0,t-1),steps=num_frames).round().to(torch.long)
            return video.index_select(0,idx)
        except Exception as e: print(f"[torchvision.io fallback] {e}")
    if _BACKENDS.get('opencv'):
        try:
            import cv2
            cap=cv2.VideoCapture(path)
            if not cap.isOpened(): raise RuntimeError("cv2.VideoCapture failed")
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx=np.linspace(0,max(0,total-1),num_frames).astype(np.int64)
            frames_list=[]; current=-1; frame=None
            for target in idx:
                while current<int(target):
                    ret, frame=cap.read()
                    if not ret: break
                    current+=1
                if current!=int(target) or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(target))
                    ret, frame=cap.read()
                    if not ret or frame is None:
                        frames_list.append(torch.zeros((224,224,3),dtype=torch.uint8)); continue
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame))
            cap.release()
            return torch.stack(frames_list,dim=0).to(torch.uint8)
        except Exception as e: print(f"[opencv fallback] {e}")
    if _BACKENDS.get('decord'):
        try:
            vr=decord.VideoReader(path); n=len(vr)
            idx=np.linspace(0,max(0,n-1),num_frames,dtype=np.int64)
            frames=vr.get_batch(idx)  # [T,H,W,C] uint8 torch
            return frames
        except Exception as e: print(f"[decord fallback] {e}")
    raise RuntimeError("No working video backend (install decord or ensure torchvision.io/opencv works).")

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# ---------- Dataset ----------
class BugClipDataset(Dataset):
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
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)], p=0.6),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
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
                print(f"[WARN] Missing file: {clip_path}")
                return None
            frames = load_video_tensor(str(clip_path), num_frames=self.num_frames)  # [T,H,W,C] uint8
            tfm = self.tf_train if self.split == "train" else self.tf
            imgs = [tfm(frames[t].numpy()) for t in range(frames.shape[0])]
            x = torch.stack(imgs, dim=0)  # [T,3,H,W]

            y_multi = self.targets_multi[idx]
            y_count = self.targets_count[idx]
            y_any   = self.targets_any[idx]
            return x, y_multi, y_count, y_any, str(clip_path)
        except Exception as e:
            print(f"[WARN] Failed to load clip ({self.split}): {clip_path} -> {e}")
            return None

# ---------- Model ----------
class TemporalAttention(nn.Module):
    def __init__(self, d_model=512):
        super().__init__(); self.W=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,1,bias=False)
    def forward(self,x):
        h=torch.tanh(self.W(x)); a=torch.softmax(self.v(h).squeeze(-1),dim=1)
        z=(x*a.unsqueeze(-1)).sum(dim=1); return z,a

class BugBiLSTM(nn.Module):
    """
    Heads:
      - types head: C logits (multi-label)
      - count head: 4 logits (0/1/2/3+)
      - any head: 1 logit (any bug vs none)
    """
    def __init__(self, num_bug_types=len(CANON_BUG_TYPES), pretrained=True):
        super().__init__()
        try: enc=resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception: enc=resnet18(weights=None)
        self.cnn=nn.Sequential(*list(enc.children())[:-1]); self.feat_dim=512
        self.lstm=nn.LSTM(input_size=self.feat_dim, hidden_size=256, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.4)
        self.attn=TemporalAttention(d_model=512)
        self.head_drop = nn.Dropout(p=0.3)  # conservative (baseline value)
        self.fc_types=nn.Linear(512,num_bug_types)
        self.fc_count=nn.Linear(512,4)
        self.fc_any  =nn.Linear(512,1)

    def forward(self, frames):  # [B,T,3,224,224]
        B,T,C,H,W=frames.shape; x=frames.reshape(B*T,C,H,W)
        f=self.cnn(x).flatten(1)
        f = f.reshape(B, T, self.feat_dim).contiguous()
        y,_=self.lstm(f); z,att=self.attn(y)        # [B,512]
        z = self.head_drop(z)
        logits_types=self.fc_types(z)               # [B,C]
        logits_count=self.fc_count(z)               # [B,4]
        logit_any=self.fc_any(z).squeeze(-1)        # [B]
        return logits_types, logits_count, logit_any, att

# ---------- Simple EMA wrapper (optional) ----------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update_parameters(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        sd=model.state_dict()
        for k,v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v)

# ---------- Metrics ----------
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

# ---------- Comprehensive Test Metrics & Visualizations ----------

def compute_overall_metrics(y_pred, y_true):
    """
    Compute overall precision, recall, F1 (micro-averaged)
    y_pred, y_true: binary arrays (N, C)
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float((y_pred == y_true).mean())
    }

def compute_per_class_metrics(y_pred_probs, y_true, class_names, thresholds=None):
    """
    Per-class precision/recall/F1 for bug types
    """
    if thresholds is None:
        thresholds = {name: 0.5 for name in class_names}
    elif isinstance(thresholds, (float, int)):
        thresholds = {name: float(thresholds) for name in class_names}
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        thr = thresholds.get(class_name, 0.5)
        y_pred_binary = (y_pred_probs[:, i] >= thr).astype(int)
        y_true_binary = y_true[:, i].astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_true_binary.sum())
        }
    
    return metrics

def compute_count_class_metrics(y_pred, y_true):
    """
    Per-class metrics for count prediction (0, 1, 2, 3 bugs)
    """
    count_labels = ['0 bugs', '1 bug', '2 bugs', '3 bugs']
    metrics = {}
    
    for i, label in enumerate(count_labels):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        metrics[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_true_binary.sum())
        }
    
    return metrics

def compute_combo_metrics(y_pred, y_true, class_names, top_k=20):
    """
    Metrics for frequent bug combinations
    """
    from collections import Counter
    
    def get_combo(arr):
        indices = np.where(arr > 0)[0]
        if len(indices) == 0:
            return "no_bugs"
        return "|".join(sorted([class_names[i] for i in indices]))
    
    y_true_combos = [get_combo(y_true[i]) for i in range(len(y_true))]
    y_pred_combos = [get_combo(y_pred[i]) for i in range(len(y_pred))]
    
    combo_counts = Counter(y_true_combos)
    most_common = [c for c, _ in combo_counts.most_common(top_k)]
    
    combo_metrics = {}
    for combo in most_common:
        y_true_binary = np.array([1 if c == combo else 0 for c in y_true_combos])
        y_pred_binary = np.array([1 if c == combo else 0 for c in y_pred_combos])
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        combo_metrics[combo] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'count': int(combo_counts[combo])
        }
    
    return combo_metrics

def plot_per_class_metrics(metrics, save_path, title="Per-class metrics"):
    """Bar chart for per-class precision/recall/F1"""
    class_names = list(metrics.keys())
    precisions = [metrics[c]['precision'] for c in class_names]
    recalls = [metrics[c]['recall'] for c in class_names]
    f1s = [metrics[c]['f1'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, f1s, width, label='F1', color='#2ca02c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_count_class_metrics(metrics, save_path, title="Per-count metrics"):
    """Bar chart for count class precision/recall/F1"""
    count_labels = list(metrics.keys())
    precisions = [metrics[c]['precision'] for c in count_labels]
    recalls = [metrics[c]['recall'] for c in count_labels]
    f1s = [metrics[c]['f1'] for c in count_labels]
    
    x = np.arange(len(count_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, f1s, width, label='F1', color='#2ca02c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(count_labels, rotation=0, ha='center')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_count(y_true, y_pred, save_path, title="Confusion Matrix (Count)"):
    """Heatmap for count confusion matrix"""
    y_pred_int = y_pred.astype(int)
    y_true_int = y_true.astype(int)
    
    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1, 2, 3])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['0 bugs', '1 bug', '2 bugs', '3 bugs'],
                yticklabels=['0 bugs', '1 bug', '2 bugs', '3 bugs'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_combo_metrics(combo_metrics, save_path, title="Bug Combination Metrics"):
    """Bar chart for bug combination metrics"""
    if len(combo_metrics) == 0:
        return
    
    combos = list(combo_metrics.keys())
    precisions = [combo_metrics[c]['precision'] for c in combos]
    recalls = [combo_metrics[c]['recall'] for c in combos]
    f1s = [combo_metrics[c]['f1'] for c in combos]
    
    x = np.arange(len(combos))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, f1s, width, label='F1', color='#2ca02c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------- Losses ----------
def focal_ce(logits, targets, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):  # <<< NEW: reduction
    ce = F.cross_entropy(logits, targets, reduction='none', weight=alpha, label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    if reduction == "none":
        return loss
    return loss.mean()

def asl_loss(logits, targets, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, reduction="mean"):   # <<< NEW: reduction
    """
    Asymmetric Loss For Multi-Label Classification
    logits: [B,C], targets: [B,C] in {0,1}
    """
    x = logits
    y = targets
    xs_pos = torch.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    log_pos = torch.log(xs_pos.clamp_min(1e-8))
    log_neg = torch.log(xs_neg.clamp_min(1e-8))
    loss_pos = y * ((1 - xs_pos) ** gamma_pos) * log_pos
    loss_neg = (1 - y) * (xs_pos ** gamma_neg) * log_neg
    loss = -(loss_pos + loss_neg)  # [B,C]
    if reduction == "none":
        return loss
    return loss.mean()

# ---------- Train/Val ----------
def run_epoch(model, loader, opt, scaler, device, *,
              amp=True, train=True,
              bce_multi=None, count_criterion=None, count_lambda=0.25,
              any_lambda=1.0, cons_lambda=0.1, sched=None, ema=None, any_pos_weight=None,
              neg_types_weight=0.25, neg_count_weight=0.25):  # <<< NEW: weights
    model.train(train)
    totL = tot = totF1_types = totAcc_cnt = totF1_any = 0.0
    pbar = tqdm(loader, desc=("Train" if train else "Val"), leave=False)

    for batch in pbar:
        if batch is None: continue

        x, y_multi, y_count, y_any, _paths = batch
        x=x.to(device, non_blocking=True)
        y_multi=y_multi.to(device, non_blocking=True)     # [B,C]
        y_count=y_count.to(device, non_blocking=True)     # [B]
        y_any  =y_any.to(device, non_blocking=True)       # [B]

        with torch_amp.autocast('cuda', enabled=(amp and device.type == 'cuda')):
            logits_types, logits_count, logit_any, _ = model(x)

            # ---- Presence loss (as before) ----
            if any_pos_weight is not None:
                loss_any = F.binary_cross_entropy_with_logits(
                    logit_any, y_any, pos_weight=any_pos_weight.to(device)
                )
            else:
                loss_any = F.binary_cross_entropy_with_logits(logit_any, y_any)

            # ---- Types loss with negative down-weighting ----
            # compute per-element loss then weight by mask (positives=1.0, negatives=neg_types_weight)
            if bce_multi is None:
                # default BCE with per-class pos_weight computed outside (kept as before)
                # We need per-element; reproduce BCE here with reduction='none'
                loss_types_elem = F.binary_cross_entropy_with_logits(
                    logits_types, y_multi, reduction='none'
                )  # [B,C]
            else:
                # If using ASL, ask it for per-element loss
                try:
                    loss_types_elem = asl_loss(
                        logits_types, y_multi, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, reduction="none"
                    )
                except TypeError:
                    # fallback to BCE if custom bce_multi provided
                    loss_types_elem = F.binary_cross_entropy_with_logits(
                        logits_types, y_multi, reduction='none'
                    )

            # mask weights: shape [B,1] -> broadcast to [B,C]
            w_types = torch.where(y_any.unsqueeze(1) > 0.5,
                                  torch.ones_like(loss_types_elem),
                                  torch.full_like(loss_types_elem, float(neg_types_weight)))  # <<< NEW
            loss_types = (loss_types_elem * w_types).mean()

            # ---- Count loss with negative down-weighting ----
            if count_criterion is None:
                # per-sample CE
                ce_per = F.cross_entropy(logits_count, y_count, reduction='none')
            else:
                # ensure criterion returns per-sample if it's focal_ce
                try:
                    ce_per = count_criterion(logits_count, y_count, reduction="none")
                except TypeError:
                    ce_per = count_criterion(logits_count, y_count)  # may be mean already
                    if ce_per.dim() == 0:
                        ce_per = ce_per.expand(x.size(0))
            w_cnt = torch.where(y_any > 0.5,
                                torch.ones_like(ce_per),
                                torch.full_like(ce_per, float(neg_count_weight)))             # <<< NEW
            loss_count = (ce_per * w_cnt).mean()

            # ---- Consistency: any ≈ OR(types) ----
            with torch.no_grad():
                max_type_prob = torch.sigmoid(logits_types).amax(dim=1)  # [B]
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
            if ema is not None:
                ema.update_parameters(model)

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

        if tot > 0:
            pbar.set_postfix(loss=f"{totL/tot:.4f}",
                             f1_types=f"{totF1_types/tot:.3f}",
                             f1_any=f"{totF1_any/tot:.3f}",
                             acc_cnt=f"{totAcc_cnt/tot:.3f}")

    if tot == 0: return (0.0,)*4
    return totL/tot, totF1_types/tot, totF1_any/tot, totAcc_cnt/tot

# ---------- Utils ----------
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
    train_idx, val_idx, test_idx = _enforce_min_coverage(Y[:, :len(CANON_BUG_TYPES)], train_idx, val_idx, test_idx, seed)
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx))

def _enforce_min_coverage(Y_bug, train_idx, val_idx, test_idx, seed=1337):
    rng = np.random.RandomState(seed); C = Y_bug.shape[1]
    splits = {"train": np.array(train_idx), "val": np.array(val_idx), "test": np.array(test_idx)}
    Ys = {k: Y_bug[v] for k, v in splits.items()}
    present_global = Y_bug.sum(axis=0) > 0; labels = np.where(present_global)[0]
    for c in labels:
        for name in ["train", "val", "test"]:
            if splits[name].size == 0:  continue
            if Ys[name][:, c].sum() == 0:
                donor = None
                for cand in ["train", "val", "test"]:
                    if cand == name or splits[cand].size == 0: continue
                    if Ys[cand][:, c].sum() > 1: donor = cand; break
                if donor is None: continue
                donor_indices = np.where(Ys[donor][:, c] == 1)[0]
                dloc = int(donor_indices[rng.randint(0, len(donor_indices))]); d_global = splits[donor][dloc]
                recv_indices = np.where(Ys[name][:, c] == 0)[0]
                if len(recv_indices) == 0: continue
                rloc = int(recv_indices[rng.randint(0, len(recv_indices))]); r_global = splits[name][rloc]
                splits[donor][dloc], splits[name][rloc] = r_global, d_global
                Ys[donor][dloc, :], Ys[name][rloc, :] = Y_bug[r_global], Y_bug[d_global]
    return splits["train"], splits["val"], splits["test"]

def make_count_class_weights(ds_train, mode="inv", beta=0.999):
    count_hist = torch.zeros(4, dtype=torch.float32)
    for y in ds_train.targets_count: count_hist[y.item()] += 1
    N = count_hist.sum().clamp_min(1.0)
    if mode == "none":
        w = torch.ones_like(count_hist)
    elif mode == "inv":
        w = (N / (len(count_hist) * count_hist.clamp_min(1.0)))
    elif mode == "effective":
        eff_num = (1.0 - beta**count_hist) / (1.0 - beta); w = (1.0 / eff_num).float()
    else:
        w = torch.ones_like(count_hist)
    w = w / w.mean()
    return w, count_hist

@torch.no_grad()
def collect_logits_targets(model, loader, device):
    model.eval()
    S_types, Y_types = [], []
    S_any, Y_any = [], []
    for batch in loader:
        if batch is None: continue
        x, y_multi, _y_count, y_any, _paths = batch
        x = x.to(device, non_blocking=True)
        logits_types, _count, logit_any, _ = model(x)
        S_types.append(torch.sigmoid(logits_types).cpu())
        Y_types.append(y_multi.cpu())
        S_any.append(torch.sigmoid(logit_any).cpu())
        Y_any.append(y_any.cpu())
    if not S_types:
        return None, None, None, None
    return torch.cat(S_types,0), torch.cat(Y_types,0), torch.cat(S_any,0), torch.cat(Y_any,0)

def per_class_f1(S, Y, class_names, thr_map=None):
    res = []
    for k, name in enumerate(class_names):
        s = S[:,k].numpy(); y = Y[:,k].numpy().astype(int)
        t = 0.5 if not thr_map else float(thr_map.get(name, 0.5))
        p = (s >= t).astype(int)
        tp = (p & y).sum(); fp=(p & (1-y)).sum(); fn=((1-p) & y).sum()
        prec = tp / (tp+fp+1e-9); rec = tp / (tp+fn+1e-9)
        f1 = 2*prec*rec / (prec+rec+1e-9)
        res.append((name, prec, rec, f1, t))
    return res

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

def make_tb_writer(logdir: str, start_epoch: int) -> SummaryWriter:
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir,
                           purge_step=(start_epoch - 1) if start_epoch > 1 else None,
                           flush_secs=10)
    return writer

def tb_log_scalars(writer: SummaryWriter, scalars: dict, step: int):
    for k, v in scalars.items():
        writer.add_scalar(k, float(v), step)

# ---------- Inference gating ----------
def gate_predictions(type_probs: torch.Tensor,
                     any_prob: torch.Tensor,
                     thr_any: float,
                     thr_map: Dict[str,float],
                     class_names: List[str],
                     count_logits: torch.Tensor = None,
                     count_nudge: bool = False):
    B, C = type_probs.shape
    device = type_probs.device
    thr_vec = torch.tensor([thr_map.get(n,0.5) for n in class_names], device=device, dtype=type_probs.dtype)  # [C]
    pred_any = (any_prob >= thr_any).to(torch.int)
    pred_types = (type_probs >= thr_vec) & (pred_any.unsqueeze(-1)==1)

    pred_count = None
    if count_logits is not None:
        pred_count = count_logits.argmax(dim=1)  # [B]
        if count_nudge:
            k = pred_types.sum(dim=1)
            too_many = (k > pred_count).nonzero(as_tuple=False).squeeze(-1)
            too_few  = (k < pred_count).nonzero(as_tuple=False).squeeze(-1)
            if too_many.numel()>0:
                adj = 0.05; t_adj = thr_vec + adj; sel = too_many
                pred_types[sel] = (type_probs[sel] >= t_adj) & (pred_any[sel].unsqueeze(-1)==1)
            if too_few.numel()>0:
                adj = 0.05; t_adj = (thr_vec - adj).clamp(0.01, 0.99); sel = too_few
                pred_types[sel] = (type_probs[sel] >= t_adj) & (pred_any[sel].unsqueeze(-1)==1)

    return pred_any.to(torch.int), pred_types.to(torch.int), (pred_count if pred_count is not None else None)

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

class EarlyStopper:
    def __init__(self, patience=6, min_delta=0.0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad = 0

    def step(self, value: float) -> bool:
        if self.mode == "max":
            improved = (value > self.best + self.min_delta)
        else:
            improved = (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.num_bad >= self.patience

# ---------- Main ----------
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
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--logdir", type=str, default="runs/bugdetector")
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
    # sampler
    ap.add_argument("--sampler-weighted", action="store_true")
    # LR schedule
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--use-cosine", action="store_true")
    # EMA
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--use-ema", action="store_true")
    # validation-time presence recall target
    ap.add_argument("--presence-recall", type=float, default=None,
                    help="If set (e.g. 0.95), choose smallest τ_any that achieves ≥ recall")
    ap.add_argument("--early-stop-metric", type=str, default="val_types_f1",
                choices=["val_types_f1","f1_types_gated","val_any_f1","val_count_acc","val_loss"],
                help="Which metric to monitor for early stopping")
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.001)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-no-optim", action="store_true")
    ap.add_argument("--resume-reset-lr", type=float, default=None)
    ap.add_argument("--types-loss", type=str, default="bce", choices=["bce", "asl"])
    ap.add_argument("--asl-gamma-pos", type=float, default=0.0)
    ap.add_argument("--asl-gamma-neg", type=float, default=2.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)

    # <<< NEW: negative down-weighting controls
    ap.add_argument("--neg-types-weight", type=float, default=0.25,
                    help="Weight for types loss on negatives (y_any=0). 1.0 = no down-weighting.")
    ap.add_argument("--neg-count-weight", type=float, default=0.25,
                    help="Weight for count loss on negatives (y_any=0). 1.0 = no down-weighting.")

    # W&B tracking
    ap.add_argument("--wandb-project", type=str, default="bug-detection-production",
                    help="W&B project name")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (use only TensorBoard)")

    args=ap.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    
    # Load from my_new_splits.jsonl using "split" field
    manifest_path = args.manifests[0] if isinstance(args.manifests, list) else args.manifests
    print(f"Loading data from: {manifest_path}")
    
    all_records = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                all_records.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line: {e}")
                continue
    
    if not all_records:
        print("ERROR: No records loaded from manifest.", file=sys.stderr)
        sys.exit(1)
    
    # Split based on "split" field in records
    train_recs = [r for r in all_records if r.get("split", "").lower() == "train"]
    val_recs = [r for r in all_records if r.get("split", "").lower() == "val"]
    test_recs = [r for r in all_records if r.get("split", "").lower() == "test"]
    
    print(f"Total clips loaded: {len(all_records)}")
    print(f"Split -> train: {len(train_recs)} | val: {len(val_recs)} | test: {len(test_recs)}")
    
    if len(train_recs) == 0:
        print("ERROR: No training samples found. Check 'split' field in my_new_splits.jsonl", file=sys.stderr)
        sys.exit(1)
    if len(val_recs) == 0:
        print("ERROR: No validation samples found. Check 'split' field in my_new_splits.jsonl", file=sys.stderr)
        sys.exit(1)

    # Datasets
    ds_train=BugClipDataset(data_root, train_recs, split="train", num_frames=16, resize_hw=224)
    ds_val  =BugClipDataset(data_root, val_recs,   split="val",   num_frames=16, resize_hw=224)

    # Count class weights + sampler
    count_weights, count_hist = make_count_class_weights(ds_train, mode=args.count_weighting)
    print("Count histogram (train) [0,1,2,3+]:", count_hist.tolist())
    print("Count class weights     [0,1,2,3+]:", count_weights.tolist())

    # pos_weight for BCE types (kept)
    with torch.no_grad():
        Ytrain = torch.stack(ds_train.targets_multi)
        pos = Ytrain.sum(0).clamp_min(1.0)
        N = torch.tensor(len(ds_train), dtype=torch.float32)
        pos_weight_types = ((N - pos) / pos).float()

    with torch.no_grad():
        any_train = torch.stack(ds_train.targets_any)  # [N]
        pos_any = any_train.sum().clamp_min(1.0)
        N_any = torch.tensor(len(ds_train), dtype=torch.float32)
        pos_weight_any = ((N_any - pos_any) / pos_any).float()

    def bce_with_pos_weight(logits, targets, device, pos_w):
        # NOTE: this path was used in older script;
        # we now do per-element losses directly in run_epoch for weighting.
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w.to(device))

    # Sampler
    train_sampler = None
    if args.sampler_weighted:
        per_sample_w = [count_weights[y.item()].item() for y in ds_train.targets_count]
        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(per_sample_w, num_samples=len(per_sample_w), replacement=True)

    # Loaders
    train_loader=DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_drop_none
    )
    val_loader  =DataLoader(
        ds_val, batch_size=max(1,args.batch_size//2),
        shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_drop_none
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:" , (torch.cuda.get_device_name(0) if device.type=='cuda' else "CPU"))
    model=BugBiLSTM(num_bug_types=len(CANON_BUG_TYPES),pretrained=True).to(device)
    
    with torch.no_grad():
        p = any_train.float().mean().item()                 # prevalence in train
        p = min(max(p, 1e-6), 1 - 1e-6)                    # clamp
        bias = math.log(p / (1 - p))
        nn.init.constant_(model.fc_any.bias, bias)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type=='cuda'))

    # Cosine LR with warmup
    sched = None
    if args.use_cosine:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * max(1, args.epochs)
        warmup_steps = int(steps_per_epoch * max(0.0, args.warmup_epochs))

        eta_min = 1e-5
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

    # Loss fns (keep selection; we will do per-element in run_epoch)
    # bce_multi kept for compatibility but not used directly now for weighting path
    bce_multi = lambda logits, targets: bce_with_pos_weight(logits, targets, device, pos_weight_types)
    cw = count_weights.to(device)

    if args.count_focal:
        def count_criterion(logits, targets, reduction="mean"):
            out = focal_ce(logits, targets, alpha=cw, gamma=args.count_gamma,
                           label_smoothing=args.count_label_smooth, reduction="none")
            if reduction == "none":
                return out
            return out.mean()
    else:
        def count_criterion(logits, targets, reduction="mean"):
            out = F.cross_entropy(logits, targets, weight=cw, label_smoothing=args.count_label_smooth, reduction='none')
            if reduction == "none":
                return out
            return out.mean()

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    start_epoch = 1
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if args.use_ema and ("ema_state" in ckpt):
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}
            print("[Resume] Restored EMA state.")
        if not args.resume_no_optim:
            if "opt_state" in ckpt:
                opt.load_state_dict(ckpt["opt_state"])
                print("[Resume] Restored optimizer state.")
            if ("sched_state" in ckpt) and (sched is not None):
                try:
                    sched.load_state_dict(ckpt["sched_state"])
                    print("[Resume] Restored scheduler state.")
                except Exception:
                    print("[Resume] Scheduler state not compatible; continuing without it.")
            if ("scaler_state" in ckpt) and (scaler is not None):
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                    print("[Resume] Restored GradScaler state.")
                except Exception:
                    print("[Resume] GradScaler state not compatible; continuing without it.")
        if args.resume_reset_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(args.resume_reset_lr)
            print(f"[Resume] LR reset to {args.resume_reset_lr}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] Starting at epoch {start_epoch}")
    # <<< END RESUME HOOK

    writer = make_tb_writer(args.logdir, start_epoch)

    # Initialize Weights & Biases
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{Path(args.save_dir).name}_ep{args.epochs}_lr{args.lr}",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "backbone": "resnet18+bilstm",
                    "loss_types": args.types_loss,
                    "count_focal": args.count_focal,
                    "use_ema": args.use_ema,
                    "use_cosine": args.use_cosine,
                    "presence_recall": args.presence_recall,
                    "early_stop_metric": args.early_stop_metric,
                    "early_stop_patience": args.early_stop_patience,
                    "train_samples": len(train_recs),
                    "val_samples": len(val_recs),
                },
                tags=["fir", "production", "remote"],
                notes=f"Training with {len(train_recs)} train + {len(val_recs)} val samples"
            )
            print(f"✓ W&B initialized: {wandb.run.get_url()}")
            print(f"  Monitor training at: https://wandb.ai")
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            print("Continuing without W&B logging...")
            wandb_run = None
    else:
        print("W&B disabled. Using only TensorBoard.")

    for i, v in enumerate(count_hist.tolist()):
        writer.add_scalar(f"train_count_hist/class_{i}", v, 0)
    for i, wv in enumerate(count_weights.tolist()):
        writer.add_scalar(f"train_count_weight/class_{i}", wv, 0)

    os.makedirs(args.save_dir,exist_ok=True)
    best_f1=-1.0; best_path=None

    # monitor
    monitor_name = args.early_stop_metric
    monitor_mode = "min" if monitor_name == "val_loss" else "max"
    best_monitor = float("inf") if monitor_mode == "min" else -float("inf")
    best_monitor_path = None

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        trL, trF1_types, trF1_any, trAcc_cnt = run_epoch(
            model, train_loader, opt, scaler, device,
            amp=(not args.no_amp), train=True,
            bce_multi=bce_multi, count_criterion=count_criterion,
            count_lambda=args.count_lambda, any_lambda=args.any_lambda,
            cons_lambda=args.cons_lambda, sched=sched, ema=ema, any_pos_weight=pos_weight_any,
            neg_types_weight=args.neg_types_weight, neg_count_weight=args.neg_count_weight  # <<< NEW
        )

        # ---- Swap to EMA weights for validation & saving ----
        raw_sd = None
        if ema is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            print(f"[EMA] Using EMA weights for validation/save at epoch {epoch}.")

        # ---- Tune thresholds on val (EMA active) ----
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
            else:
                thr_map = {name: 0.5 for name in CANON_BUG_TYPES}
            f1_types_gated = gated_types_micro_f1(S_types, Y_types, S_any, tau_any, thr_map, CANON_BUG_TYPES)

        # ---- Standard val (EMA active) ----
        vaL, vaF1_types, vaF1_any, vaAcc_cnt = run_epoch(
            model, val_loader, opt, scaler, device,
            amp=False, train=False,
            bce_multi=bce_multi, count_criterion=count_criterion,
            count_lambda=args.count_lambda, any_lambda=args.any_lambda,
            cons_lambda=args.cons_lambda, any_pos_weight=pos_weight_any,
            neg_types_weight=args.neg_types_weight, neg_count_weight=args.neg_count_weight  # <<< NEW (keeps eval stable)
        )
        dt = time.time() - t0

        # ---- Log ----
        step = epoch
        tb_log_scalars(writer, {
            "loss/train": trL,
            "loss/val":   vaL,
            "f1_types/train": trF1_types,
            "f1_types/val":   vaF1_types,
            "f1_any/train":   trF1_any,
            "f1_any/val":     vaF1_any,
            "acc_count/train": trAcc_cnt,
            "acc_count/val":   vaAcc_cnt,
            "f1_types_gated/val": (0.0 if f1_types_gated is None else f1_types_gated),
            "lr": opt.param_groups[0]["lr"],
        }, step)

        # Log to W&B
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "loss/train": trL,
                "loss/val": vaL,
                "f1_types/train": trF1_types,
                "f1_types/val": vaF1_types,
                "f1_any/train": trF1_any,
                "f1_any/val": vaF1_any,
                "acc_count/train": trAcc_cnt,
                "acc_count/val": vaAcc_cnt,
                "f1_types_gated/val": (0.0 if f1_types_gated is None else f1_types_gated),
                "learning_rate": opt.param_groups[0]["lr"],
            }, step=epoch)

        print(f"[Epoch {epoch:02d}] loss_tr={trL:.4f} loss_va={vaL:.4f} | "
              f"typesF1_tr={trF1_types:.3f} typesF1_va={vaF1_types:.3f} | "
              f"anyF1_tr={trF1_any:.3f} anyF1_va={vaF1_any:.3f} | "
              f"cntAcc_tr={trAcc_cnt:.3f} cntAcc_va={vaAcc_cnt:.3f} ({dt/60:.1f} min)")
        if S_any is not None:
            print(f"  tuned τ_any={thr_any_obj['thr']:.2f} (val prec={thr_any_obj['prec']:.2f}, rec={thr_any_obj['rec']:.2f})")
        if thr_map is not None:
            focus = ["corrupted_texture","geometry_corruption"]
            print("  tuned per-class τ:", {k: round(thr_map[k], 2) for k in focus if k in thr_map})
        if f1_types_gated is not None:
            print(f"  gated_typesF1_va={f1_types_gated:.3f} | τ_any={thr_any_obj['thr']:.2f}")

        # ---- Save-best (EMA active) ----
        metrics_for_stop = {
            "val_types_f1":   float(vaF1_types),
            "f1_types_gated": float(f1_types_gated) if f1_types_gated is not None else (-float("inf")),
            "val_any_f1":     float(vaF1_any),
            "val_count_acc":  float(vaAcc_cnt),
            "val_loss":       float(vaL),
        }
        current = metrics_for_stop[monitor_name]
        min_delta = float(args.early_stop_min_delta)
        is_better = ((current < best_monitor - min_delta) if monitor_mode == "min"
                     else (current > best_monitor + min_delta))
        if is_better:
            best_monitor = current
            best_path = Path(args.save_dir)/(
                f"bugdetector_anygate_best_epoch{epoch:02d}_"
                f"{monitor_name}_{current:.3f}_typesF1{vaF1_types:.3f}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": (sched.state_dict() if sched is not None else None),
                "scaler_state": (scaler.state_dict() if scaler is not None else None),
                "ema_state": (ema.shadow if (ema is not None) else None),
                "val_types_f1": vaF1_types,
                "val_any_f1": vaF1_any,
                "val_count_acc": vaAcc_cnt,
                "val_loss": vaL,
                "config": vars(args),
                "bug_types": CANON_BUG_TYPES,
                "thr_map": thr_map,
                "thr_any": thr_any_obj["thr"],
            }, best_path)
            print(f"  ↳ Saved new best ({monitor_name}): {current:.4f} -> {best_path}")

            # Log best checkpoint to W&B
            if wandb_run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"model-best-{monitor_name}",
                        type="model",
                        description=f"Best model at epoch {epoch} with {monitor_name}={current:.4f}",
                        metadata={
                            "epoch": epoch,
                            "val_types_f1": float(vaF1_types),
                            "val_any_f1": float(vaF1_any),
                            "val_loss": float(vaL),
                        }
                    )
                    artifact.add_file(str(best_path))
                    wandb.log_artifact(artifact)
                    print(f"  ↳ Logged to W&B artifacts")
                except Exception as e:
                    print(f"  Warning: Failed to log artifact to W&B: {e}")
            best_monitor_path = best_path
            if monitor_name == "val_types_f1":
                best_f1 = vaF1_types

        # ---- Restore raw weights to continue training ----
        if raw_sd is not None:
            model.load_state_dict(raw_sd)

        # Save checkpoint every epoch for resume capability
        checkpoint_every = Path(args.save_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "sched_state": (sched.state_dict() if sched is not None else None),
            "scaler_state": (scaler.state_dict() if scaler is not None else None),
            "ema_state": (ema.shadow if (ema is not None) else None),
            "val_types_f1": vaF1_types,
            "val_any_f1": vaF1_any,
            "val_count_acc": vaAcc_cnt,
            "val_loss": vaL,
            "best_monitor": best_monitor,
            "config": vars(args),
        }, checkpoint_every)
        print(f"  ↳ Checkpoint saved: {checkpoint_every.name}")
        
        # Keep only last 3 checkpoints to save space
        checkpoints = sorted(Path(args.save_dir).glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                print(f"  ↳ Removed old checkpoint: {old_ckpt.name}")

    # ========================================================================
    # FINAL TEST EVALUATION (after training completes)
    # ========================================================================
    
    print("")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    
    # Load test records
    test_recs = [r for r in all_records if r.get("split", "").lower() == "test"]
    
    if len(test_recs) > 0:
        print(f"Found {len(test_recs)} test samples")
        
        # Create test dataset
        ds_test = BugClipDataset(data_root, test_recs, split="test", num_frames=16, resize_hw=224)
        test_loader = DataLoader(
            ds_test, batch_size=max(1, args.batch_size//2),
            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_drop_none
        )
        
        # Load best checkpoint for evaluation
        if best_monitor_path and best_monitor_path.exists():
            print(f"Loading best checkpoint: {best_monitor_path}")
            best_ckpt = torch.load(best_monitor_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state"], strict=True)
            print("✓ Best model loaded")
        else:
            print("WARNING: No best checkpoint found, using current model weights")
        
        # Collect logits and targets for test set
        print("Evaluating on test set...")
        S_types_test, Y_types_test, S_any_test, Y_any_test = collect_logits_targets(model, test_loader, device)
        
        # Use tuned thresholds from validation
        if best_monitor_path and best_monitor_path.exists():
            thr_any_test = float(best_ckpt.get("thr_any", 0.5))
            thr_map_test = best_ckpt.get("thr_map", {name: 0.5 for name in CANON_BUG_TYPES})
        else:
            thr_any_test = 0.5
            thr_map_test = {name: 0.5 for name in CANON_BUG_TYPES}
        
        # Run standard evaluation on test
        test_loss, test_f1_types, test_f1_any, test_acc_cnt = run_epoch(
            model, test_loader, opt, scaler, device,
            amp=False, train=False,
            bce_multi=bce_multi, count_criterion=count_criterion,
            count_lambda=args.count_lambda, any_lambda=args.any_lambda,
            cons_lambda=args.cons_lambda, any_pos_weight=pos_weight_any,
            neg_types_weight=args.neg_types_weight, neg_count_weight=args.neg_count_weight
        )
        
        # Calculate gated F1 on test
        test_f1_gated = None
        if (S_types_test is not None) and (S_any_test is not None):
            test_f1_gated = gated_types_micro_f1(
                S_types_test, Y_types_test, S_any_test, 
                thr_any_test, thr_map_test, CANON_BUG_TYPES
            )
        
        # Print test results
        print("")
        print("=" * 70)
        print("TEST SET RESULTS (Final Performance)")
        print("=" * 70)
        print(f"Test Loss:           {test_loss:.4f}")
        print(f"Test Types F1:       {test_f1_types:.4f}")
        print(f"Test Any F1:         {test_f1_any:.4f}")
        print(f"Test Count Accuracy: {test_acc_cnt:.4f}")
        if test_f1_gated is not None:
            print(f"Test Gated Types F1: {test_f1_gated:.4f}")
        print("=" * 70)
        
        # Log test metrics to W&B
        if wandb_run is not None:
            wandb.log({
                "test/loss": test_loss,
                "test/f1_types": test_f1_types,
                "test/f1_any": test_f1_any,
                "test/acc_count": test_acc_cnt,
                "test/f1_types_gated": (0.0 if test_f1_gated is None else test_f1_gated),
            })
            wandb.run.summary["test_types_f1"] = test_f1_types
            wandb.run.summary["test_any_f1"] = test_f1_any
            wandb.run.summary["test_count_acc"] = test_acc_cnt
            wandb.run.summary["test_f1_gated"] = (0.0 if test_f1_gated is None else test_f1_gated)
            print("✓ Test metrics logged to W&B")
        
        # Save test results to file
        test_results = {
            "test_loss": float(test_loss),
            "test_f1_types": float(test_f1_types),
            "test_f1_any": float(test_f1_any),
            "test_count_acc": float(test_acc_cnt),
            "test_f1_gated": float(test_f1_gated) if test_f1_gated is not None else None,
            "thresholds_used": {
                "thr_any": float(thr_any_test),
                "thr_map": {k: float(v) for k, v in thr_map_test.items()}
            }
        }
        
        test_results_path = Path(args.save_dir) / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"✓ Test results saved to: {test_results_path}")
        
        # =====================================================================
        # COMPREHENSIVE METRICS & VISUALIZATIONS
        # =====================================================================
        print("")
        print("=" * 80)
        print("GENERATING COMPREHENSIVE METRICS & VISUALIZATIONS")
        print("=" * 80)
        
        # Create visualizations directory
        viz_dir = Path(args.save_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Collect predictions with gating for detailed analysis
        print("\nCollecting detailed predictions...")
        all_preds_types = []
        all_true_types = []
        all_preds_count = []
        all_true_count = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting predictions"):
                if batch is None:
                    continue
                x, y_types, y_count, y_any, _ = batch
                x = x.to(device)
                
                logits_types, logits_count, logit_any, _ = model(x)
                
                # Apply gating
                probs_types = torch.sigmoid(logits_types).cpu().numpy()
                prob_any = torch.sigmoid(logit_any).cpu().numpy()
                count_preds = torch.argmax(logits_count, dim=1).cpu().numpy()
                
                for i in range(len(probs_types)):
                    if prob_any[i] > thr_any_test:
                        pred_binary = np.array([1 if probs_types[i][j] > thr_map_test.get(CANON_BUG_TYPES[j], 0.5) else 0
                                               for j in range(len(CANON_BUG_TYPES))])
                    else:
                        pred_binary = np.zeros(len(CANON_BUG_TYPES))
                    
                    all_preds_types.append(pred_binary)
                    all_true_types.append(y_types[i].numpy())
                
                all_preds_count.extend(count_preds)
                all_true_count.extend(y_count.cpu().numpy())
        
        all_preds_types = np.array(all_preds_types)
        all_true_types = np.array(all_true_types)
        all_preds_count = np.array(all_preds_count)
        all_true_count = np.array(all_true_count)
        
        print(f"✓ Collected {len(all_preds_types)} predictions")
        
        # 1. Overall multilabel metrics
        print("\n1. Computing overall metrics...")
        overall_metrics = compute_overall_metrics(all_preds_types, all_true_types)
        
        print("\nOVERALL TEST METRICS (Gated):")
        print("-" * 80)
        print(f"  Accuracy:    {overall_metrics['accuracy']:.4f}")
        print(f"  Precision:   {overall_metrics['precision']:.4f}")
        print(f"  Recall:      {overall_metrics['recall']:.4f}")
        print(f"  F1 Score:    {overall_metrics['f1']:.4f}")
        print("-" * 80)
        
        # 2. Per-class bug type metrics
        print("\n2. Computing per-class bug type metrics...")
        per_class_bug_metrics = compute_per_class_metrics(
            torch.sigmoid(S_types_test).numpy(), 
            Y_types_test.numpy(), 
            CANON_BUG_TYPES, 
            thresholds=thr_map_test
        )
        
        print("\nPER-CLASS BUG TYPE METRICS:")
        print("-" * 80)
        print(f"{'Bug Type':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)
        for bug_type in CANON_BUG_TYPES:
            m = per_class_bug_metrics[bug_type]
            print(f"{bug_type:<25} {m['precision']:>11.4f} {m['recall']:>11.4f} {m['f1']:>11.4f} {m['support']:>9}")
        print("-" * 80)
        
        plot_per_class_metrics(
            per_class_bug_metrics,
            viz_dir / "per_class_bug_types.png",
            title="Per-Class Bug Type Metrics (TEST SET)"
        )
        print(f"✓ Saved: per_class_bug_types.png")
        
        # 3. Per-count class metrics
        print("\n3. Computing per-count class metrics...")
        per_count_metrics = compute_count_class_metrics(all_preds_count, all_true_count)
        
        print("\nPER-COUNT CLASS METRICS:")
        print("-" * 80)
        print(f"{'Count Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)
        for count_label in ['0 bugs', '1 bug', '2 bugs', '3 bugs']:
            m = per_count_metrics[count_label]
            print(f"{count_label:<15} {m['precision']:>11.4f} {m['recall']:>11.4f} {m['f1']:>11.4f} {m['support']:>9}")
        print("-" * 80)
        
        plot_count_class_metrics(
            per_count_metrics,
            viz_dir / "per_count_class_metrics.png",
            title="Per-Count Class Metrics (TEST SET)"
        )
        print(f"✓ Saved: per_count_class_metrics.png")
        
        # 4. Confusion matrix for count
        print("\n4. Generating confusion matrix for count...")
        plot_confusion_matrix_count(
            all_true_count, all_preds_count,
            viz_dir / "confusion_matrix_count.png",
            title="Confusion Matrix - Bug Count (TEST SET)"
        )
        print(f"✓ Saved: confusion_matrix_count.png")
        
        # 5. Bug combination metrics
        print("\n5. Computing bug combination metrics...")
        combo_metrics = compute_combo_metrics(
            all_preds_types, all_true_types, CANON_BUG_TYPES, top_k=20
        )
        
        print("\nTOP BUG COMBINATIONS:")
        print("-" * 95)
        print(f"{'Combination':<50} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 95)
        for combo, metrics in sorted(combo_metrics.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"{combo:<50} {metrics['count']:<8} {metrics['precision']:>11.4f} {metrics['recall']:>11.4f} {metrics['f1']:>11.4f}")
        print("-" * 95)
        
        plot_combo_metrics(
            combo_metrics,
            viz_dir / "bug_combination_metrics.png",
            title="Bug Combination Metrics (TEST SET)"
        )
        print(f"✓ Saved: bug_combination_metrics.png")
        
        # 6. Save comprehensive metrics JSON
        comprehensive_metrics = {
            "overall_metrics": overall_metrics,
            "per_class_bug_type_metrics": per_class_bug_metrics,
            "per_count_class_metrics": per_count_metrics,
            "bug_combination_metrics": combo_metrics,
            "basic_test_results": test_results
        }
        
        comprehensive_path = viz_dir / "comprehensive_test_metrics.json"
        with open(comprehensive_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        print(f"✓ Saved: comprehensive_test_metrics.json")
        
        # 7. Log comprehensive metrics to W&B
        if wandb_run is not None:
            # Overall metrics
            wandb.log({
                "test_comprehensive/accuracy": overall_metrics['accuracy'],
                "test_comprehensive/precision": overall_metrics['precision'],
                "test_comprehensive/recall": overall_metrics['recall'],
                "test_comprehensive/f1": overall_metrics['f1'],
            })
            
            # Per-class bug type metrics
            for bug_type in CANON_BUG_TYPES:
                m = per_class_bug_metrics[bug_type]
                wandb.log({
                    f"test_bug_types/{bug_type}/precision": m['precision'],
                    f"test_bug_types/{bug_type}/recall": m['recall'],
                    f"test_bug_types/{bug_type}/f1": m['f1'],
                })
            
            # Per-count class metrics
            for count_label in ['0 bugs', '1 bug', '2 bugs', '3 bugs']:
                m = per_count_metrics[count_label]
                safe_label = count_label.replace(' ', '_')
                wandb.log({
                    f"test_count_classes/{safe_label}/precision": m['precision'],
                    f"test_count_classes/{safe_label}/recall": m['recall'],
                    f"test_count_classes/{safe_label}/f1": m['f1'],
                })
            
            # Upload visualizations
            try:
                wandb.log({
                    "test_viz/per_class_bug_types": wandb.Image(str(viz_dir / "per_class_bug_types.png")),
                    "test_viz/per_count_classes": wandb.Image(str(viz_dir / "per_count_class_metrics.png")),
                    "test_viz/confusion_matrix": wandb.Image(str(viz_dir / "confusion_matrix_count.png")),
                    "test_viz/bug_combinations": wandb.Image(str(viz_dir / "bug_combination_metrics.png")),
                })
                print("✓ Visualizations uploaded to W&B")
            except Exception as e:
                print(f"Warning: Could not upload images to W&B: {e}")
            
            # Update summary
            wandb.run.summary.update({
                "test_accuracy": overall_metrics['accuracy'],
                "test_precision": overall_metrics['precision'],
                "test_recall": overall_metrics['recall'],
                "test_f1": overall_metrics['f1'],
            })
        
        print("")
        print("=" * 80)
        print("COMPREHENSIVE TEST EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\n📊 All results saved to: {viz_dir}")
        print(f"   ✓ per_class_bug_types.png")
        print(f"   ✓ per_count_class_metrics.png")
        print(f"   ✓ confusion_matrix_count.png")
        print(f"   ✓ bug_combination_metrics.png")
        print(f"   ✓ comprehensive_test_metrics.json")
        print("")
        print("=" * 80)
        
    else:
        print("No test samples found in split file. Skipping test evaluation.")
        print("If you want test evaluation, add samples with 'split': 'test' to my_new_splits.jsonl")
    
    print("=" * 70)

    writer.close()

    # Finish W&B
    if wandb_run is not None:
        wandb.run.summary["best_val_f1"] = best_f1
        wandb.run.summary["best_checkpoint"] = str(best_path) if best_path else "N/A"
        wandb.run.summary["total_epochs_trained"] = epoch
        wandb.finish()
        print("✓ W&B run finished")
    print("Done. Best val types micro-F1:", f"{best_f1:.3f}" if best_f1>=0 else "N/A")
    if best_path: print("Best checkpoint:", str(best_path))

if __name__=="__main__": main()