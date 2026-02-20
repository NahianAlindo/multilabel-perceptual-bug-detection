#!/usr/bin/env python3
# ==============================================================================
# ResNet18+GRU MODEL - Complete Training + Validation + Test Pipeline
# ==============================================================================
# Alternative temporal architecture: GRU instead of BiLSTM
# Purpose: Compare GRU vs BiLSTM for temporal modeling
# - Trains on "train" split (60 epochs)
# - Validates on "val" split (every epoch)
# - Tests on "test" split (automatic after training)
# - Saves 4 visualization charts with 4 decimal precision
# ==============================================================================

import os, sys, json, math, time, argparse, glob, gzip, io, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import amp as torch_amp

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARNING] matplotlib/seaborn not available. Visualizations will be skipped.")

# Sklearn for metrics
try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] sklearn not available. Some metrics may be unavailable.")

# Weights & Biases (with timeout handling)
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[WARNING] wandb not installed. W&B logging disabled.")

# Video loading backends
_BACKENDS = {}
try:
    import decord
    decord.bridge.set_bridge('torch')
    _BACKENDS['decord'] = True
except Exception:
    _BACKENDS['decord'] = False
try:
    from torchvision.io import read_video
    _BACKENDS['tvio'] = True
except Exception:
    _BACKENDS['tvio'] = False
try:
    import cv2
    _BACKENDS['opencv'] = True
except Exception:
    _BACKENDS['opencv'] = False

# ==============================================================================
# CONSTANTS
# ==============================================================================

CANON_BUG_TYPES = ["z-clipping", "corrupted_texture", "geometry_corruption", "z-fighting", "boundary_hole"]
_ALIAS = {
    "z_clipping": "z-clipping", "z_fighting": "z-fighting", "boundary_hole": "boundary_hole",
    "corrupted_texture": "corrupted_texture", "geometry_corruption": "geometry_corruption",
}
BUG2IDX = {b: i for i, b in enumerate(CANON_BUG_TYPES)}

def norm_bug_type(s: str) -> str:
    s2 = s.strip().lower().replace(" ", "_").replace("-", "_")
    canon = _ALIAS.get(s2, s2)
    if canon in ("z_clipping", "z-clipping"):
        return "z-clipping"
    if canon in ("z_fighting", "z-fighting"):
        return "z-fighting"
    return canon

# ==============================================================================
# JSONL UTILITIES
# ==============================================================================

def open_maybe_gzip(path: Path):
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if str(path).endswith(".gz") else path.open("r", encoding="utf-8")

def discover_manifests(root: Path, pattern: str) -> List[Path]:
    if root.is_file():
        return [root]
    paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=True)]
    return sorted(p for p in paths if p.suffix in (".jsonl", ".json", ".gz"))

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

# ==============================================================================
# VIDEO LOADING (with multiple backend fallbacks)
# ==============================================================================

def load_video_tensor(path: str, num_frames=16) -> torch.Tensor:
    """Load video and return [T,H,W,C] tensor in uint8."""
    path = str(path)
    
    # Try torchvision.io first
    if _BACKENDS.get('tvio'):
        try:
            video, _, _ = read_video(path, pts_unit='sec')
            if video.dtype != torch.uint8:
                video = torch.clamp(video, 0, 255).to(torch.uint8)
            if len(video) >= num_frames:
                idx = torch.linspace(0, len(video) - 1, num_frames).long()
                return video[idx]
        except Exception:
            pass
    
    # Try decord
    if _BACKENDS.get('decord'):
        try:
            import decord
            vr = decord.VideoReader(path)
            total = len(vr)
            if total >= num_frames:
                idx = torch.linspace(0, total - 1, num_frames).long().tolist()
                frames = vr.get_batch(idx)
                if isinstance(frames, torch.Tensor):
                    return frames
                return torch.from_numpy(frames.asnumpy()).to(torch.uint8)
        except Exception:
            pass
    
    # Try OpenCV
    if _BACKENDS.get('opencv'):
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if len(frames_list) >= num_frames:
                idx = np.linspace(0, len(frames_list) - 1, num_frames, dtype=int)
                selected = [frames_list[i] for i in idx]
                arr = np.stack(selected, axis=0)
                return torch.from_numpy(arr).to(torch.uint8)
        except Exception:
            pass
    
    return None

# ==============================================================================
# DATASET
# ==============================================================================

class BugClipDataset(Dataset):
    def __init__(self, data_root: Path, records: list, split: str = "train",
                 num_frames: int = 16, resize_hw: int = 224):
        self.data_root = Path(data_root)
        self.records = records
        self.split = split
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        
        # Prepare targets
        self.targets_multi = []
        self.targets_count = []
        self.targets_any = []
        
        for rec in self.records:
            bugs_raw = rec.get("bug_types", [])
            if isinstance(bugs_raw, str):
                bugs_raw = [bugs_raw]
            bug_types_norm = [norm_bug_type(b) for b in bugs_raw if norm_bug_type(b) in CANON_BUG_TYPES]
            
            y_multi = np.zeros(len(CANON_BUG_TYPES), dtype=np.float32)
            for bt in bug_types_norm:
                if bt in BUG2IDX:
                    y_multi[BUG2IDX[bt]] = 1.0
            
            self.targets_multi.append(y_multi)
            self.targets_count.append(min(len(bug_types_norm), 3))
            self.targets_any.append(1.0 if len(bug_types_norm) > 0 else 0.0)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        try:
            rec = self.records[idx]
            relpath = rec.get("relpath") or rec.get("clip_path")
            
            # Fix Windows paths (backslash to forward slash)
            relpath = str(relpath).replace("\\", "/")
            
            clip_path = self.data_root / relpath
            
            # Load video [T,H,W,C]
            frames = load_video_tensor(str(clip_path), self.num_frames)
            if frames is None or frames.shape[0] < self.num_frames:
                return None
            
            # Transform each frame
            tfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_hw, self.resize_hw)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            imgs = [tfm(frames[t].numpy()) for t in range(self.num_frames)]
            x = torch.stack(imgs, dim=0)  # [T,C,H,W]
            
            y_types = torch.tensor(self.targets_multi[idx], dtype=torch.float32)
            y_count = torch.tensor(self.targets_count[idx], dtype=torch.long)
            y_any = torch.tensor(self.targets_any[idx], dtype=torch.float32)
            
            return x, y_types, y_count, y_any
        except Exception as e:
            return None

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# MODEL ARCHITECTURE - ResNet18 + GRU
# ==============================================================================

class ResNet18GRUModel(nn.Module):
    """ResNet18 backbone with GRU for temporal modeling."""
    def __init__(self, num_bug_types=5, hidden_dim=256, num_layers=1, dropout=0.5, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        try:
            enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            enc = resnet18(weights=None)
        
        # Remove FC layer, keep feature extractor
        self.encoder = nn.Sequential(*list(enc.children())[:-1])  # Output: [B, 512, 1, 1]
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # Unidirectional GRU (vs BiLSTM)
        )
        
        # Prediction heads
        self.head_types = nn.Linear(hidden_dim, num_bug_types)
        self.head_count = nn.Linear(hidden_dim, 4)  # 0, 1, 2, 3+
        self.head_any = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, frames):
        # frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        
        # Encode each frame
        x = frames.view(B * T, C, H, W)
        feats = self.encoder(x)  # [B*T, 512, 1, 1]
        feats = feats.view(B, T, 512)  # [B, T, 512]
        
        # GRU temporal modeling
        gru_out, _ = self.gru(feats)  # [B, T, hidden_dim]
        
        # Use last time step
        temporal_feats = gru_out[:, -1, :]  # [B, hidden_dim]
        temporal_feats = self.dropout(temporal_feats)
        
        # Predictions
        logits_types = self.head_types(temporal_feats)
        logits_count = self.head_count(temporal_feats)
        logits_any = self.head_any(temporal_feats).squeeze(-1)
        
        return logits_types, logits_count, logits_any

# ==============================================================================
# LOSSES
# ==============================================================================

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multilabel classification."""
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Positive samples
        probs_pos = probs
        if self.clip is not None and self.clip > 0:
            probs_pos = probs_pos + self.clip
            probs_pos = torch.clamp(probs_pos, max=1.0)
        loss_pos = -targets * torch.log(probs_pos) * ((1 - probs_pos) ** self.gamma_pos)
        
        # Negative samples
        probs_neg = 1 - probs
        loss_neg = -(1 - targets) * torch.log(probs_neg) * (probs ** self.gamma_neg)
        
        loss = loss_pos + loss_neg
        return loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        probs = torch.exp(-ce_loss)
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def compute_multilabel_metrics(preds, targets):
    """Compute precision, recall, F1 for multilabel classification."""
    if SKLEARN_AVAILABLE:
        prec, rec, f1, _ = precision_recall_fscore_support(
            targets, preds, average='micro', zero_division=0
        )
        return {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
    else:
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        return {'precision': prec, 'recall': rec, 'f1': f1}

def compute_per_class_metrics(probs, targets, class_names):
    """Compute per-class precision, recall, F1."""
    metrics = {}
    preds = (probs > 0.5).astype(int)
    
    for idx, name in enumerate(class_names):
        y_true = targets[:, idx]
        y_pred = preds[:, idx]
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics[name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
    
    return metrics

def compute_count_class_metrics(preds, targets):
    """Compute metrics for each bug count class (0, 1, 2, 3+)."""
    metrics = {}
    
    for count in range(4):
        mask_true = (targets == count)
        mask_pred = (preds == count)
        
        tp = (mask_pred & mask_true).sum()
        fp = (mask_pred & ~mask_true).sum()
        fn = (~mask_pred & mask_true).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics[f'{count}_bugs'] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'support': int(mask_true.sum())
        }
    
    return metrics

def compute_combo_metrics(probs, targets, class_names, top_k=20):
    """Compute metrics for top-K bug combinations."""
    preds = (probs > 0.5).astype(int)
    
    # Find all unique combinations
    combos = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for i in range(len(targets)):
        true_bugs = tuple(idx for idx in range(len(class_names)) if targets[i, idx] == 1)
        pred_bugs = tuple(idx for idx in range(len(class_names)) if preds[i, idx] == 1)
        
        if true_bugs == pred_bugs and len(true_bugs) > 0:
            combos[true_bugs]['tp'] += 1
        elif len(pred_bugs) > 0:
            combos[pred_bugs]['fp'] += 1
        if len(true_bugs) > 0 and true_bugs != pred_bugs:
            combos[true_bugs]['fn'] += 1
    
    # Calculate metrics for each combo
    combo_results = []
    for combo, counts in combos.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        total = tp + fp + fn
        
        if total > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            bug_names = [class_names[idx] for idx in combo]
            combo_results.append({
                'combo': '+'.join(bug_names),
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'count': total
            })
    
    # Sort by count and return top-K
    combo_results.sort(key=lambda x: x['count'], reverse=True)
    return combo_results[:top_k]

# ==============================================================================
# VISUALIZATION (4 charts with 4 decimal precision)
# ==============================================================================

def create_test_visualizations(test_results, per_class_metrics, per_count_metrics,
                               combo_metrics, confusion_mat, save_dir):
    """Create 4 publication-ready bar charts with 4 decimal precision."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping charts.")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    
    # Chart 1: Per-Class Bug Type Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    bug_names = list(per_class_metrics.keys())
    x = np.arange(len(bug_names))
    width = 0.25
    
    precisions = [per_class_metrics[b]['precision'] for b in bug_names]
    recalls = [per_class_metrics[b]['recall'] for b in bug_names]
    f1s = [per_class_metrics[b]['f1'] for b in bug_names]
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#e74c3c')
    
    # Add value labels with 4 decimals
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Bug Type')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance: Per-Class Bug Type Metrics (ResNet18+GRU Model)')
    ax.set_xticks(x)
    ax.set_xticklabels(bug_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_bug_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Per-Count Class Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    count_names = list(per_count_metrics.keys())
    x = np.arange(len(count_names))
    
    precisions = [per_count_metrics[c]['precision'] for c in count_names]
    recalls = [per_count_metrics[c]['recall'] for c in count_names]
    f1s = [per_count_metrics[c]['f1'] for c in count_names]
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#9b59b6')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#f39c12')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#1abc9c')
    
    # Add value labels with 4 decimals
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Bug Count')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance: Per-Count Class Metrics (ResNet18+GRU Model)')
    ax.set_xticks(x)
    ax.set_xticklabels(count_names, rotation=0)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_dir / 'per_count_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Confusion Matrix (Count Prediction)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2', '3+'],
                yticklabels=['0', '1', '2', '3+'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    ax.set_xlabel('Predicted Bug Count')
    ax.set_ylabel('True Bug Count')
    ax.set_title('Test Set: Bug Count Confusion Matrix (ResNet18+GRU Model)')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 4: Top Bug Combinations
    if len(combo_metrics) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        combo_names = [c['combo'] for c in combo_metrics]
        x = np.arange(len(combo_names))
        
        precisions = [c['precision'] for c in combo_metrics]
        recalls = [c['recall'] for c in combo_metrics]
        f1s = [c['f1'] for c in combo_metrics]
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#e67e22')
        bars2 = ax.bar(x, recalls, width, label='Recall', color='#16a085')
        bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#c0392b')
        
        # Add value labels with 4 decimals
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Bug Combination')
        ax.set_ylabel('Score')
        ax.set_title('Test Set: Top 20 Bug Combination Performance (ResNet18+GRU Model)')
        ax.set_xticks(x)
        ax.set_xticklabels(combo_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(save_dir / 'bug_combination_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 4 visualization charts saved to: {save_dir}")

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def run_epoch(model, loader, optimizer, scaler, device, amp_enabled=True, train=True,
              criterion_types=None, criterion_count=None, criterion_any=None):
    """Run one epoch of training or evaluation."""
    model.train() if train else model.eval()
    
    total_loss = 0.0
    all_preds_types = []
    all_true_types = []
    all_preds_count = []
    all_true_count = []
    
    pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'}", leave=False)
    
    for batch in pbar:
        if batch is None:
            continue
        
        x, y_types, y_count, y_any = batch
        x = x.to(device)
        y_types = y_types.to(device)
        y_count = y_count.to(device)
        y_any = y_any.to(device)
        
        if train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(train):
            if amp_enabled and train:
                with torch_amp.autocast('cuda'):
                    logits_types, logits_count, logits_any = model(x)
                    loss_types = criterion_types(logits_types, y_types)
                    loss_count = criterion_count(logits_count, y_count)
                    loss_any = F.binary_cross_entropy_with_logits(logits_any, y_any)
                    loss = loss_types + loss_count + loss_any
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits_types, logits_count, logits_any = model(x)
                loss_types = criterion_types(logits_types, y_types)
                loss_count = criterion_count(logits_count, y_count)
                loss_any = F.binary_cross_entropy_with_logits(logits_any, y_any)
                loss = loss_types + loss_count + loss_any
                
                if train:
                    loss.backward()
                    optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        # Collect predictions
        preds_types = (torch.sigmoid(logits_types) > 0.5).cpu().numpy()
        preds_count = logits_count.argmax(dim=1).cpu().numpy()
        
        all_preds_types.append(preds_types)
        all_true_types.append(y_types.cpu().numpy())
        all_preds_count.append(preds_count)
        all_true_count.append(y_count.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Aggregate
    all_preds_types = np.concatenate(all_preds_types, axis=0)
    all_true_types = np.concatenate(all_true_types, axis=0)
    all_preds_count = np.concatenate(all_preds_count, axis=0)
    all_true_count = np.concatenate(all_true_count, axis=0)
    
    # Metrics
    avg_loss = total_loss / len(loader.dataset)
    metrics_types = compute_multilabel_metrics(all_preds_types, all_true_types)
    count_acc = (all_preds_count == all_true_count).mean()
    
    return avg_loss, metrics_types['f1'], count_acc, all_preds_types, all_true_types, all_preds_count, all_true_count

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global WANDB_AVAILABLE  # Allow modification in main()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--manifests', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--wandb-project', type=str, default='resnet18-gru-bug-detection')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("ResNet18+GRU MODEL - BUG DETECTION TRAINING")
    print(f"{'='*80}\n")
    
    # W&B with timeout handling
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                settings=wandb.Settings(init_timeout=300)
            )
            print(f"✓ W&B initialized: {wandb.run.get_url()}")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            print("   Continuing with TensorBoard only...")
            WANDB_AVAILABLE = False
            wandb_run = None
    else:
        print("W&B disabled. Using only TensorBoard.")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"✓ TensorBoard: {args.logdir}")
    
    # Load data
    print(f"\nLoading data from: {args.manifests}")
    manifest_paths = discover_manifests(Path(args.data_root), args.manifests)
    all_records = load_all_records(manifest_paths)
    print(f"Total clips loaded: {len(all_records)}")
    
    train_recs = [r for r in all_records if r.get("split", "").lower() == "train"]
    val_recs = [r for r in all_records if r.get("split", "").lower() == "val"]
    test_recs = [r for r in all_records if r.get("split", "").lower() == "test"]
    
    print(f"Split -> train: {len(train_recs)} | val: {len(val_recs)} | test: {len(test_recs)}")
    
    # Datasets
    ds_train = BugClipDataset(Path(args.data_root), train_recs, "train", 16, 224)
    ds_val = BugClipDataset(Path(args.data_root), val_recs, "val", 16, 224)
    
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_drop_none)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           drop_last=False, collate_fn=collate_drop_none)
    
    # Model
    model = ResNet18GRUModel(num_bug_types=len(CANON_BUG_TYPES), hidden_dim=256, num_layers=1, dropout=0.5).to(device)
    print(f"✓ Model: ResNet18+GRU on {device}")
    
    # Losses
    criterion_types = AsymmetricLoss(gamma_pos=0.0, gamma_neg=2.0, clip=0.05)
    criterion_count = FocalLoss(gamma=1.5)
    criterion_any = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # AMP
    scaler = torch_amp.GradScaler('cuda', enabled=(not args.no_amp and device.type == 'cuda'))
    
    # Resume
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"\nLoading checkpoint: {args.resume}")
        try:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['opt_state'])
            if scheduler and 'sched_state' in ckpt and ckpt['sched_state']:
                scheduler.load_state_dict(ckpt['sched_state'])
            if scaler and 'scaler_state' in ckpt and ckpt['scaler_state']:
                scaler.load_state_dict(ckpt['scaler_state'])
            if ema and 'ema_state' in ckpt and ckpt['ema_state']:
                ema.shadow = ckpt['ema_state']
            start_epoch = ckpt['epoch'] + 1
            print(f"✓ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("   Starting fresh from epoch 0")
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    best_val_f1 = -1.0
    best_path = None
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_f1, train_acc, _, _, _, _ = run_epoch(
            model, train_loader, optimizer, scaler, device,
            amp_enabled=(not args.no_amp), train=True,
            criterion_types=criterion_types,
            criterion_count=criterion_count,
            criterion_any=criterion_any
        )
        
        # Update EMA
        if ema:
            ema.update(model)
        
        # Validation
        if ema:
            ema.apply_shadow(model)
        
        val_loss, val_f1, val_acc, _, _, _, _ = run_epoch(
            model, val_loader, optimizer, scaler, device,
            amp_enabled=False, train=False,
            criterion_types=criterion_types,
            criterion_count=criterion_count,
            criterion_any=criterion_any
        )
        
        if ema:
            ema.restore(model)
        
        # Scheduler
        if scheduler:
            scheduler.step()
        
        # Print
        print(f"\nTrain: loss={train_loss:.4f} | f1={train_f1:.4f} | count_acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f} | f1={val_f1:.4f} | count_acc={val_acc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/f1_types', train_f1, epoch)
        writer.add_scalar('train/count_acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/f1_types', val_f1, epoch)
        writer.add_scalar('val/count_acc', val_acc, epoch)
        
        # Log to W&B
        if wandb_run:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/f1_types': train_f1,
                'train/count_acc': train_acc,
                'val/loss': val_loss,
                'val/f1_types': val_f1,
                'val/count_acc': val_acc,
            })
        
        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = Path(args.save_dir) / f"best_epoch_{epoch:03d}_f1_{val_f1:.4f}.pt"
            
            # Save with EMA weights
            if ema:
                ema.apply_shadow(model)
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
            }, best_path)
            
            if ema:
                ema.restore(model)
            
            print(f"✓ New best model saved: {best_path.name}")
        
        # Save checkpoint
        ckpt_path = Path(args.save_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict() if scheduler else None,
            'scaler_state': scaler.state_dict() if scaler else None,
            'ema_state': ema.shadow if ema else None,
            'val_f1': val_f1,
        }, ckpt_path)
        print(f"✓ Checkpoint saved: {ckpt_path.name}")
        
        # Keep last 3
        checkpoints = sorted(Path(args.save_dir).glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old in checkpoints[:-3]:
                old.unlink()
    
    # ==============================================================================
    # TEST EVALUATION
    # ==============================================================================
    
    print(f"\n{'='*80}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*80}\n")
    
    if len(test_recs) > 0:
        print(f"Found {len(test_recs)} test samples")
        
        ds_test = BugClipDataset(Path(args.data_root), test_recs, "test", 16, 224)
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                drop_last=False, collate_fn=collate_drop_none)
        
        # Load best model
        if best_path and best_path.exists():
            print(f"Loading best model: {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            print("✓ Best model loaded")
        
        # Evaluate
        test_loss, test_f1, test_acc, test_preds_types, test_true_types, test_preds_count, test_true_count = run_epoch(
            model, test_loader, optimizer, scaler, device,
            amp_enabled=False, train=False,
            criterion_types=criterion_types,
            criterion_count=criterion_count,
            criterion_any=criterion_any
        )
        
        # Compute detailed metrics
        test_probs_types = test_preds_types.astype(float)  # Already binary predictions
        per_class_metrics = compute_per_class_metrics(test_probs_types, test_true_types, CANON_BUG_TYPES)
        per_count_metrics = compute_count_class_metrics(test_preds_count, test_true_count)
        combo_metrics = compute_combo_metrics(test_probs_types, test_true_types, CANON_BUG_TYPES, top_k=20)
        
        if SKLEARN_AVAILABLE:
            confusion_mat = confusion_matrix(test_true_count, test_preds_count, labels=[0, 1, 2, 3])
        else:
            confusion_mat = np.zeros((4, 4), dtype=int)
            for true, pred in zip(test_true_count, test_preds_count):
                confusion_mat[true, pred] += 1
        
        # Print results
        print(f"\n{'='*80}")
        print("TEST SET RESULTS")
        print(f"{'='*80}")
        print(f"Test Loss:        {test_loss:.4f}")
        print(f"Test F1 (Types):  {test_f1:.4f}")
        print(f"Test Acc (Count): {test_acc:.4f}")
        print(f"{'='*80}\n")
        
        # Save results
        test_results = {
            'test_loss': float(test_loss),
            'test_f1_types': float(test_f1),
            'test_acc_count': float(test_acc),
            'per_class_metrics': {k: {kk: float(vv) for kk, vv in v.items()} 
                                 for k, v in per_class_metrics.items()},
            'per_count_metrics': {k: {kk: float(vv) for kk, vv in v.items()} 
                                 for k, v in per_count_metrics.items()},
            'combo_metrics': [{k: (float(v) if isinstance(v, (int, float)) else v) 
                              for k, v in c.items()} for c in combo_metrics],
        }
        
        results_path = Path(args.save_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"✓ Test results saved: {results_path}")
        
        # Create visualizations
        viz_dir = Path(args.save_dir) / 'visualizations'
        create_test_visualizations(
            test_results, per_class_metrics, per_count_metrics,
            combo_metrics, confusion_mat, viz_dir
        )
        
        # Log to W&B
        if wandb_run:
            wandb.log({
                'test/loss': test_loss,
                'test/f1_types': test_f1,
                'test/acc_count': test_acc,
            })
            
            # Upload visualizations
            if VISUALIZATION_AVAILABLE and viz_dir.exists():
                for img_path in viz_dir.glob('*.png'):
                    wandb.log({f'test_viz/{img_path.stem}': wandb.Image(str(img_path))})
            
            print("✓ Test metrics logged to W&B")
    
    else:
        print("No test samples found. Skipping test evaluation.")
    
    writer.close()
    if wandb_run:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    if best_path:
        print(f"Best checkpoint: {best_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
