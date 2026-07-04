#!/usr/bin/env python3
"""
RESNET18-ONLY ABLATION - COMPLETE TRAINING SCRIPT
==================================================
Architecture: ResNet18 → Average Pooling → Linear Heads (NO BiLSTM)
Purpose: Prove temporal modeling (BiLSTM) is essential

Complete Features:
- Train/Val/Test automatic
- ALL metrics: accuracy, loss, precision, recall, F1
- Per-class metrics for bug types
- Per-count metrics for 0,1,2,3 bugs
- Bug combination metrics
- Confusion matrices
- W&B + TensorBoard logging (COMPLETE PARITY)
- 4 visualization charts saved
"""

import os, sys, json, math, time, argparse, glob, io, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Video backends
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

# Bug types
CANON_BUG_TYPES = ["z-clipping","corrupted_texture","geometry_corruption","z-fighting","boundary_hole"]

def norm_bug_type(s):
    s = s.strip().lower().replace(" ","_").replace("-","_")
    if "clipping" in s: return "z-clipping"
    if "fighting" in s: return "z-fighting"
    return s

# Video loading with improved error handling
_VIDEO_LOAD_FAILURES = 0
_VIDEO_LOAD_SUCCESS = 0

def load_video_cv2(path, num_frames=16):
    global _VIDEO_LOAD_FAILURES, _VIDEO_LOAD_SUCCESS
    
    if not CV2_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            _VIDEO_LOAD_FAILURES += 1
            if _VIDEO_LOAD_FAILURES <= 3:  # Print first 3 failures
                print(f"⚠️  Failed to open video: {path}")
            return None
        
        frames = []
        frame_count = 0
        max_frames = 1000  # Safety limit
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        
        cap.release()
        
        if len(frames) >= num_frames:
            indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
            selected = [frames[i] for i in indices]
            tensor = torch.from_numpy(np.stack(selected)).permute(0,3,1,2)
            _VIDEO_LOAD_SUCCESS += 1
            return tensor.to(torch.uint8)
        else:
            _VIDEO_LOAD_FAILURES += 1
            if _VIDEO_LOAD_FAILURES <= 3:
                print(f"⚠️  Video too short ({len(frames)} frames): {path}")
            return None
            
    except Exception as e:
        _VIDEO_LOAD_FAILURES += 1
        if _VIDEO_LOAD_FAILURES <= 3:
            print(f"⚠️  Video loading exception: {e}")
        return None

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Dataset
class BugClipDataset:
    def __init__(self, data_root, records, split="train", num_frames=16, resize_hw=224):
        self.data_root = Path(data_root)
        self.records = records
        self.split = split
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        
        self.targets_multi = []
        self.targets_count = []
        self.targets_any = []
        
        for rec in records:
            bug_types = rec.get("bug_types", [])
            bug_types_norm = [norm_bug_type(bt) for bt in bug_types]
            
            y_multi = torch.zeros(5, dtype=torch.float32)
            for bt in bug_types_norm:
                if bt in CANON_BUG_TYPES:
                    y_multi[CANON_BUG_TYPES.index(bt)] = 1.0
            
            self.targets_multi.append(y_multi)
            self.targets_count.append(min(len(bug_types_norm), 3))
            self.targets_any.append(1.0 if len(bug_types_norm) > 0 else 0.0)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        try:
            rec = self.records[idx]
            relpath = rec.get("relpath") or rec.get("clip_path")
            
            # Fix Windows backslashes to Linux forward slashes
            relpath = str(relpath).replace("\\", "/")
            
            clip_path = self.data_root / relpath
            
            frames = load_video_cv2(str(clip_path), self.num_frames)
            if frames is None or frames.shape[0] < self.num_frames:
                return None
            
            tfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_hw, self.resize_hw)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
            
            imgs = [tfm(frames[t].numpy()) for t in range(self.num_frames)]
            x = torch.stack(imgs, dim=0)
            
            return x, self.targets_multi[idx], self.targets_count[idx], self.targets_any[idx], str(clip_path)
        except:
            return None

# Model: ResNet18-only (NO BiLSTM)
class ResNet18OnlyModel(nn.Module):
    """
    Ablation model: ResNet18 → Average Pooling → Heads
    NO temporal modeling (BiLSTM)
    """
    def __init__(self, num_bug_types=5, pretrained=True):
        super().__init__()
        try:
            enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except:
            enc = resnet18(weights=None)
        
        self.cnn = nn.Sequential(*list(enc.children())[:-1])
        self.feat_dim = 512
        
        # Heads (same as main model)
        self.fc_types = nn.Linear(512, num_bug_types)
        self.fc_count = nn.Linear(512, 4)
        self.fc_any = nn.Linear(512, 1)
    
    def forward(self, frames):
        # frames: [B, T, 3, H, W]
        B, T, C, H, W = frames.shape
        x = frames.reshape(B*T, C, H, W)
        
        # Extract features per frame
        f = self.cnn(x).flatten(1)  # [B*T, 512]
        f = f.reshape(B, T, self.feat_dim)  # [B, T, 512]
        
        # Average pooling across time (NO BiLSTM!)
        z = f.mean(dim=1)  # [B, 512]
        
        # Heads
        logits_types = self.fc_types(z)
        logits_count = self.fc_count(z)
        logit_any = self.fc_any(z).squeeze(-1)
        
        return logits_types, logits_count, logit_any

# Metrics
def compute_multilabel_metrics(y_pred, y_true):
    """Overall accuracy, precision, recall, F1"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def compute_per_class_metrics(y_pred_probs, y_true, class_names, thresholds=0.5):
    """Per-class precision, recall, F1"""
    if isinstance(thresholds, (int, float)):
        thresholds = {name: float(thresholds) for name in class_names}
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        thr = thresholds.get(class_name, 0.5) if isinstance(thresholds, dict) else thresholds
        y_pred_binary = (y_pred_probs[:, i] >= thr).astype(int)
        y_true_binary = y_true[:, i].astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)
        
        metrics[class_name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_true_binary.sum())
        }
    
    return metrics

def compute_count_class_metrics(y_pred, y_true):
    """Per-count metrics (0,1,2,3 bugs)"""
    count_labels = ['0_bugs', '1_bug', '2_bugs', '3_bugs']
    metrics = {}
    
    for i, label in enumerate(count_labels):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)
        
        metrics[label] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_true_binary.sum())
        }
    
    return metrics

def compute_combo_metrics(y_pred, y_true, class_names, top_k=20):
    """Bug combination metrics"""
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
        acc = accuracy_score(y_true_binary, y_pred_binary)
        
        combo_metrics[combo] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'count': int(combo_counts[combo])
        }
    
    return combo_metrics

# Visualizations
def plot_per_class_metrics(metrics, save_path, title="Per-class metrics"):
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
    ax.set_xticklabels(class_names, rotation=0)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix - Count"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['0 bugs','1 bug','2 bugs','3 bugs'],
                yticklabels=['0 bugs','1 bug','2 bugs','3 bugs'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_combo_metrics(combo_metrics, save_path, title="Bug Combination Metrics"):
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

# Loss functions
def asl_loss(logits, targets):
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos
    xs_neg = (xs_neg + 0.05).clamp(max=1.0)
    log_pos = torch.log(xs_pos.clamp_min(1e-8))
    log_neg = torch.log(xs_neg.clamp_min(1e-8))
    loss_pos = targets * log_pos
    loss_neg = (1 - targets) * (xs_pos ** 2.0) * log_neg
    loss = -(loss_pos + loss_neg)
    return loss.mean()

def focal_ce(logits, targets):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** 1.5) * ce
    return loss.mean()

# Training loop
def run_epoch(model, loader, opt, scaler, device, train=True, epoch=0, 
              writer=None, wandb_run=None, prefix='train'):
    model.train() if train else model.eval()
    
    total_loss = 0
    all_preds_types = []
    all_true_types = []
    all_probs_types = []
    all_preds_count = []
    all_true_count = []
    
    pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} Epoch {epoch}")
    
    for batch in pbar:
        if batch is None:
            continue
        
        x, y_types, y_count, y_any, _ = batch
        x = x.to(device)
        y_types = y_types.to(device)
        y_count = y_count.to(device).long()
        y_any = y_any.to(device)
        
        if train:
            opt.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits_types, logits_count, logit_any = model(x)
            
            loss_types = asl_loss(logits_types, y_types)
            loss_count = focal_ce(logits_count, y_count)
            loss_any = F.binary_cross_entropy_with_logits(logit_any, y_any)
            
            loss = loss_types + loss_count + loss_any
        
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            probs_types = torch.sigmoid(logits_types).cpu().numpy()
            preds_types = (probs_types > 0.5).astype(int)
            preds_count = torch.argmax(logits_count, dim=1).cpu().numpy()
            
            all_probs_types.append(probs_types)
            all_preds_types.append(preds_types)
            all_true_types.append(y_types.cpu().numpy())
            all_preds_count.append(preds_count)
            all_true_count.append(y_count.cpu().numpy())
    
    # Check if we collected any valid predictions
    if len(all_probs_types) == 0:
        print(f"\n⚠️  WARNING: No valid batches collected in {prefix} epoch {epoch}!")
        print("   All videos failed to load. Check OpenCV installation.")
        # Return dummy metrics to continue
        return 0.0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, 0.0, {}, {}
    
    # Aggregate
    all_probs_types = np.concatenate(all_probs_types, axis=0)
    all_preds_types = np.concatenate(all_preds_types, axis=0)
    all_true_types = np.concatenate(all_true_types, axis=0)
    all_preds_count = np.concatenate(all_preds_count, axis=0)
    all_true_count = np.concatenate(all_true_count, axis=0)
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    overall_metrics = compute_multilabel_metrics(all_preds_types, all_true_types)
    count_acc = (all_preds_count == all_true_count).mean()
    
    per_class_metrics = compute_per_class_metrics(all_probs_types, all_true_types, CANON_BUG_TYPES)
    per_count_metrics = compute_count_class_metrics(all_preds_count, all_true_count)
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar(f'{prefix}/loss', avg_loss, epoch)
        writer.add_scalar(f'{prefix}/accuracy', overall_metrics['accuracy'], epoch)
        writer.add_scalar(f'{prefix}/precision', overall_metrics['precision'], epoch)
        writer.add_scalar(f'{prefix}/recall', overall_metrics['recall'], epoch)
        writer.add_scalar(f'{prefix}/f1', overall_metrics['f1'], epoch)
        writer.add_scalar(f'{prefix}/count_accuracy', count_acc, epoch)
        
        for bug_type in CANON_BUG_TYPES:
            m = per_class_metrics[bug_type]
            writer.add_scalar(f'{prefix}/per_class/{bug_type}/precision', m['precision'], epoch)
            writer.add_scalar(f'{prefix}/per_class/{bug_type}/recall', m['recall'], epoch)
            writer.add_scalar(f'{prefix}/per_class/{bug_type}/f1', m['f1'], epoch)
        
        for count_label in ['0_bugs', '1_bug', '2_bugs', '3_bugs']:
            m = per_count_metrics[count_label]
            writer.add_scalar(f'{prefix}/per_count/{count_label}/precision', m['precision'], epoch)
            writer.add_scalar(f'{prefix}/per_count/{count_label}/recall', m['recall'], epoch)
            writer.add_scalar(f'{prefix}/per_count/{count_label}/f1', m['f1'], epoch)
    
    # Log to W&B
    if wandb_run:
        try:
            log_dict = {
                f'{prefix}/loss': avg_loss,
                f'{prefix}/accuracy': overall_metrics['accuracy'],
                f'{prefix}/precision': overall_metrics['precision'],
                f'{prefix}/recall': overall_metrics['recall'],
                f'{prefix}/f1': overall_metrics['f1'],
                f'{prefix}/count_accuracy': count_acc,
                'epoch': epoch
            }
            
            for bug_type in CANON_BUG_TYPES:
                m = per_class_metrics[bug_type]
                log_dict[f'{prefix}/per_class/{bug_type}/precision'] = m['precision']
                log_dict[f'{prefix}/per_class/{bug_type}/recall'] = m['recall']
                log_dict[f'{prefix}/per_class/{bug_type}/f1'] = m['f1']
            
            for count_label in ['0_bugs', '1_bug', '2_bugs', '3_bugs']:
                m = per_count_metrics[count_label]
                log_dict[f'{prefix}/per_count/{count_label}/precision'] = m['precision']
                log_dict[f'{prefix}/per_count/{count_label}/recall'] = m['recall']
                log_dict[f'{prefix}/per_count/{count_label}/f1'] = m['f1']
            
            wandb.log(log_dict)
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")
    
    return avg_loss, overall_metrics, count_acc, per_class_metrics, per_count_metrics

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--manifests', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--wandb-project', type=str, default='resnet18-ablation')
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Declare global variables
    global WANDB_AVAILABLE, _VIDEO_LOAD_FAILURES, _VIDEO_LOAD_SUCCESS
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("RESNET18-ONLY ABLATION TRAINING")
    print(f"{'='*80}\n")
    
    # Check OpenCV
    if CV2_AVAILABLE:
        print(f"✓ OpenCV available: {cv2.__version__}")
        print(f"  Build info: {cv2.getBuildInformation()[:200]}...")
    else:
        print("⚠️  OpenCV NOT available - video loading will fail!")
        print("   Install with: pip install opencv-python-headless --break-system-packages")
        sys.exit(1)
    
    print()
    # W&B with timeout fix
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            # Increase timeout to 5 minutes and add retry logic
            wandb_run = wandb.init(
                project=args.wandb_project, 
                config=vars(args),
                settings=wandb.Settings(
                    init_timeout=300,  # 5 minutes instead of 90 seconds
                    _disable_stats=True,  # Reduce network calls
                    _disable_meta=True
                )
            )
            print(f"✓ W&B initialized: {wandb.run.get_url()}")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            print("   Continuing with TensorBoard only...")
            wandb_run = None
            WANDB_AVAILABLE = False
    else:
        print("⚠️  W&B not available - using TensorBoard only")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"✓ TensorBoard: {args.logdir}")
    
    # Load data
    print(f"\nLoading data from: {args.manifests}")
    with open(args.manifests) as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    
    train_recs = [r for r in all_records if r.get("split", "").lower() == "train"]
    val_recs = [r for r in all_records if r.get("split", "").lower() == "val"]
    test_recs = [r for r in all_records if r.get("split", "").lower() == "test"]
    
    print(f"Split -> train: {len(train_recs)} | val: {len(val_recs)} | test: {len(test_recs)}")
    
    # Datasets
    ds_train = BugClipDataset(args.data_root, train_recs, split="train")
    ds_val = BugClipDataset(args.data_root, val_recs, split="val")
    ds_test = BugClipDataset(args.data_root, test_recs, split="test")
    
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_drop_none, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_drop_none, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_drop_none, pin_memory=True)
    
    # Model
    model = ResNet18OnlyModel(num_bug_types=5, pretrained=True).to(device)
    print(f"\n✓ Model: ResNet18-Only (NO BiLSTM) on {device}")
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.use_cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = None
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['opt_state'])
        if sched and 'sched_state' in ckpt:
            sched.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Training
    best_val_f1 = 0
    best_path = None
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*80)
        
        # Train
        train_loss, train_overall, train_count_acc, train_per_class, train_per_count = run_epoch(
            model, train_loader, opt, scaler, device, train=True,
            epoch=epoch, writer=writer, wandb_run=wandb_run, prefix='train'
        )
        
        # Val
        val_loss, val_overall, val_count_acc, val_per_class, val_per_count = run_epoch(
            model, val_loader, opt, scaler, device, train=False,
            epoch=epoch, writer=writer, wandb_run=wandb_run, prefix='val'
        )
        
        # Log LR
        current_lr = opt.param_groups[0]['lr']
        if writer:
            writer.add_scalar('learning_rate', current_lr, epoch)
        if wandb_run:
            wandb.log({'learning_rate': current_lr, 'epoch': epoch})
        
        # Scheduler
        if sched:
            sched.step()
        
        # Print
        print(f"\nTrain: loss={train_loss:.4f} | acc={train_overall['accuracy']:.4f} | "
              f"prec={train_overall['precision']:.4f} | rec={train_overall['recall']:.4f} | "
              f"f1={train_overall['f1']:.4f} | count_acc={train_count_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f} | acc={val_overall['accuracy']:.4f} | "
              f"prec={val_overall['precision']:.4f} | rec={val_overall['recall']:.4f} | "
              f"f1={val_overall['f1']:.4f} | count_acc={val_count_acc:.4f}")
        
        # Print video loading stats after first epoch
        if epoch == 0:
            print(f"\n📹 Video Loading Stats:")
            print(f"   Success: {_VIDEO_LOAD_SUCCESS}")
            print(f"   Failures: {_VIDEO_LOAD_FAILURES}")
            if _VIDEO_LOAD_FAILURES > 0:
                print(f"   ⚠️  High failure rate! Check OpenCV and video files.")
        
        # Save best
        if val_overall['f1'] > best_val_f1:
            best_val_f1 = val_overall['f1']
            best_path = Path(args.save_dir) / f"best_epoch_{epoch:03d}_f1_{val_overall['f1']:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': sched.state_dict() if sched else None,
                'val_f1': val_overall['f1'],
                'config': vars(args)
            }, best_path)
            print(f"✓ Saved best: {best_path.name}")
        
        # Save checkpoint
        ckpt_path = Path(args.save_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'sched_state': sched.state_dict() if sched else None,
            'val_f1': val_overall['f1'],
        }, ckpt_path)
        
        # Keep only last 3
        checkpoints = sorted(Path(args.save_dir).glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
    
    # Test evaluation
    if len(test_recs) > 0 and best_path and best_path.exists():
        print(f"\n{'='*80}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*80}\n")
        
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"✓ Loaded best model from epoch {ckpt['epoch']}")
        
        # Collect predictions
        all_preds_types = []
        all_true_types = []
        all_probs_types = []
        all_preds_count = []
        all_true_count = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation"):
                if batch is None:
                    continue
                x, y_types, y_count, y_any, _ = batch
                x = x.to(device)
                
                logits_types, logits_count, logit_any = model(x)
                
                probs_types = torch.sigmoid(logits_types).cpu().numpy()
                preds_types = (probs_types > 0.5).astype(int)
                preds_count = torch.argmax(logits_count, dim=1).cpu().numpy()
                
                all_probs_types.append(probs_types)
                all_preds_types.append(preds_types)
                all_true_types.append(y_types.cpu().numpy())
                all_preds_count.append(preds_count)
                all_true_count.append(y_count.cpu().numpy())
        
        all_probs_types = np.concatenate(all_probs_types, axis=0)
        all_preds_types = np.concatenate(all_preds_types, axis=0)
        all_true_types = np.concatenate(all_true_types, axis=0)
        all_preds_count = np.concatenate(all_preds_count, axis=0)
        all_true_count = np.concatenate(all_true_count, axis=0)
        
        # Compute all metrics
        test_overall = compute_multilabel_metrics(all_preds_types, all_true_types)
        test_count_acc = (all_preds_count == all_true_count).mean()
        test_per_class = compute_per_class_metrics(all_probs_types, all_true_types, CANON_BUG_TYPES)
        test_per_count = compute_count_class_metrics(all_preds_count, all_true_count)
        test_combo = compute_combo_metrics(all_preds_types, all_true_types, CANON_BUG_TYPES, top_k=20)
        
        # Print results
        print(f"\n{'='*80}")
        print("TEST SET RESULTS - COMPREHENSIVE")
        print(f"{'='*80}\n")
        
        print("OVERALL METRICS:")
        print(f"  Accuracy:    {test_overall['accuracy']:.4f}")
        print(f"  Precision:   {test_overall['precision']:.4f}")
        print(f"  Recall:      {test_overall['recall']:.4f}")
        print(f"  F1 Score:    {test_overall['f1']:.4f}")
        print(f"  Count Acc:   {test_count_acc:.4f}")
        
        print(f"\nPER-CLASS BUG TYPE METRICS:")
        print(f"{'Bug Type':<25} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-"*57)
        for bug_type in CANON_BUG_TYPES:
            m = test_per_class[bug_type]
            print(f"{bug_type:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        
        print(f"\nPER-COUNT CLASS METRICS:")
        print(f"{'Count':<12} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-"*44)
        for count_label in ['0_bugs', '1_bug', '2_bugs', '3_bugs']:
            m = test_per_count[count_label]
            print(f"{count_label:<12} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        
        print(f"\nTOP BUG COMBINATIONS:")
        print(f"{'Combination':<40} {'Count':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-"*80)
        for combo, metrics in sorted(test_combo.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"{combo:<40} {metrics['count']:<8} {metrics['precision']:>7.4f} {metrics['recall']:>7.4f} {metrics['f1']:>7.4f}")
        
        # Generate visualizations
        viz_dir = Path(args.save_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        plot_per_class_metrics(test_per_class, viz_dir / "per_class_bug_types.png",
                               title="Per-Class Bug Type Metrics (TEST)")
        print("✓ per_class_bug_types.png")
        
        plot_per_class_metrics(test_per_count, viz_dir / "per_count_class_metrics.png",
                               title="Per-Count Class Metrics (TEST)")
        print("✓ per_count_class_metrics.png")
        
        plot_confusion_matrix(all_true_count, all_preds_count,
                             viz_dir / "confusion_matrix_count.png",
                             title="Confusion Matrix - Bug Count (TEST)")
        print("✓ confusion_matrix_count.png")
        
        plot_combo_metrics(test_combo, viz_dir / "bug_combination_metrics.png",
                          title="Bug Combination Metrics (TEST)")
        print("✓ bug_combination_metrics.png")
        
        # Save comprehensive results
        comprehensive_results = {
            'overall_metrics': test_overall,
            'count_accuracy': float(test_count_acc),
            'per_class_bug_types': test_per_class,
            'per_count_classes': test_per_count,
            'bug_combinations': test_combo
        }
        
        results_path = Path(args.save_dir) / "comprehensive_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        print(f"✓ comprehensive_test_results.json")
        
        # Log to W&B
        if wandb_run:
            wandb.log({
                'test/accuracy': test_overall['accuracy'],
                'test/precision': test_overall['precision'],
                'test/recall': test_overall['recall'],
                'test/f1': test_overall['f1'],
                'test/count_accuracy': test_count_acc
            })
            
            for bug_type in CANON_BUG_TYPES:
                m = test_per_class[bug_type]
                wandb.log({
                    f'test/per_class/{bug_type}/precision': m['precision'],
                    f'test/per_class/{bug_type}/recall': m['recall'],
                    f'test/per_class/{bug_type}/f1': m['f1']
                })
            
            for count_label in ['0_bugs', '1_bug', '2_bugs', '3_bugs']:
                m = test_per_count[count_label]
                wandb.log({
                    f'test/per_count/{count_label}/precision': m['precision'],
                    f'test/per_count/{count_label}/recall': m['recall'],
                    f'test/per_count/{count_label}/f1': m['f1']
                })
            
            try:
                wandb.log({
                    "test_viz/per_class_bug_types": wandb.Image(str(viz_dir / "per_class_bug_types.png")),
                    "test_viz/per_count_classes": wandb.Image(str(viz_dir / "per_count_class_metrics.png")),
                    "test_viz/confusion_matrix": wandb.Image(str(viz_dir / "confusion_matrix_count.png")),
                    "test_viz/bug_combinations": wandb.Image(str(viz_dir / "bug_combination_metrics.png"))
                })
            except:
                pass
            
            wandb.run.summary.update({
                'test_accuracy': test_overall['accuracy'],
                'test_precision': test_overall['precision'],
                'test_recall': test_overall['recall'],
                'test_f1': test_overall['f1'],
                'test_count_accuracy': test_count_acc
            })
            
            print("✓ All metrics logged to W&B")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('test/accuracy', test_overall['accuracy'], args.epochs)
            writer.add_scalar('test/precision', test_overall['precision'], args.epochs)
            writer.add_scalar('test/recall', test_overall['recall'], args.epochs)
            writer.add_scalar('test/f1', test_overall['f1'], args.epochs)
            writer.add_scalar('test/count_accuracy', test_count_acc, args.epochs)
            
            for bug_type in CANON_BUG_TYPES:
                m = test_per_class[bug_type]
                writer.add_scalar(f'test/per_class/{bug_type}/precision', m['precision'], args.epochs)
                writer.add_scalar(f'test/per_class/{bug_type}/recall', m['recall'], args.epochs)
                writer.add_scalar(f'test/per_class/{bug_type}/f1', m['f1'], args.epochs)
            
            for count_label in ['0_bugs', '1_bug', '2_bugs', '3_bugs']:
                m = test_per_count[count_label]
                writer.add_scalar(f'test/per_count/{count_label}/precision', m['precision'], args.epochs)
                writer.add_scalar(f'test/per_count/{count_label}/recall', m['recall'], args.epochs)
                writer.add_scalar(f'test/per_count/{count_label}/f1', m['f1'], args.epochs)
            
            print("✓ All metrics logged to TensorBoard")
        
        print(f"\n{'='*80}")
        print("✓ TEST EVALUATION COMPLETE")
        print(f"{'='*80}\n")
    
    writer.close()
    if wandb_run:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best val F1: {best_val_f1:.4f}")
    if best_path:
        print(f"Best model: {best_path}")

if __name__ == "__main__":
    main()
