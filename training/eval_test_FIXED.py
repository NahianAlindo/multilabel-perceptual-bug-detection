#!/usr/bin/env python3
"""
Standalone Test Evaluation Script - FIXED FOR YOUR EXACT MODEL
Run this with your existing checkpoint to generate all visualizations

Usage:
python eval_test_FIXED.py \
  --checkpoint /path/to/best_checkpoint.pt \
  --data-root /path/to/bugdatasetbig \
  --manifests my_new_splits.jsonl \
  --output-dir ./visualizations
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Video loading
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except:
    AV_AVAILABLE = False

try:
    import decord
    DECORD_AVAILABLE = True
except:
    DECORD_AVAILABLE = False

# Bug types
CANON_BUG_TYPES = ["z-clipping", "corrupted_texture", "geometry_corruption", "z-fighting", "boundary_hole"]

# ==============================================================================
# MODEL (EXACT COPY FROM YOUR TRAINING SCRIPT)
# ==============================================================================

class TemporalAttention(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, x):
        h = torch.tanh(self.W(x))
        a = torch.softmax(self.v(h).squeeze(-1), dim=1)
        z = (x * a.unsqueeze(-1)).sum(dim=1)
        return z, a

class BugBiLSTM(nn.Module):
    """Your actual model with attention"""
    def __init__(self, num_bug_types=5, pretrained=False):
        super().__init__()
        try:
            enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except:
            enc = resnet18(weights=None)
        
        self.cnn = nn.Sequential(*list(enc.children())[:-1])
        self.feat_dim = 512
        
        # 2-layer BiLSTM with hidden=256
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        
        self.attn = TemporalAttention(d_model=512)
        self.head_drop = nn.Dropout(p=0.3)
        
        # Output heads
        self.fc_types = nn.Linear(512, num_bug_types)
        self.fc_count = nn.Linear(512, 4)  # 4 classes for count
        self.fc_any = nn.Linear(512, 1)
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        f = self.cnn(x).flatten(1)
        f = f.reshape(B, T, self.feat_dim).contiguous()
        y, _ = self.lstm(f)
        z, att = self.attn(y)
        z = self.head_drop(z)
        
        logits_types = self.fc_types(z)
        logits_count = self.fc_count(z)
        logit_any = self.fc_any(z).squeeze(-1)
        
        return logits_types, logits_count, logit_any, att

# ==============================================================================
# DATASET
# ==============================================================================

class BugClipDataset:
    def __init__(self, data_root, records, num_frames=16, resize_hw=224):
        self.data_root = Path(data_root)
        self.records = records
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_hw, resize_hw)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.records)
    
    def load_video_frames(self, video_path, num_frames=16):
        frames = []
        
        # Try OpenCV
        if CV2_AVAILABLE:
            cap = cv2.VideoCapture(str(video_path))
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            
            if len(all_frames) >= num_frames:
                indices = np.linspace(0, len(all_frames)-1, num_frames, dtype=int)
                return [all_frames[i] for i in indices]
        
        return None
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        video_path = self.data_root / rec["relpath"]
        
        frames = self.load_video_frames(video_path, self.num_frames)
        if frames is None or len(frames) < self.num_frames:
            return None
        
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        
        bug_types = rec.get("bug_types", [])
        y_types = torch.zeros(len(CANON_BUG_TYPES), dtype=torch.float32)
        for bt in bug_types:
            if bt in CANON_BUG_TYPES:
                y_types[CANON_BUG_TYPES.index(bt)] = 1.0
        
        y_count = len(bug_types)  # 0, 1, 2, or 3
        y_any = 1.0 if len(bug_types) > 0 else 0.0
        
        return frames_tensor, y_types, torch.tensor(y_count, dtype=torch.long), torch.tensor(y_any)

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def compute_per_class_metrics(y_pred, y_true, class_names):
    metrics = {}
    for i, class_name in enumerate(class_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    return metrics

def plot_per_class_metrics(metrics, save_path):
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
    ax.set_title('Per-class metrics (GATED)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_confusion_matrix_count(y_true, y_pred, save_path):
    """y_pred and y_true are class indices (0-3)"""
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
    ax.set_title('Confusion Matrix (Count Head)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def compute_combo_metrics(y_pred, y_true, class_names, top_k=20):
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

def plot_combo_metrics(combo_metrics, save_path):
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
    ax.set_title('Frequent bug-type combo metrics (GATED)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--manifests', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./visualizations')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model with exact architecture
    model = BugBiLSTM(num_bug_types=5, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("✓ Model loaded successfully")
    
    # Get thresholds
    thr_any = checkpoint.get('thr_any', 0.5)
    thr_map = checkpoint.get('thr_map', {name: 0.5 for name in CANON_BUG_TYPES})
    print(f"Using any threshold: {thr_any:.4f}")
    print(f"Using per-class thresholds: {thr_map}")
    
    # Load test data
    print(f"\nLoading test data from: {args.manifests}")
    with open(args.manifests) as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    
    test_recs = [r for r in all_records if r.get("split", "").lower() == "test"]
    print(f"Test samples: {len(test_recs)}")
    
    if len(test_recs) == 0:
        print("ERROR: No test samples found!")
        return
    
    # Create dataset
    test_ds = BugClipDataset(args.data_root, test_recs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_drop_none)
    
    # Collect predictions
    print("\n" + "="*70)
    print("COLLECTING TEST PREDICTIONS")
    print("="*70)
    
    all_preds_types = []
    all_true_types = []
    all_preds_count = []
    all_true_count = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue
            
            x, y_types, y_count, y_any = batch
            x = x.to(device)
            
            logits_types, logits_count, logit_any, _ = model(x)
            
            probs_types = torch.sigmoid(logits_types).cpu().numpy()
            prob_any = torch.sigmoid(logit_any).cpu().numpy()
            count_preds = torch.argmax(logits_count, dim=1).cpu().numpy()
            
            # Apply gating for types
            for i in range(len(probs_types)):
                if prob_any[i] > thr_any:
                    pred_binary = np.array([1 if probs_types[i][j] > thr_map.get(CANON_BUG_TYPES[j], 0.5) else 0
                                           for j in range(len(CANON_BUG_TYPES))])
                else:
                    pred_binary = np.zeros(len(CANON_BUG_TYPES))
                
                all_preds_types.append(pred_binary)
                all_true_types.append(y_types[i].numpy())
            
            all_preds_count.extend(count_preds)
            all_true_count.extend(y_count.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples...")
    
    all_preds_types = np.array(all_preds_types)
    all_true_types = np.array(all_true_types)
    all_preds_count = np.array(all_preds_count)
    all_true_count = np.array(all_true_count)
    
    print(f"\n✓ Collected {len(all_preds_types)} predictions")
    
    # Compute overall metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    # Types F1
    tp = ((all_preds_types == 1) & (all_true_types == 1)).sum()
    fp = ((all_preds_types == 1) & (all_true_types == 0)).sum()
    fn = ((all_preds_types == 0) & (all_true_types == 1)).sum()
    types_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # Count accuracy
    count_acc = (all_preds_count == all_true_count).mean()
    
    print(f"\nOverall Test Metrics:")
    print(f"  Types F1 (micro):    {types_f1:.4f}")
    print(f"  Count Accuracy:      {count_acc:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Per-class metrics
    print("\n1. Computing per-class metrics...")
    per_class_metrics = compute_per_class_metrics(all_preds_types, all_true_types, CANON_BUG_TYPES)
    plot_per_class_metrics(per_class_metrics, output_dir / "per_class_metrics.png")
    
    # Print table
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Bug Type':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for bug_type in CANON_BUG_TYPES:
        m = per_class_metrics[bug_type]
        print(f"{bug_type:<25} {m['precision']:>11.4f} {m['recall']:>11.4f} {m['f1']:>11.4f}")
    print("-" * 70)
    
    # 2. Confusion matrix
    print("\n2. Generating confusion matrix for count...")
    plot_confusion_matrix_count(all_true_count, all_preds_count, output_dir / "confusion_matrix_count.png")
    
    # 3. Combo metrics
    print("\n3. Computing bug combination metrics...")
    combo_metrics = compute_combo_metrics(all_preds_types, all_true_types, CANON_BUG_TYPES, top_k=20)
    plot_combo_metrics(combo_metrics, output_dir / "combo_metrics.png")
    
    # Print top combos
    print("\nTop Bug Combinations:")
    print("-" * 90)
    print(f"{'Combination':<50} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 90)
    for combo, metrics in sorted(combo_metrics.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"{combo:<50} {metrics['count']:<8} {metrics['precision']:>11.4f} {metrics['recall']:>11.4f} {metrics['f1']:>11.4f}")
    print("-" * 90)
    
    # 4. Save detailed metrics
    print("\n4. Saving detailed metrics JSON...")
    detailed_metrics = {
        "overall": {
            "types_f1": float(types_f1),
            "count_accuracy": float(count_acc)
        },
        "per_class_metrics": per_class_metrics,
        "combo_metrics": combo_metrics,
        "thresholds": {
            "any": float(thr_any),
            "per_class": {k: float(v) for k, v in thr_map.items()}
        }
    }
    
    with open(output_dir / "detailed_test_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"✓ Saved: {output_dir / 'detailed_test_metrics.json'}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\n📊 All results saved to: {output_dir}")
    print(f"   ✓ per_class_metrics.png")
    print(f"   ✓ confusion_matrix_count.png")
    print(f"   ✓ combo_metrics.png")
    print(f"   ✓ detailed_test_metrics.json")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
