#!/usr/bin/env python3
# eval_anygate.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if you want windows to pop up
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

# ---- import core defs from your NEW training script ----
# Name must match your file: train_bugdetector_anygate.py
from train_bugdetector_anygate import (
    BugClipDataset, collate_drop_none,
    discover_manifests, load_all_records,
    CANON_BUG_TYPES, gate_predictions,
)

# (If your train file exposes a build_combo_names helper, you can import that instead)

import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

# -------------------------
# Model defs (compatible)
# -------------------------
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

class BugBiLSTM_2HEAD(nn.Module):
    """Matches older 2-head checkpoints with keys: .cls and .count"""
    def __init__(self, num_bug_types, pretrained=False):
        super().__init__()
        try:
            enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            enc = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(enc.children())[:-1]); self.feat_dim = 512
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = TemporalAttention(d_model=512)
        self.cls   = nn.Linear(512, num_bug_types)
        self.count = nn.Linear(512, 4)
    def forward(self, frames):
        B,T,C,H,W = frames.shape
        x = frames.view(B*T, C, H, W)
        f = self.cnn(x).flatten(1).view(B, T, self.feat_dim)
        y,_ = self.lstm(f)
        z, att = self.attn(y)
        return self.cls(z), self.count(z), att  # (types, count, att)

class BugBiLSTM_ANYGATE(nn.Module):
    """Matches your new 3-head checkpoints with keys: .fc_types, .fc_count, .fc_any"""
    def __init__(self, num_bug_types, pretrained=False):
        super().__init__()
        try:
            enc = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            enc = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(enc.children())[:-1]); self.feat_dim = 512
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = TemporalAttention(d_model=512)
        self.fc_types = nn.Linear(512, num_bug_types)
        self.fc_count = nn.Linear(512, 4)
        self.fc_any   = nn.Linear(512, 1)
    def forward(self, frames):
        B,T,C,H,W = frames.shape
        x = frames.view(B*T, C, H, W)
        f = self.cnn(x).flatten(1).view(B, T, self.feat_dim)
        y,_ = self.lstm(f)
        z, att = self.attn(y)
        logits_types = self.fc_types(z)
        logits_count = self.fc_count(z)
        logit_any    = self.fc_any(z).squeeze(-1)
        return logits_types, logits_count, logit_any, att

def build_model_for_ckpt(state_dict, num_bug_types, device):
    keys = list(state_dict.keys())
    is_anygate = any(k.startswith("fc_types.") or k.startswith("fc_any.") for k in keys)
    if is_anygate:
        model = BugBiLSTM_ANYGATE(num_bug_types=num_bug_types, pretrained=False).to(device)
    else:
        model = BugBiLSTM_2HEAD(num_bug_types=num_bug_types, pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, is_anygate

# -------------------------
# Metrics / tuning helpers
# -------------------------
@torch.no_grad()
def compute_confusion_count(model, loader, device, is_anygate, num_classes=4):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        if batch is None: continue
        # New dataset tuple: (x, y_multi, y_count, y_any, path)
        x, _y_multi, y_count, _y_any, _path = batch
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits_count = out[1]  # index 1 for both models defined above
        preds.append(logits_count.argmax(dim=1).cpu().numpy())
        gts.append(y_count.cpu().numpy())
    if not preds:
        return None
    y_pred = np.concatenate(preds); y_true = np.concatenate(gts)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred): cm[int(t), int(p)] += 1
    return cm

def plot_confusion_count(cm, class_names=None, save_path=None, show=False):
    if class_names is None:
        class_names = ["0 bugs","1 bug","2 bugs","3 bugs"]
    if cm is None:
        print("No samples for confusion matrix."); return
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Count Head)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150); print(f"Saved confusion matrix to: {save_path}")
    if show: plt.show()
    plt.close(fig)

@torch.no_grad()
def compute_confusion_count_gated(model, loader, device, is_anygate, class_names, thr_any, thr_map):
    """
    Uses presence gating + per-class thresholds and enables count_nudge.
    Also forces count=0 when pred_any=0 to reduce 0→1 leakage.
    """
    model.eval()
    cm = np.zeros((4, 4), dtype=np.int64)

    for batch in loader:
        if batch is None:
            continue
        x, y_multi, y_count, y_any, _ = batch
        x = x.to(device, non_blocking=True)

        if is_anygate:
            logits_types, logits_count, logit_any, _ = model(x)
            type_probs = torch.sigmoid(logits_types)
            any_prob   = torch.sigmoid(logit_any).squeeze(-1)
        else:
            # 2-head: no explicit any head — proxy with max type prob
            logits_types, logits_count, _ = model(x)
            type_probs = torch.sigmoid(logits_types)
            any_prob   = type_probs.max(dim=1).values

        # --- Gated predictions with nudging ---
        pred_any, pred_types, pred_count = gate_predictions(
            type_probs, any_prob, float(thr_any), thr_map, class_names,
            count_logits=logits_count, count_nudge=True
        )

        # Presence gate the count (force 0 if no bug predicted)
        pred_count = torch.where(pred_any == 0, torch.zeros_like(pred_count), pred_count)

        # Accumulate confusion
        y_true = y_count.cpu().numpy()
        y_pred = pred_count.cpu().numpy()
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

    return cm

@torch.no_grad()
def collect_probs_targets(model, loader, device, is_anygate, class_names):
    """
    Returns numpy arrays:
      S_types [N,C] probs, Y_types [N,C] {0,1}
      S_any   [N]   probs (presence head if available; else max type prob)
      Y_any   [N]   {0,1} from GT (any positive type)
    """
    model.eval()
    S_types, Y_types, S_any, Y_any = [], [], [], []
    for batch in loader:
        if batch is None: continue
        x, y_multi, _y_count, y_any, _path = batch
        x = x.to(device, non_blocking=True)
        y_multi = y_multi.to(device, non_blocking=True)
        out = model(x)
        if is_anygate:
            logits_types, _count, logit_any = out[0], out[1], out[2]
            any_prob = torch.sigmoid(logit_any)
        else:
            logits_types = out[0]
            any_prob = torch.sigmoid(logits_types).amax(dim=1)  # OR approx
        S_types.append(torch.sigmoid(logits_types).cpu())
        Y_types.append(y_multi.cpu())
        S_any.append(any_prob.cpu())
        # use GT y_any directly (already in dataset)
        Y_any.append(y_any.cpu())

    if not S_types:
        return None, None, None, None
    S_types = torch.cat(S_types, 0).numpy()
    Y_types = torch.cat(Y_types, 0).numpy().astype(int)
    S_any   = torch.cat(S_any,   0).numpy()
    Y_any   = torch.cat(Y_any,   0).numpy().astype(int)
    return S_types, Y_types, S_any, Y_any

def tune_presence_threshold_np(S_any, Y_any, grid=None, target_recall=None):
    if grid is None: grid = np.linspace(0.01, 0.99, 99)
    best = {"f1": -1.0, "thr": 0.5, "prec": 0.0, "rec": 0.0}
    for t in grid:
        p = (S_any >= t).astype(int)
        tp = (p & Y_any).sum(); fp = (p & (1 - Y_any)).sum(); fn = ((1 - p) & Y_any).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if target_recall is not None:
            if rec >= target_recall and (t < best["thr"] or best["f1"] < 0):
                best = {"f1": f1, "thr": float(t), "prec": prec, "rec": rec}
        else:
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": float(t), "prec": prec, "rec": rec}
    return best

def tune_thresholds_np(S_types, Y_types, class_names, grid=None):
    if grid is None: grid = np.linspace(0.2, 0.8, 13)  # coarse
    thr_map = {}
    for k, name in enumerate(class_names):
        s = S_types[:, k]; y = Y_types[:, k]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            p = (s >= t).astype(int)
            tp = (p & y).sum(); fp = (p & (1 - y)).sum(); fn = ((1 - p) & y).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        thr_map[name] = float(np.clip(best_t, 0.2, 0.8))
    return thr_map

def per_class_f1_np(S_types, Y_types, class_names, thr_map):
    rows = []
    for k, name in enumerate(class_names):
        s = S_types[:, k]; y = Y_types[:, k]
        t = float(thr_map.get(name, 0.5))
        p = (s >= t).astype(int)
        tp = (p & y).sum(); fp = (p & (1 - y)).sum(); fn = ((1 - p) & y).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append((name, prec, rec, f1, t))
    return rows

def gated_types_micro_f1_np(S_types, Y_types, S_any, thr_any, thr_map, class_names):
    thr_vec = np.array([thr_map.get(n, 0.5) for n in class_names])[None, :]
    pred_any = (S_any >= float(thr_any)).astype(int)[:, None]     # [N,1]
    p_types  = ((S_types >= thr_vec).astype(int)) * pred_any      # [N,C]
    tp = np.logical_and(p_types == 1, Y_types == 1).sum()
    fp = np.logical_and(p_types == 1, Y_types == 0).sum()
    fn = np.logical_and(p_types == 0, Y_types == 1).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-9))

# Fallback: build frequent combo names from Y (min_support = 5)
def build_combo_names_from_Y(Y_types_np, class_names, min_support=5):
    C = len(class_names)
    combos = []
    cnt = Counter()
    for row in Y_types_np:
        present = [class_names[i] for i in range(C) if row[i] == 1]
        if not present: continue
        key = "|".join(sorted(set(present)))
        cnt[key] += 1
    for k, v in cnt.items():
        if v >= min_support and k != "NONE":
            combos.append(k)
    return sorted(combos)

def combo_f1_np(S_types, Y_types, class_names, combos, thr_map, thr_any=None, S_any=None):
    scores = {}
    N, C = S_types.shape
    if thr_any is not None and S_any is not None:
        gate = (S_any >= float(thr_any)).astype(int)[:, None]
    else:
        gate = np.ones((N, 1), dtype=int)
    for combo in combos:
        want = combo.split("|")
        kidx = [class_names.index(c) for c in want]
        prob_min = S_types[:, kidx].min(axis=1, keepdims=True)  # AND as min
        t_c = min(thr_map.get(c, 0.5) for c in want)
        pred = ((prob_min >= t_c).astype(int) * gate).squeeze(1)
        gt = (Y_types[:, kidx].sum(axis=1) == len(kidx)).astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        scores[combo] = (prec, rec, f1)
    return scores

# ---------- plotting helpers ----------
def plot_multilabel_metrics(metrics, save_path=None, show=False, title=None):
    if metrics is None:
        print("No samples for multilabel metrics."); return
    names = [m[0] for m in metrics]; precs = [m[1] for m in metrics]
    recs  = [m[2] for m in metrics]; f1s   = [m[3] for m in metrics]
    x = np.arange(len(names)); width = 0.25
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x - width, precs, width, label="Precision")
    ax.bar(x,         recs,  width, label="Recall")
    ax.bar(x + width, f1s,   width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(title or "Per-class metrics"); ax.legend()
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150); print(f"Saved multilabel metrics to: {save_path}")
    if show: plt.show()
    plt.close(fig)

def plot_combo_metrics(combo_scores, save_path=None, show=False, title="Frequent combo metrics"):
    if not combo_scores:
        print("No frequent combos to plot (none met min_support)."); return
    items = sorted(combo_scores.items(), key=lambda kv: kv[1][2], reverse=True)
    names = [k for k,_ in items]
    precs = [v[0] for _,v in items]; recs  = [v[1] for _,v in items]; f1s   = [v[2] for _,v in items]
    N = len(names)
    if N > 25:
        names, precs, recs, f1s = names[:25], precs[:25], recs[:25], f1s[:25]
        print(f"Plotting top 25 combos by F1 (out of {N})")
    x = np.arange(len(names)); width = 0.28
    fig, ax = plt.subplots(figsize=(max(10, 0.5*len(names)), 5))
    ax.bar(x - width, precs, width, label="Precision")
    ax.bar(x,         recs,  width, label="Recall")
    ax.bar(x + width, f1s,   width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150); print(f"Saved combo metrics to: {save_path}")
    if show: plt.show()
    plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt)")
    ap.add_argument("--save-dir", type=str, help="Where to save outputs")
    ap.add_argument("--no-show", action="store_true", help="Only save figures, don't open windows")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="Threshold for simple (ungated) multilabel plots")
    ap.add_argument("--presence-recall", type=float, default=0.95,
                    help="Target recall for τ_any when thresholds not saved in ckpt")
    ap.add_argument("--data-root", type=str, default=None,
                help="Override dataset root (defaults to value saved in checkpoint)")
    ap.add_argument("--manifests-glob", type=str, default=None,
                    help="Override manifests glob (defaults to value saved in checkpoint)")
    args = ap.parse_args()

    # Interactive fallbacks
    if not args.checkpoint:
        args.checkpoint = input("Enter path to checkpoint (.pt): ").strip()
    if not args.save_dir:
        d = input("Enter directory to save evaluation results (default=eval_outputs): ").strip()
        args.save_dir = d if d else "eval_outputs"
    if not args.no_show:
        s = input("Show figures interactively? [y]/n: ").strip().lower()
        args.no_show = (s == "n")

    print("\n--- Final arguments ---")
    print("Checkpoint :", args.checkpoint)
    print("Save dir   :", args.save_dir)
    print("No-show    :", args.no_show)
    print("Ungated thr:", args.thr)

    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})
    bug_types = ckpt.get("bug_types", CANON_BUG_TYPES)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print("Epoch:", ckpt.get("epoch"))
    if "val_types_f1" in ckpt:
        print("Best val types F1:", ckpt.get("val_types_f1"))

     # --- rebuild val dataset the same way as training ---
    saved_root = Path(cfg["data_root"]) if "data_root" in cfg else None
    data_root = Path(args.data_root) if args.data_root else saved_root
    if data_root is None:
        raise ValueError("No data_root found. Pass --data-root D:\\path\\to\\bugdataset")

    glob_pat = args.manifests_glob if args.manifests_glob else cfg.get("manifests_glob", "**/*.jsonl")

    print(f"[dataset] data_root = {data_root}")
    print(f"[dataset] manifests_glob = {glob_pat}")

    manifests = discover_manifests(data_root, glob_pat)
    print(f"[dataset] discovered {len(manifests)} manifest file(s)")
    if len(manifests) == 0:
        print("ERROR: No manifests found. Check --data-root / --manifests-glob.")
        return

    all_records = load_all_records(manifests)
    print(f"[dataset] loaded {len(all_records)} total record(s) from manifests")
    if len(all_records) == 0:
        print("ERROR: Manifests contained zero records. Verify jsonl contents/paths.")
        return

    seed = cfg.get("seed", 1337)
    train_ratio = cfg.get("train_ratio", 0.8)
    val_ratio   = cfg.get("val_ratio",   0.1)

    # deterministic simple split (matches your trainer’s ratios)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(all_records))
    rng.shuffle(idx)

    n_train = int(len(idx) * train_ratio)
    n_val   = int(len(idx) * val_ratio)

    # --- Fallbacks to avoid empty validation ---
    if n_val == 0:
        n_val = max(1, int(round(0.1 * len(idx))))  # at least 1 sample in val
    va_idx = idx[n_train:n_train+n_val]
    if len(va_idx) == 0:
        # if ratios led to empty, just take last 10% as val
        cut = max(1, int(round(0.1 * len(idx))))
        va_idx = idx[-cut:]

    val_recs = [all_records[i] for i in va_idx]
    print(f"[dataset] using {len(val_recs)} validation record(s)")

    ds_val = BugClipDataset(
        data_root, val_recs, split="val",
        num_frames=cfg.get("num_frames", 16),
        resize_hw=cfg.get("resize_hw", 224)
    )
    print(f"[dataset] BugClipDataset(len) = {len(ds_val)}")

    if len(ds_val) == 0:
        print("ERROR: Dataset built but length is 0. Likely path mismatches in records.")
        # Print a few example paths to help debug existence
        sample = val_recs[:3]
        for r in sample:
            vp = Path(r.get("video_path") or r.get("path") or "")
            print("  exists?", vp.exists(), "→", str(vp))
        return

    val_loader = DataLoader(
        ds_val, batch_size=max(1, cfg.get("batch_size", 8)//2),
        shuffle=False, num_workers=cfg.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_drop_none
    )

    # --- build proper model for ckpt & load weights ---
    num_types = len(bug_types)
    model, is_anygate = build_model_for_ckpt(ckpt["model_state"], num_types, device)
    model.eval()

    
    

    # --- Collect probs/targets once (works for both) ---
    S_types, Y_types, S_any, Y_any = collect_probs_targets(model, val_loader, device, is_anygate, bug_types)
    if S_types is None:
        print("No validation samples to evaluate."); return

    # --- Ungated per-class plot at user-specified threshold ---
    ungated_rows = []
    for k, name in enumerate(bug_types):
        s = S_types[:, k]; y = Y_types[:, k]
        p = (s >= args.thr).astype(int)
        tp = (p & y).sum(); fp = (p & (1 - y)).sum(); fn = ((1 - p) & y).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        ungated_rows.append((name, prec, rec, f1))
    plot_multilabel_metrics(
        ungated_rows,
        save_path=str(out_dir/f"multilabel_metrics_ungated_thr{args.thr:.2f}.png"),
        show=(not args.no_show),
        title=f"Per-class (UNGATED) @ thr={args.thr:.2f}"
    )

    # --- Presence + gated thresholds ---
    thr_map = ckpt.get("thr_map", None)
    thr_any = ckpt.get("thr_any", None)

    if (thr_map is None) or (thr_any is None):
        # Tune presence first (target recall), then per-class on gated subset
        any_obj = tune_presence_threshold_np(S_any, Y_any, target_recall=args.presence_recall)
        thr_any = any_obj["thr"]
        mask = (S_any >= thr_any)
        if mask.sum() >= 100:
            thr_map = tune_thresholds_np(S_types[mask], Y_types[mask], bug_types, grid=np.linspace(0.2,0.8,13))
        else:
            thr_map = {name: 0.5 for name in bug_types}
        print(f"[tuned] τ_any={thr_any:.2f} (prec={any_obj['prec']:.2f}, rec={any_obj['rec']:.2f})")
    else:
        print(f"[from ckpt] τ_any={float(thr_any):.2f} and {len(thr_map)} per-class τ loaded")

    # Presence metrics (for 2-head we used max(type) as proxy)
    p = (S_any >= float(thr_any)).astype(int)
    tp = (p & Y_any).sum(); fp = (p & (1 - Y_any)).sum(); fn = ((1 - p) & Y_any).sum()
    prec_any = tp / (tp + fp + 1e-9); rec_any = tp / (tp + fn + 1e-9)
    f1_any = 2 * prec_any * rec_any / (prec_any + rec_any + 1e-9)
    print(f"Presence metrics: P={prec_any:.3f} R={rec_any:.3f} F1={f1_any:.3f}")

    # --- Confusion matrix with presence gate + per-class thresholds + nudging ---
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        cm_gated = np.zeros((4, 4), dtype=np.int64)
        for batch in val_loader:
            if batch is None:
                continue
            x, y_multi, y_count, y_any, _ = batch
            x = x.to(device, non_blocking=True)

            if is_anygate:
                logits_types, logits_count, logit_any, _ = model(x)
                type_probs = torch.sigmoid(logits_types)
                any_prob   = torch.sigmoid(logit_any).squeeze(-1)
            else:
                # 2-head: proxy presence with max type prob
                logits_types, logits_count, _ = model(x)
                type_probs = torch.sigmoid(logits_types)
                any_prob   = type_probs.max(dim=1).values

            # <<< NUDGED, GATED PREDICTIONS >>>
            pred_any, pred_types, pred_count = gate_predictions(
                type_probs, any_prob, float(thr_any), thr_map, bug_types,
                count_logits=logits_count, count_nudge=True
            )

            # Presence-gate the count: if no bug predicted, force count=0
            pred_count = torch.where(pred_any == 0, torch.zeros_like(pred_count), pred_count)

            y_true = y_count.cpu().numpy()
            y_pred = pred_count.cpu().numpy()
            for t, p in zip(y_true, y_pred):
                cm_gated[int(t), int(p)] += 1

    plot_confusion_count(
        cm_gated,
        class_names=["0 bugs","1 bug","2 bugs","3 bugs"],
        save_path=str(out_dir / "confusion_count_GATED_NUDGED.png"),
        show=(not args.no_show)
    )
    print("Saved confusion matrix to:", str(out_dir / "confusion_count_GATED_NUDGED.png"))

    # Per-class F1 under gated thresholds (text) + plot
    type_rows = per_class_f1_np(S_types, Y_types, bug_types, thr_map)
    print("\nPer-class (gated) F1:")
    for name, P, R, F1, t in type_rows:
        print(f"{name:>18}: P={P:.3f} R={R:.3f} F1={F1:.3f} τ={t:.2f}")

    plot_multilabel_metrics(
        [(n,p,r,f) for (n,p,r,f,_) in type_rows],
        save_path=str(out_dir/"multilabel_metrics_GATED.png"),
        show=(not args.no_show),
        title="Per-class metrics (GATED)"
    )

    # Micro-F1 for types after gating (deployment metric)
    f1_types_gated = gated_types_micro_f1_np(S_types, Y_types, S_any, thr_any, thr_map, bug_types)
    print(f"\nMicro-F1 (types, GATED): {f1_types_gated:.3f}")

    # Frequent combos (gated) — use fallback builder from Y
    combos = build_combo_names_from_Y(Y_types, bug_types, min_support=5)
    if combos:
        combo_scores = combo_f1_np(S_types, Y_types, bug_types, combos, thr_map,
                                   thr_any=float(thr_any), S_any=S_any)
        print("\nFrequent combo (gated) F1:")
        for name,(P,R,F1) in sorted(combo_scores.items(), key=lambda kv: kv[1][2], reverse=True):
            print(f"{name:>24}: P={P:.3f} R={R:.3f} F1={F1:.3f}")
        plot_combo_metrics(
            combo_scores,
            save_path=str(out_dir/"combo_metrics_GATED.png"),
            show=(not args.no_show),
            title="Frequent bug-type combo metrics (GATED)"
        )
    else:
        print("\nNo frequent combos (min_support=5).")

if __name__ == "__main__":
    main()
