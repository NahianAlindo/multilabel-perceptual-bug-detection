#!/usr/bin/env python3
"""
Frame-cache alignment diagnostic (read-only, no training).

extract_frames_at_fps() picks stride = round(native_fps / target_fps) and
keeps every stride-th frame. The cache's REAL fps is native_fps / stride,
which only equals the nominal target_fps (8.0) when native_fps divides
evenly by it — e.g. 30fps source -> stride 4 -> actual 7.5 fps, not 8.
meta.json only records the requested nominal fps, never the actual
per-video one, so any code that indexes the cache assuming exactly 8.0 fps
silently drifts further from the true timestamp the longer the video runs.

This script checks, for every cached video, whether its actual cache fps
(cached_frame_count / duration) matches the nominal fps, and reports how
much accumulated time-drift that implies by the end of each video. It does
not need torch/decord/opencv — only numpy and the dataset JSON + cache dir.

Usage:
    python check_frame_alignment.py \
        --temporal-dataset /home/nahian26/scratch/temporal_bug_dataset.json \
        --frame-cache-dir  /home/nahian26/scratch/frame_cache_fps8_224 \
        --nominal-fps 8.0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def frame_cache_name(rel_video_path: str) -> str:
    # Mirrors train.py's frame_cache_name() exactly — must stay in sync.
    key = rel_video_path.replace("\\", "/").strip("/")
    key = os.path.splitext(key)[0].replace("/", "__")
    return key + ".npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal-dataset", required=True)
    ap.add_argument("--frame-cache-dir", required=True)
    ap.add_argument("--nominal-fps", type=float, default=8.0)
    ap.add_argument("--tolerance-pct", type=float, default=2.0,
                    help="Flag videos whose actual cache fps differs from nominal "
                         "by more than this percent")
    ap.add_argument("--out", default="frame_alignment_report.json")
    args = ap.parse_args()

    with open(args.temporal_dataset) as f:
        dataset = json.load(f)
    videos = dataset.get("videos", [])
    cache_dir = Path(args.frame_cache_dir)

    rows = []
    missing = 0
    for vid in videos:
        vp = vid["video_path"]
        duration = float(vid.get("duration", 0))
        if duration <= 0:
            continue
        cache_path = cache_dir / frame_cache_name(vp)
        if not cache_path.exists():
            missing += 1
            continue
        arr = np.load(cache_path, mmap_mode="r")
        n_frames = int(arr.shape[0])
        if n_frames == 0:
            continue
        actual_fps = n_frames / duration
        drift_pct = (actual_fps - args.nominal_fps) / args.nominal_fps * 100.0
        # If __getitem__ assumes nominal fps, the frame index it computes for real
        # time T actually corresponds to real time T * (nominal_fps / actual_fps).
        implied_drift_at_end_sec = duration * (args.nominal_fps / actual_fps - 1.0)
        rows.append({
            "video_id": vid.get("video_id", vp),
            "video_path": vp,
            "duration_sec": round(duration, 1),
            "cached_frames": n_frames,
            "expected_frames_at_nominal_fps": round(duration * args.nominal_fps, 1),
            "actual_cache_fps": round(actual_fps, 4),
            "drift_pct": round(drift_pct, 2),
            "implied_time_drift_at_video_end_sec": round(implied_drift_at_end_sec, 2),
        })

    if missing:
        print(f"[WARN] {missing} videos have no cache file yet — skipped "
              f"(run ensure_frame_cache first, or point --frame-cache-dir correctly)")

    if not rows:
        print("[ERROR] No cached videos found to check.")
        sys.exit(1)

    drift_pcts = np.array([r["drift_pct"] for r in rows])
    bad = [r for r in rows if abs(r["drift_pct"]) > args.tolerance_pct]

    print(f"\nChecked {len(rows)} cached videos (nominal fps = {args.nominal_fps})")
    print(f"  drift_pct: min={drift_pcts.min():.2f}  max={drift_pcts.max():.2f}  "
          f"mean={drift_pcts.mean():.2f}  median={np.median(drift_pcts):.2f}")
    print(f"  videos with |drift| > {args.tolerance_pct}%: {len(bad)}/{len(rows)} "
          f"({100 * len(bad) / len(rows):.1f}%)")

    if bad:
        worst = sorted(bad, key=lambda r: -abs(r["drift_pct"]))[:10]
        print("\nWorst 10 by |drift_pct| (implied drift is how far into the wrong "
              "moment of the video the served frames land by the final window):")
        print(f"  {'video_id':<42} {'dur(s)':>8} {'actual_fps':>11} "
              f"{'drift%':>8} {'end_drift(s)':>13}")
        for r in worst:
            print(f"  {r['video_id']:<42} {r['duration_sec']:>8} "
                  f"{r['actual_cache_fps']:>11} {r['drift_pct']:>8} "
                  f"{r['implied_time_drift_at_video_end_sec']:>13}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull per-video report written to {args.out}")

    if bad:
        print(f"\n[RESULT] CONFIRMED: {len(bad)}/{len(rows)} videos have frame-cache "
              f"fps drift beyond {args.tolerance_pct}%. This matches the diagnosed bug "
              f"in TemporalWindowDataset.__getitem__ (train.py) — already patched to use "
              f"each video's actual cache fps instead of the nominal one.")
        sys.exit(2)
    else:
        print("\n[RESULT] No significant drift detected across cached videos.")


if __name__ == "__main__":
    main()
