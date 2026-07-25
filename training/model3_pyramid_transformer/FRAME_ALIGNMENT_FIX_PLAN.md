# Model 3 — Frame-Cache Alignment Fix: Deployment Plan

## What was wrong

`extract_frames_at_fps()` picked `stride = round(native_fps / 8)` and kept every
`stride`-th frame. The cache's *real* fps is `native_fps / stride`, which only
equals exactly 8.0 when `native_fps` divides evenly by 8. For the most common
capture rates (30fps, 60fps -> effective 7.5fps; 25fps -> effective 8.33fps),
it doesn't. `meta.json` only ever recorded the requested nominal fps, and
`TemporalWindowDataset.__getitem__` computed which cached frame to serve via
`f_start = int(win_start * 8.0)` — always assuming the nominal rate. Labels are
computed correctly in true seconds, but the *pixels* served for a window
increasingly came from the wrong moment in the video the longer it ran (~2s off
at 30s in, ~20s off at 5 minutes, ~40s off at 10 minutes for a 30/60fps source).

This is a plausible root cause of the near-random per-class AUC-ROC
(0.40-0.52) seen in both Model 2 and Model 3 — two unrelated architectures
sharing this same cache code — while the pre-existing baseline (which
decodes aligned frames directly, no cache) got a healthy 0.61 clip F1 on the
same videos/labels.

## What changed (all in `training/model3_pyramid_transformer/`)

- **`train.py`**
  - `extract_frames_at_fps()`: now samples by exact output timestamp
    (nearest native frame to `k/8` seconds per output slot) instead of a
    fixed integer stride — a freshly-built cache is genuinely at 8fps for
    any native fps.
  - `TemporalWindowDataset`: computes each video's *actual* cache fps from
    the cached array itself (`frame_count / duration`) and uses that for
    indexing, instead of assuming the nominal value — a safety net on top
    of the extraction fix.
- **`check_frame_alignment.py`** (new) — standalone, read-only, no GPU/torch
  needed. Checks every cached video's actual fps against nominal and reports
  the implied time-drift. Confirmed correct against synthetic test fixtures.
- **`run_fir_pilot.sh`** (new) — 5-epoch pilot (`--pilot`, no HPO) on the
  corrected cache, ~2h time limit, to cheaply confirm the fix before
  committing a multi-day allocation.
- **`run_fir_clean.sh`** (new) — the original `run_fir.sh` full pipeline
  (HPO + 100 epochs + test), repointed at fresh `_clean`-suffixed
  checkpoint/log directories so it never collides with or resumes from the
  original (drifted-data) run.
- **`sweep_postprocess.sh`** / **`run_fir_finetune.sh`** — repointed at the
  `_clean` checkpoint/log paths; unchanged logic otherwise. These now run
  *after* the clean retrain, not on top of the original corrupted checkpoint.

## Sequence on fir (primary cluster — already has the videos + original cache)

1. `git pull` in the scratch clone of this repo.
2. Rename the old frame cache out of the way (already decided):
   `mv frame_cache_fps8_224 frame_cache_fps8_224-bak2`
3. *(Optional but cheap and gives concrete evidence)* run the diagnostic
   against the renamed-old cache before it's touched further:
   ```bash
   python training/model3_pyramid_transformer/check_frame_alignment.py \
       --temporal-dataset /home/nahian26/scratch/temporal_bug_dataset.json \
       --frame-cache-dir  /home/nahian26/scratch/frame_cache_fps8_224-bak2
   ```
4. Submit the pilot (this also regenerates the (now-empty)
   `frame_cache_fps8_224` with the corrected extraction — one-time ~30-60 min
   cost, shared by every later job that points at the same dir):
   ```bash
   sbatch training/model3_pyramid_transformer/run_fir_pilot.sh
   ```
5. Check the pilot's output — F1/mAP@0.5 across 5 epochs should look like
   real learning (moving, trending), not flat/noisy near-baseline numbers.
6. If it looks healthy, submit the full clean retrain (2-3 days):
   ```bash
   sbatch training/model3_pyramid_transformer/run_fir_clean.sh
   ```
7. Once that finishes and `checkpoints/model3_pyramid_transformer_clean/best_composite.pt`
   exists, submit the post-processing sweep (~6h, no retrain):
   ```bash
   sbatch training/model3_pyramid_transformer/sweep_postprocess.sh
   ```
8. Only if precision/F1 still isn't good enough after the sweep, submit the
   loss-reweighted fine-tune (reads the sweep's winning combo via
   `FINETUNE_MIN_CONF`/`FINETUNE_NMS_THR` env vars):
   ```bash
   FINETUNE_MIN_CONF=<winner> FINETUNE_NMS_THR=<winner> \
       sbatch training/model3_pyramid_transformer/run_fir_finetune.sh
   ```

## Using server 2 (nibi) in parallel

SHARCNET/Alliance `/scratch` is **per-cluster, not shared** — nibi doesn't
automatically see anything on fir's scratch. To run the same pipeline there
in parallel (e.g. a second attempt, a different HPO seed, or just to not
wait on a single 2-3 day job), you need to get data onto nibi's own scratch:

**1. Code — no transfer needed, just clone directly on nibi:**
```bash
cd /home/nahian26/scratch
git clone <your-repo-url> multilabel-perceptual-bug-detection
```

**2. Dataset JSON — small, transfer directly cluster-to-cluster:**
```bash
# run from fir (or wherever has it), pushing to nibi:
rsync -avP /home/nahian26/scratch/temporal_bug_dataset.json \
    nahian26@nibi.alliancecan.ca:/home/nahian26/scratch/
```

**3. Raw video footage — the big transfer, do NOT transfer the 66GB frame
cache instead; let nibi build its own with the fixed extraction (avoids
moving a redundant copy of data you're rebuilding anyway):**
```bash
rsync -avP /home/nahian26/scratch/fnafautobug2 \
           /home/nahian26/scratch/fpsmicroautobug2 \
           /home/nahian26/scratch/mkartautobug2 \
    nahian26@nibi.alliancecan.ca:/home/nahian26/scratch/
```
(or, if `gameplayvideos.zip` on fir's scratch already contains all of these,
transferring that single archive may be simpler — check its contents first.)
This is large and will take a while; run it in a `tmux`/`screen` session (or
as a lightweight CPU-only job) so it survives your SSH session ending, and
kick it off now since it doesn't need a GPU allocation or wait in the GPU
queue.

**4. Once the transfer completes, run the identical sequence on nibi**
(steps 4-8 above work unchanged — `run_fir_pilot.sh`/`run_fir_clean.sh` etc.
aren't fir-specific despite the filename, they're generic Alliance SLURM
scripts using the same `/home/nahian26/scratch/...` path convention).

## What NOT to do

- Don't run `run_fir_clean.sh` (or the pilot) against a `frame_cache_fps8_224`
  directory that still has the old `.npy` files in it — `ensure_frame_cache()`
  only extracts videos that are *missing*, so it will silently keep serving
  the old drifted cache. Both new scripts print a warning if they detect this,
  but don't ignore it.
- Don't run `sweep_postprocess.sh` or `run_fir_finetune.sh` before the clean
  retrain finishes — they're already repointed at the `_clean` checkpoint
  path, which won't exist yet.
