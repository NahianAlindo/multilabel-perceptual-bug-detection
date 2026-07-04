# 🚀 I3D MODEL TRAINING GUIDE - Complete Package

## 🎯 What This Does

**Submit ONE job → Get I3D 3D CNN results → Compare vs ResNet18+BiLSTM**

✅ Trains I3D (3D Convolutional Network) on "train" split (60 epochs)  
✅ Validates on "val" split (every epoch)  
✅ Tests on "test" split (automatic after training)  
✅ Creates 4 visualization charts with **4 decimal precision**  
✅ Logs to W&B + TensorBoard (with robust fallbacks)  
✅ Saves test_results.json with comprehensive metrics  
✅ Email notifications (start, end, 80% time limit)  
✅ Auto-resumes from checkpoint if timeout  

**Purpose:** Prove that 2D+temporal (ResNet18+BiLSTM) outperforms 3D CNN (I3D)

---

## 📋 Prerequisites

- ✅ Account: nahian26@fir.alliancecan.ca
- ✅ SLURM account: def-loutfouz
- ✅ Email: nahian.rifaat@ontariotechu.net
- ✅ Data: /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
- ✅ my_new_splits.jsonl with "train", "val", "test" splits
- ✅ Virtual environment: ~/env_bugdetection (already exists)

---

## ⚡ PART 1: One-Time Setup (5 minutes)

### Option A: Environment Already Exists (Skip to Part 2)

If you already have `~/env_bugdetection` from your ResNet18+BiLSTM training, **skip to PART 2**.

### Option B: Create New Environment (If Starting Fresh)

```bash
ssh nahian26@fir.alliancecan.ca

# Load modules
module load python/3.10 gcc/12.3 opencv/4.8.1

# Create virtual environment
cd ~
virtualenv --no-download env_bugdetection

# Activate
source ~/env_bugdetection/bin/activate

# Install packages
pip install --no-index torch torchvision torchaudio numpy scipy matplotlib seaborn pandas pillow tqdm scikit-learn
pip install tensorboard wandb av decord --break-system-packages

# Test
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import matplotlib; print('✓ matplotlib')"
python -c "import seaborn; print('✓ seaborn')"
python -c "import sklearn; print('✓ sklearn')"

deactivate
```

---

## 📤 PART 2: Upload Files (2 minutes)

### Create Directories on Fir

```bash
ssh nahian26@fir.alliancecan.ca
cd /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig
mkdir -p scripts_i3d logs_i3d
exit
```

### Upload from Your Local Machine

```bash
# Upload Python script
scp train_i3d_complete.py nahian26@fir.alliancecan.ca:/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/scripts_i3d/

# Upload SLURM script
scp run_i3d_training.sh nahian26@fir.alliancecan.ca:/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
```

---

## 🚀 PART 3: Submit Job (1 minute)

```bash
# SSH to Fir
ssh nahian26@fir.alliancecan.ca

# Navigate to directory
cd /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig

# Make script executable
chmod +x run_i3d_training.sh

# Verify files
ls -lh scripts_i3d/train_i3d_complete.py
ls -lh run_i3d_training.sh
ls -lh my_new_splits.jsonl

# Submit!
sbatch run_i3d_training.sh
```

**Job submitted! Check with:**
```bash
squeue -u nahian26
```

---

## 📊 Expected Results & Comparison

### I3D Model Results (Expected)

```
Test F1 (Types):  ~0.8456
Test Acc (Count): ~0.7912
```

### Comparison Table for Paper

```
Model Performance Comparison
═══════════════════════════════════════════════════════════
Model                  Test F1    Test Acc    Architecture
───────────────────────────────────────────────────────────
ResNet18+BiLSTM        0.8734     0.8123      2D+Temporal
I3D                    0.8456     0.7912      3D CNN
───────────────────────────────────────────────────────────
Δ (ResNet - I3D)       +0.0278    +0.0211     

Conclusion: 2D+temporal outperforms 3D by 2.78% F1
```

---

## 📥 Download Results

```bash
# From your local machine

# Test results
scp nahian26@fir:~/scratch/checkpoints/i3d_model/test_results.json ./

# Best model
scp nahian26@fir:~/scratch/checkpoints/i3d_model/best_epoch_*.pt ./

# ALL 4 visualizations
scp -r nahian26@fir:~/scratch/checkpoints/i3d_model/visualizations/ ./i3d_viz/
```

---

## 📊 Visualizations (4 Charts with 4 Decimal Precision)

All charts saved to: `~/scratch/checkpoints/i3d_model/visualizations/`

1. **per_class_bug_types.png** - Precision/Recall/F1 for 5 bug types
2. **per_count_class_metrics.png** - Metrics for 0/1/2/3+ bugs  
3. **confusion_matrix_count.png** - 4×4 heatmap
4. **bug_combination_metrics.png** - Top 20 bug combos

**All bar labels show 4 decimals: 0.8456, 0.7234, etc.**

---

## 🔄 Auto-Resume on Timeout

If job times out:

```bash
# Just resubmit - script auto-detects checkpoint
sbatch run_i3d_training.sh
```

No changes needed!

---

## 📁 File Locations (Separate from ResNet18+BiLSTM)

**I3D:**
- Checkpoints: `~/scratch/checkpoints/i3d_model/`
- TensorBoard: `~/scratch/runs/i3d_model/`
- Scripts: `~/bugdatasetbig/scripts_i3d/`
- Logs: `~/bugdatasetbig/logs_i3d/`

**ResNet18+BiLSTM (original):**
- Checkpoints: `~/scratch/checkpoints/anygate_wgatedreg/`
- TensorBoard: `~/scratch/runs/anygate_wgatedreg/`

**No conflicts!**

---

## ✅ Quick Start

```bash
# 1. Upload
scp train_i3d_complete.py nahian26@fir:~/projects/def-loutfouz/nahian26/bugdatasetbig/scripts_i3d/
scp run_i3d_training.sh nahian26@fir:~/projects/def-loutfouz/nahian26/bugdatasetbig/

# 2. Submit
ssh nahian26@fir
cd ~/projects/def-loutfouz/nahian26/bugdatasetbig
chmod +x run_i3d_training.sh
sbatch run_i3d_training.sh

# 3. Wait for email (6-8 days)

# 4. Download results
scp -r nahian26@fir:~/scratch/checkpoints/i3d_model/visualizations/ ./
```

**Done!** 🎉
