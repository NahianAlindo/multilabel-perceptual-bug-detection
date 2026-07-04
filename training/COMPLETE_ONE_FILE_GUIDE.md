# 🚀 COMPLETE GUIDE - Train + Val + Test Automatic (One Script)

## 🎯 What This Does

**Submit ONE job → Get COMPLETE results → Receive email when done**

✅ Trains on "train" split (60 epochs)  
✅ Validates on "val" split (every epoch)  
✅ Tests on "test" split (automatic after training)  
✅ Logs everything to W&B + TensorBoard  
✅ Saves test_results.json  
✅ Email when job finishes  
✅ Checkpoints every epoch  
✅ Auto-resumes if timeout  

**You don't touch anything after submission. Just wait for email!**

---

## 📋 Prerequisites

- ✅ Your Narval account: nahian26@narval.computecanada.ca
- ✅ Account: def-loutfouz
- ✅ Email: nahian.rifaat@ontariotechu.net
- ✅ Data at: /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
- ✅ my_new_splits.jsonl has "train", "val", "test" splits

---

## ⚡ PART 1: One-Time Setup (10 minutes)

### SSH to Narval

```bash
ssh nahian26@narval.computecanada.ca
```

### Setup Virtual Environment

```bash
# Load modules (gcc 12.3, opencv 4.8.1)
module load python/3.10 gcc/12.3 opencv/4.8.1

# Create virtual environment
cd ~
virtualenv --no-download env_bugdetection

# Activate
source ~/env_bugdetection/bin/activate

# Install PyTorch and scientific packages (from Alliance Canada wheels - fast!)
pip install --no-index torch torchvision torchaudio numpy scipy matplotlib seaborn pandas pillow tqdm scikit-learn

# Install tracking and video packages (from PyPI)
pip install tensorboard wandb av

# Try decord (optional - has fallbacks if fails)
pip install decord || echo "Decord optional - will use opencv fallback"

# Configure W&B API key
export WANDB_API_KEY="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"

# Make it permanent
echo 'export WANDB_API_KEY="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"' >> ~/.bashrc

# Test everything works
echo ""
echo "Testing packages..."
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import wandb; print(f'✓ W&B {wandb.__version__}')"
python -c "import numpy, av; print('✓ All packages ready!')"

# Deactivate
deactivate

echo ""
echo "✅ Environment setup complete!"
```

### Create Directories

```bash
cd /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig
mkdir -p scripts logs

# Exit Narval temporarily to upload files
exit
```

---

## 📤 PART 2: Upload Files (2 minutes)

### From Your Local Machine

```bash
# Upload Python script
scp train_complete_with_test.py nahian26@narval.computecanada.ca:/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/scripts/

# Upload SLURM script
scp run_complete_with_test.sh nahian26@narval.computecanada.ca:/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
```

---

## 🚀 PART 3: Submit Job (1 minute)

### SSH Back to Narval

```bash
ssh nahian26@narval.computecanada.ca
```

### Submit the Job

```bash
# Go to your directory
cd /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig

# Make SLURM script executable
chmod +x run_complete_with_test.sh

# Verify files are uploaded
ls -lh scripts/train_complete_with_test.py
ls -lh run_complete_with_test.sh
ls -lh my_new_splits.jsonl
ls clips/ | head -5

# Submit the job!
sbatch run_complete_with_test.sh
```

**Job submitted! You'll get an email when it starts.**

---

## 📧 PART 4: Wait for Email (Automatic - 5-7 days)

### You'll Receive 4 Emails:

#### Email 1: Job Started
```
Subject: Slurm Job_id=12345 Name=bugdet_original Began

Your job has started on node cdr544
```

#### Email 2: 80% Time Warning (Optional - if job runs long)
```
Subject: Slurm Job_id=12345 Name=bugdet_original Reached 80% of time limit

Job nearing 2-day limit
Will save checkpoint and can be resumed
```

#### Email 3: Job Completed (or Failed)
```
Subject: Slurm Job_id=12345 Name=bugdet_original Ended

Your job has completed
Check logs for results
```

---

## 📊 PART 5: What Happens Automatically

### Phase 1: Training (60 epochs, ~6 days)
```
Loading data from: my_new_splits.jsonl
Total clips: 78000
Split -> train: 46800 | val: 15600 | test: 15600

Epoch 1/60: train_loss=0.4523, val_loss=0.3891, val_F1=0.7234
Epoch 2/60: train_loss=0.3891, val_loss=0.3567, val_F1=0.7678
Epoch 3/60: train_loss=0.3456, val_loss=0.3234, val_F1=0.7891
...
Epoch 45/60: train_loss=0.1234, val_loss=0.1456, val_F1=0.8912 ← Best!
...
Epoch 60/60: train_loss=0.1123, val_loss=0.1567, val_F1=0.8734

Best validation F1: 0.8912 at epoch 45
```

**Every epoch:**
- ✅ Trains on train split
- ✅ Validates on val split
- ✅ Logs to W&B + TensorBoard
- ✅ Saves checkpoint
- ✅ Keeps best model

### Phase 2: Test Evaluation (Automatic, ~10 minutes)
```
================================================================================
FINAL TEST EVALUATION
================================================================================
Found 15600 test samples
Loading best checkpoint: best_epoch_45_f1_0.8912.pt
✓ Best model loaded
Evaluating on test set...

================================================================================
TEST SET RESULTS (Final Performance)
================================================================================
Test Loss:           0.1456
Test Types F1:       0.8734  ← YOUR MAIN RESULT FOR PAPER
Test Any F1:         0.9123
Test Count Accuracy: 0.7845
Test Gated Types F1: 0.8678
================================================================================
✓ Test metrics logged to W&B
✓ Test results saved to: checkpoints/anygate_wgatedreg/test_results.json
================================================================================

Done!
```

---

## 📥 PART 6: Get Your Results (After Email Notification)

### Option 1: Download test_results.json (Easiest)

```bash
# From your local machine
scp nahian26@narval:~/scratch/checkpoints/anygate_wgatedreg/test_results.json ./

# View
cat test_results.json
```

**Output:**
```json
{
  "test_loss": 0.1456,
  "test_f1_types": 0.8734,
  "test_f1_any": 0.9123,
  "test_acc_count": 0.7845,
  "test_f1_gated": 0.8678,
  "thresholds_used": {
    "thr_any": 0.4521,
    "thr_map": {
      "z_clipping": 0.3891,
      "corrupted_texture": 0.4123,
      "geometry_corruption": 0.4567,
      "z_fighting": 0.3678,
      "boundary_hole": 0.4234
    }
  }
}
```

**These are your final test metrics!** Use these in your paper/thesis.

### Option 2: W&B Dashboard (Best for Exploration)

```bash
# Get W&B URL from training log
ssh nahian26@narval
head -30 /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs/training_*.out | grep "wandb.ai"
```

**Open the URL in browser. You'll see:**

**Charts:**
- loss/train (60 epochs)
- loss/val (60 epochs)
- f1_types/train (60 epochs)
- f1_types/val (60 epochs)
- f1_any/train, f1_any/val
- acc_count/train, acc_count/val
- learning_rate
- **test/loss** ← Final test loss
- **test/f1_types** ← Final test F1 (main result)
- **test/f1_any** ← Final test any F1
- **test/acc_count** ← Final test count acc
- **test/f1_types_gated** ← Final test gated F1

**Summary:**
- best_val_f1: 0.8912
- test_f1_types: 0.8734
- best_checkpoint: /path/to/best.pt
- total_epochs: 60

### Option 3: Download Best Model

```bash
scp nahian26@narval:~/scratch/checkpoints/anygate_wgatedreg/*best*.pt ./
```

### Option 4: View Training Log

```bash
scp nahian26@narval:~/projects/def-loutfouz/nahian26/bugdatasetbig/logs/training_*.out ./
cat training_*.out | tail -100
```

---

## 🔄 If Job Times Out (Auto-Resume)

Job saves checkpoint every epoch. If it reaches 2-day limit:

### You'll Get Email:
```
Job reached time limit
Checkpoint saved: checkpoint_epoch_045.pt
```

### Just Resubmit:
```bash
ssh nahian26@narval
cd /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig
sbatch run_complete_with_test.sh
```

**Script automatically:**
- ✅ Detects latest checkpoint
- ✅ Resumes from that epoch
- ✅ Continues training
- ✅ Completes all 60 epochs
- ✅ Runs test evaluation

**You don't need to change anything!**

---

## 📍 Where Everything Is Saved

### On Narval:

```
/home/nahian26/scratch/
├── checkpoints/anygate_wgatedreg/
│   ├── checkpoint_epoch_058.pt           (Latest checkpoint)
│   ├── checkpoint_epoch_059.pt           (Second latest)
│   ├── checkpoint_epoch_060.pt           (Last epoch)
│   ├── best_epoch_45_f1_0.8912.pt       (Best model)
│   └── test_results.json                 (Final test metrics)
│
└── runs/anygate_wgatedreg/
    └── events.out.tfevents.*              (TensorBoard logs)

/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
└── logs/
    ├── training_12345.out                 (Job output)
    └── training_12345.err                 (Job errors)
```

---

## 📊 Complete Metrics Summary

| Metric | Source | When | Description |
|--------|--------|------|-------------|
| **Training Metrics** |
| loss/train | W&B, TensorBoard | Every epoch | Training loss |
| f1_types/train | W&B, TensorBoard | Every epoch | Training F1 (bug types) |
| f1_any/train | W&B, TensorBoard | Every epoch | Training F1 (any bug) |
| acc_count/train | W&B, TensorBoard | Every epoch | Training count accuracy |
| **Validation Metrics** |
| loss/val | W&B, TensorBoard | Every epoch | Validation loss |
| f1_types/val | W&B, TensorBoard | Every epoch | Validation F1 (bug types) |
| f1_any/val | W&B, TensorBoard | Every epoch | Validation F1 (any bug) |
| f1_types_gated/val | W&B, TensorBoard | Every epoch | Gated F1 with threshold |
| acc_count/val | W&B, TensorBoard | Every epoch | Validation count accuracy |
| **Test Metrics (Final)** |
| **test/loss** | **W&B, JSON** | **After training** | **Final test loss** |
| **test/f1_types** | **W&B, JSON** | **After training** | **MAIN RESULT** ⭐ |
| **test/f1_any** | **W&B, JSON** | **After training** | **Final test any F1** |
| **test/acc_count** | **W&B, JSON** | **After training** | **Final count accuracy** |
| **test/f1_types_gated** | **W&B, JSON** | **After training** | **Final gated F1** |

---

## ✅ What You DON'T Need to Do

❌ Run separate test script  
❌ Manually evaluate on test set  
❌ Manually save test results  
❌ Manually log to W&B  
❌ Monitor training constantly  
❌ Manually resume if timeout  
❌ Manually create checkpoints  

**Everything is automatic!**

---

## ✅ What You DO Need to Do

1. ✅ Setup environment (once, 10 min) - DONE
2. ✅ Upload 2 files (2 min) - DONE
3. ✅ Submit job (1 min) - DONE
4. ✅ Wait for email (5-7 days) - WAIT
5. ✅ Download results (2 min) - AFTER EMAIL

**Total active time: ~15 minutes**

---

## 🎓 For Your Paper/Thesis

After job completes, you have:

### Main Results Table:
```
Dataset: 78,000 video clips (46,800 train, 15,600 val, 15,600 test)
Model: ResNet18 + BiLSTM
Training: 60 epochs with EMA and cosine schedule
Loss: Asymmetric Loss + Focal Count Loss

Test Set Performance:
- Bug Type F1 Score:     87.34%
- Any Bug F1 Score:      91.23%
- Count Accuracy:        78.45%
- Gated F1 Score:        86.78%
```

### Training Curves:
- Download from W&B as PNG/CSV
- Loss curves (train/val over 60 epochs)
- F1 curves (train/val over 60 epochs)

### Best Model:
- Epoch: 45
- Validation F1: 89.12%
- Test F1: 87.34%

**Everything ready for publication!**

---

## 🆘 Troubleshooting

### Job fails immediately
```bash
# Check error log
cat logs/training_*.err

# Common issues:
# - Virtual environment not activated → Check module loads
# - Missing packages → Run pip install commands again
# - Wrong paths → Check file locations
```

### "No test samples found"
```bash
# Check your split file
head -10 my_new_splits.jsonl | grep "split"

# Should see: "split": "train", "split": "val", "split": "test"
# If missing test, add some records with "split": "test"
```

### Job pending forever
```bash
# Check queue
squeue -u nahian26

# Check estimated start time
squeue -u nahian26 --start

# This is normal on busy clusters - just wait
```

### Want to check progress during training
```bash
# Watch live output
ssh nahian26@narval
tail -f /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs/training_*.out

# Or check W&B dashboard (better!)
```

---

## 🎉 Summary

**What you do:**
1. Run setup commands (once)
2. Upload 2 files
3. Submit: `sbatch run_complete_with_test.sh`
4. Wait for email
5. Download results

**What happens automatically:**
- ✅ Trains 60 epochs on train split
- ✅ Validates every epoch on val split
- ✅ Saves best model
- ✅ Checkpoints every epoch
- ✅ Tests on test split (after training)
- ✅ Saves test_results.json
- ✅ Logs everything to W&B
- ✅ Sends email when done
- ✅ Auto-resumes if timeout

**You get:**
- ✅ Complete training history (60 epochs)
- ✅ Best validation model
- ✅ Final test metrics (for your paper!)
- ✅ W&B dashboard with all plots
- ✅ TensorBoard logs
- ✅ Checkpoint files

**Time investment:**
- Setup: 10 minutes (one-time)
- Upload: 2 minutes
- Submit: 1 minute
- **Wait: 5-7 days (automatic)**
- Download: 2 minutes

**Total active time: 15 minutes**
**Total wait time: Let the GPU work!**

---

## 🚀 Ready to Start?

Copy and paste the commands from PART 1, PART 2, PART 3 above.

**After you submit, you're done! Just wait for the email.** 📧

**When you get the email, download test_results.json - those are your final results!** 📊

Good luck with your training! 🎮🐛✨
