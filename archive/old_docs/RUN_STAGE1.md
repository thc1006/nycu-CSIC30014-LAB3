# 🚀 Stage 1 Optimization - Quick Start Guide

## 📋 Overview

This Stage 1 optimization implements:
- ✅ **ConvNeXt-Base** (upgraded from ResNet18)
- ✅ **512×512 resolution** (upgraded from 224×224)
- ✅ **Improved Focal Loss** with targeted class weights [1.0, 1.5, 2.0, 1.2]
- ✅ **Mixup/CutMix** augmentation (50% probability)
- ✅ **Stochastic Weight Averaging (SWA)** (epochs 25-30)
- ✅ **Advanced augmentation** (rotation, affine, color jitter, random erasing)
- ✅ **Test-Time Augmentation (TTA)** for final predictions

**Expected Performance**: 85-87% Macro-F1 Score (up from 80%)

---

## 🖥️ Local Training (RTX 3050)

### Step 1: Test Configuration
```bash
# Sanity check - make sure everything works
python -m src.train_v2 --config configs/model_stage1.yaml
```

**Note**: On RTX 3050, 512×512 with batch_size=8 will take approximately **4-5 hours** for 30 epochs.

### Step 2: Train Model
```bash
# Full training
python -m src.train_v2 --config configs/model_stage1.yaml
```

### Step 3: Generate Predictions with TTA
```bash
# Standard prediction
python -m src.predict --config configs/model_stage1.yaml --ckpt outputs/stage1_convnext512/best.pt

# With Test-Time Augmentation (+2-3% boost)
python -m src.tta_predict --config configs/model_stage1.yaml --ckpt outputs/stage1_convnext512/best.pt
```

---

## ☁️ A100 Colab Training (RECOMMENDED)

### Setup in Colab

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone/upload your project
!git clone <your-repo> /content/project
# OR upload manually

# 3. Install dependencies
!pip install timm  # For advanced models if needed

# 4. Upload data to Colab
# Make sure train_images, val_images, test_images, and CSVs are accessible
```

### Modify Config for A100

Edit `configs/model_stage1.yaml`:
```yaml
train:
  batch_size: 24      # Increase from 8 (A100 has 40GB memory)

perf:
  cudnn_benchmark: true  # Already set, but confirm
```

### Train on A100

```bash
# Training will take ~2 hours on A100 (vs 4-5 hours on RTX 3050)
!python -m src.train_v2 --config configs/model_stage1.yaml
```

### Download Results

```python
from google.colab import files

# Download best model
files.download('outputs/stage1_convnext512/best.pt')

# Download submission
files.download('data/submission_stage1.csv')
```

---

## 📊 What Changed?

### Model Architecture
| Component | Before | After |
|-----------|--------|-------|
| Backbone | ResNet18 (11M params) | ConvNeXt-Base (89M params) |
| Resolution | 224×224 | 512×512 |
| Epochs | 10 | 30 |
| Learning Rate | 0.0003 | 0.0001 |

### Loss Function
| Component | Before | After |
|-----------|--------|-------|
| Type | Focal Loss | **Improved Focal Loss** |
| Class Weights | None | [1.0, 1.5, 2.0, 1.2] |
| Label Smoothing | 0.05 | 0.1 |
| Gamma | 2.0 | 2.0 |

**Why these weights?**
- Normal: 1.0 (performing well)
- Bacteria: 1.5 (70% acc, needs boost)
- **Virus: 2.0** (67% acc, worst class)
- COVID-19: 1.2 (100% but only 7 samples)

### Data Augmentation

**Standard (before)**:
- RandomHorizontalFlip (p=0.5)
- RandomRotation (10°)
- ColorJitter (brightness=0.1, contrast=0.1)

**Advanced (Stage 1)**:
- RandomHorizontalFlip (p=0.5)
- RandomRotation (15°) ⬆️
- **RandomAffine** (translate=0.1, scale=0.9-1.1, shear=10) ✨ NEW
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05) ⬆️
- **RandomErasing** (p=0.3) ✨ NEW
- **Mixup/CutMix** (p=0.5) ✨ NEW

### Test-Time Augmentation (TTA)

Applies 6 transformations and averages predictions:
1. Original
2. Horizontal Flip
3. Vertical Flip
4. Rotate 90°
5. Rotate 180°
6. Rotate 270°

**Expected boost**: +2-3% accuracy with minimal extra time

---

## 🔍 Monitor Training

Expected training log:
```
[device] cuda | CUDA name: NVIDIA GeForce RTX 3050 Laptop GPU
[loss] ImprovedFocalLoss (gamma=2.0, alpha=[1.0, 1.5, 2.0, 1.2], smoothing=0.1)
[augment] Mixup/CutMix enabled (alpha=1.0, prob=0.5)
[SWA] enabled (start epoch=25, lr=5e-05)

[epoch 01] train acc=0.3500 f1=0.2800 | val acc=0.4500 f1=0.3500
[epoch 02] train acc=0.5200 f1=0.4800 | val acc=0.6000 f1=0.5500
...
[epoch 10] train acc=0.8000 f1=0.7800 | val acc=0.7900 f1=0.7700
...
[epoch 20] train acc=0.8800 f1=0.8700 | val acc=0.8500 f1=0.8400
...
[epoch 30] train acc=0.9200 f1=0.9100 | val acc=0.8700 f1=0.8600
  -> saved new best to outputs/stage1_convnext512/best.pt (val macro-F1=0.8600)

[SWA] Updating BatchNorm statistics...
[SWA final] val acc=0.8750 f1=0.8650
  -> saved SWA model to outputs/stage1_convnext512/best_swa.pt (val macro-F1=0.8650)
```

---

## 🎯 Expected Performance

| Metric | Baseline (ResNet18) | Stage 1 (ConvNeXt-Base) | Stage 1 + TTA |
|--------|---------------------|-------------------------|---------------|
| Macro-F1 | 0.801 | 0.850-0.870 | 0.870-0.890 |
| Normal F1 | 0.897 | 0.920-0.940 | 0.930-0.950 |
| **Bacteria F1** | 0.762 | 0.820-0.850 | 0.840-0.870 |
| **Virus F1** | 0.619 | 0.780-0.820 | 0.800-0.840 |
| COVID-19 F1 | 0.875 | 0.900-0.950 | 0.920-0.970 |

**Key Improvements**:
- ✅ Virus class: +16-22% (biggest bottleneck addressed)
- ✅ Bacteria class: +6-11%
- ✅ Overall: +7-9% with TTA

---

## 🐛 Troubleshooting

### OutOfMemoryError (CUDA OOM)

**On RTX 3050**:
```yaml
train:
  batch_size: 4  # Reduce from 8
```

### Training Too Slow

**Option 1**: Reduce epochs
```yaml
train:
  epochs: 20  # Instead of 30
  swa_start: 15
```

**Option 2**: Use Colab A100 (2x faster)

### Import Errors

Make sure to use the new training script:
```bash
# ❌ Don't use
python -m src.train --config configs/model_stage1.yaml

# ✅ Use this
python -m src.train_v2 --config configs/model_stage1.yaml
```

---

## 📈 Next Steps (Stage 2)

After Stage 1 training completes, you can further improve with:

1. **Multi-model Ensemble** (+2-4%)
   - Train 3 models with different seeds
   - Average predictions

2. **Multi-scale Training** (+1-2%)
   - Train on 384, 448, 512 simultaneously

3. **Pseudo-Labeling** (+1-3%)
   - Use high-confidence test predictions for semi-supervised learning

**Expected Final Score**: 90-93% 🎯

---

## 📝 Files Created

- `configs/model_stage1.yaml` - Stage 1 configuration
- `src/train_v2.py` - Enhanced training script with SWA, Mixup, etc.
- `src/tta_predict.py` - Test-Time Augmentation prediction
- `src/losses.py` - Added ImprovedFocalLoss
- `src/aug.py` - Added Mixup/CutMix functions

---

## ✅ Checklist

Before starting training:
- [ ] Config file exists: `configs/model_stage1.yaml`
- [ ] Data paths are correct in base.yaml
- [ ] CUDA is available (run `torch.cuda.is_available()`)
- [ ] Sufficient disk space (~500MB for model checkpoints)
- [ ] test_data.csv exists in data/ folder

Ready to train! 🚀
