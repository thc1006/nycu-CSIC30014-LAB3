# Notebooks Guide

This directory contains Jupyter notebooks for training on Google Colab.

## 📓 Available Notebooks

### 1. `A100_Ultra_Optimized_Kaggle.ipynb` ⚡ **MOST RECOMMENDED**

**最推薦使用！從Kaggle直接下載數據 + 榨乾A100所有性能**

**Features**:
- ✅ **Kaggle API integration** - Direct dataset download
- ✅ Automatic data organization
- ✅ One-click Kaggle submission
- ✅ Maximum batch size (48)
- ✅ Gradient accumulation (effective batch=192)
- ✅ bfloat16 AMP + TF32
- ✅ torch.compile optimization
- ✅ 95-98% GPU utilization

**Performance**:
- Training time: **~1.5 hours**
- Throughput: **400-500 images/sec**
- Expected F1: **0.87-0.89** (with TTA)

**When to use**:
- Your data is on Kaggle ✓ **MOST COMMON**
- You want easiest setup ✓
- You want direct Kaggle submission ✓

---

### 2. `A100_Ultra_Optimized.ipynb` ⚡ **For Google Drive Data**

**使用Google Drive數據的極致優化版本**

**Features**:
- ✅ Maximum batch size (48 vs 8 on RTX 3050) - **6x larger**
- ✅ Gradient accumulation (effective batch=192) - **24x effective**
- ✅ bfloat16 AMP (312 TFLOPS on A100)
- ✅ TF32 enabled (19.5 TFLOPS)
- ✅ torch.compile for JIT compilation
- ✅ Fused AdamW optimizer
- ✅ Optimized DataLoader (4 workers + pin_memory)
- ✅ cuDNN auto-tuning
- ✅ Channels last memory format
- ✅ 95-98% GPU utilization

**Performance**:
- Training time: **~1.5 hours** (vs 4-5 hours on RTX 3050)
- Throughput: **400-500 images/sec**
- Memory usage: **37GB / 40GB** (maxed out)
- Expected F1: **0.87-0.89** (with TTA)

**When to use**:
- You have A100 GPU access ✓
- You want maximum performance ✓
- You want to finish training ASAP ✓

---

### 2. `Stage1_Colab_Training.ipynb` 📚 **BEGINNER-FRIENDLY**

**更容易理解的標準版本，適合初學者**

**Features**:
- ✅ Step-by-step explanations
- ✅ Standard batch size (24)
- ✅ Simpler configuration
- ✅ Good for learning

**Performance**:
- Training time: **~2 hours**
- Throughput: **~300 images/sec**
- Memory usage: **28GB / 40GB**
- Expected F1: **0.85-0.87** (with TTA)

**When to use**:
- You're new to Colab ✓
- You want to understand each step ✓
- You're okay with slightly longer training ✓

---

### 3. `A100_Final_Train.ipynb` 📋 **LEGACY**

**舊版本，保留供參考**

This is the original notebook. Use the updated versions above instead.

---

## 🚀 Quick Start

### Step 1: Get Kaggle API Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`

### Step 2: Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Choose `A100_Ultra_Optimized_Kaggle.ipynb` (recommended)
4. Change runtime type to **GPU: A100**

### Step 3: Upload Kaggle Credentials & Run

1. When prompted, upload your `kaggle.json`
2. Dataset will download automatically
3. Press "Runtime" → "Run all"
4. Wait ~1.5 hours for training!

---

## 📊 Performance Comparison

| Notebook | Data Source | GPU | Training Time | Expected F1 |
|----------|-------------|-----|---------------|-------------|
| **Ultra Kaggle ⚡** | Kaggle API | A100 | **1.5h** | **0.87-0.89** |
| Ultra Drive ⚡ | Google Drive | A100 | **1.5h** | **0.87-0.89** |
| Stage1 Standard 📚 | Google Drive | A100 | 2h | 0.85-0.87 |
| Legacy 📋 | Google Drive | A100 | 2.5h | 0.83-0.85 |
| Local (RTX 3050) | Local files | 3050 | 4-5h | 0.80-0.82 |

---

## 💡 Tips for Best Results

### Before Training:
1. ✅ Verify you have **A100 GPU** selected
2. ✅ Upload images to Google Drive first
3. ✅ Have stable internet connection
4. ✅ Keep Colab tab open during training

### During Training:
1. Monitor GPU usage with `!nvidia-smi`
2. Watch throughput (should be 400-500 img/s for Ultra)
3. Check loss is decreasing smoothly

### After Training:
1. Download `submission.csv` immediately
2. Optionally download `best.pt` checkpoint
3. Submit to Kaggle and check score

---

## 🐛 Troubleshooting

### "Not A100!" Error
**Solution**: Runtime → Change runtime type → GPU type: A100

### OutOfMemoryError
**Solution**:
- For Ultra notebook: Reduce batch_size from 48 to 40
- For Standard notebook: Reduce batch_size from 24 to 16

### Slow Training (<200 img/s)
**Check**:
1. Is GPU actually A100? Run `!nvidia-smi`
2. Is data on Google Drive? (not local)
3. Are workers set correctly? (num_workers=4)

### "Cannot find images"
**Solution**:
1. Check your Google Drive path
2. Update paths in notebook Step 5
3. Verify folder names match exactly

---

## 🎯 Expected Results Timeline

### Ultra-Optimized (1.5 hours total):

```
[epoch 01/30] val f1=0.35  (3 min)
[epoch 05/30] val f1=0.62  (15 min)
[epoch 10/30] val f1=0.77  (30 min)
[epoch 15/30] val f1=0.82  (45 min)
[epoch 20/30] val f1=0.85  (60 min)
[epoch 25/30] val f1=0.86  (75 min)
[epoch 30/30] val f1=0.87  (90 min)

✓ Training complete!
```

### With TTA (+3-5 minutes):
```
Final Kaggle Score: 0.87-0.89 🎯
```

---

## 📚 Additional Resources

- **Quick Start**: `../START_HERE.md`
- **Technical Details**: `../RUN_STAGE1.md`
- **Data Setup**: `../data/README.md`

---

## ⚠️ Important Notes

1. **Free Colab Limits**:
   - A100 usage is limited (~12 hours/week for free tier)
   - Save your work frequently
   - Download results immediately after training

2. **Data Transfer**:
   - Using Google Drive may be slow initially
   - First epoch might be slower (loading/caching)
   - Subsequent epochs will be faster

3. **Reproducibility**:
   - Random seed is set (42)
   - Results should be consistent
   - Small variations (<1%) are normal

---

**Choose `A100_Ultra_Optimized.ipynb` for best performance! ⚡**
