# 📓 Notebooks Guide

This directory contains Jupyter notebooks for training on Google Colab.

---

## 🚀 **RECOMMENDED: Use This Notebook**

### **`Colab_A100_Final.ipynb`** ⭐ **START HERE**

**This is the production-ready, clean notebook for Google Colab!**

#### ✨ Features:
- ✅ **Complete end-to-end workflow** (12 simple steps)
- ✅ **Zero configuration needed** - just run all cells
- ✅ **Automatic data download** from Kaggle
- ✅ **Ultra-optimized for A100** (95-98% GPU utilization)
- ✅ **State-of-the-art model**: ConvNeXt-Base @ 512px
- ✅ **Expected score: 0.87-0.89** (Macro F1)

#### 📋 Quick Start:
1. **Open in Colab**:
   - Go to: https://colab.research.google.com/
   - Upload `Colab_A100_Final.ipynb`
   - Or use: `File` → `Open notebook` → `GitHub` → paste repo URL

2. **Select A100 GPU**:
   - `Runtime` → `Change runtime type`
   - Hardware accelerator: **GPU**
   - GPU type: **A100** (recommended) or T4

3. **Get Kaggle API key**:
   - Go to: https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

4. **Run all cells**:
   - `Runtime` → `Run all`
   - Upload `kaggle.json` when prompted
   - Wait ~1.5 hours (A100) or ~5 hours (T4)

5. **Download submission**:
   - Last cell will download `submission_a100_ultra.csv`
   - Submit to Kaggle competition

#### ⏱️ Time Required:
- **Setup**: 5-10 minutes
- **Training**: ~1.5 hours (A100) or ~5 hours (T4)
- **Inference**: 5 minutes

#### 🎯 Expected Performance:
- **Val F1**: 0.86-0.87
- **Public Score**: 0.87-0.89
- **GPU Throughput**: 400-500 images/sec
- **Memory Usage**: 35-38GB / 40GB

---

## 📚 Other Notebooks (Reference/Advanced)

### `A100_Ultra_Optimized_Kaggle.ipynb`
- Original development notebook
- More verbose with detailed explanations
- Use if you want to understand each optimization step

### `A100_Ultra_Optimized.ipynb`
- For Google Drive-based data loading
- Use if you already have data uploaded to Drive

### `Stage1_Colab_Training.ipynb`
- Beginner-friendly version with step-by-step guide
- Slower but easier to understand

---

## 🔍 Notebook Comparison

| Notebook | Best For | Training Time | Expected F1 | Complexity |
|----------|----------|---------------|-------------|------------|
| **Colab_A100_Final.ipynb** ⭐ | **Production use** | **1.5h (A100)** | **0.87-0.89** | **Simple** |
| A100_Ultra_Optimized_Kaggle.ipynb | Learning optimizations | 1.5h (A100) | 0.87-0.89 | Medium |
| A100_Ultra_Optimized.ipynb | Google Drive users | 1.5h (A100) | 0.87-0.89 | Medium |
| Stage1_Colab_Training.ipynb | Beginners | 2h (A100) | 0.85-0.87 | Simple |

---

## 💡 Tips for Best Results

### Before Training:
1. ✅ **Always select A100 GPU** (if available)
2. ✅ **Have Kaggle API token ready** (kaggle.json)
3. ✅ **Stable internet connection** for data download
4. ✅ **Keep Colab tab open** during training

### During Training:
1. Monitor GPU usage: `Resources` → `View resources`
2. GPU utilization should be **95-98%**
3. Throughput should be **400-500 img/sec**
4. Don't close the browser tab!

### After Training:
1. **Download submission.csv immediately** (before session expires)
2. Optionally download `outputs/a100_ultra/best.pt` (model checkpoint)
3. Submit to Kaggle and check leaderboard

---

## 🐛 Troubleshooting

### "Not A100!" Error
**Solution**: Runtime → Change runtime type → GPU type: A100

### OutOfMemoryError
**Solution**: In `Colab_A100_Final.ipynb`, this shouldn't happen (batch size is optimized)
- If it does: Check GPU type (should be A100)
- T4 users: Contact us for adjusted config

### "403 Forbidden" when downloading data
**Solution**:
- For **public dataset**: No action needed (notebook uses public dataset)
- For **competition**: Must join competition and accept rules first

### "Cannot find images"
**Solution**:
- Re-run Step 5 (Data Reorganization)
- Check that Step 4 (Download) completed successfully

### Training is slow (<200 img/s)
**Check**:
1. Is GPU actually A100? Run Step 0 to verify
2. Are you using the right notebook? (`Colab_A100_Final.ipynb`)
3. Is batch size correct? (should be 48 for A100)

---

## 📧 Need Help?

If you encounter issues not covered here:

1. **Check the troubleshooting guide**: `KAGGLE_SETUP_GUIDE.md`
2. **Review training output** for error messages
3. **Verify all steps** completed successfully

---

## 🎯 Expected Results Timeline

### On A100 GPU (1.5 hours total):

```
[epoch 01/30] val_f1=0.35  (3 min)
[epoch 05/30] val_f1=0.62  (15 min)
[epoch 10/30] val_f1=0.77  (30 min)
[epoch 15/30] val_f1=0.82  (45 min)
[epoch 20/30] val_f1=0.85  (60 min)
[epoch 25/30] val_f1=0.86  (75 min)
[epoch 30/30] val_f1=0.87  (90 min)

✓ Training complete!
```

### With TTA (+3-5 minutes):
```
Final Kaggle Score: 0.87-0.89 🎯
```

---

**Start with `Colab_A100_Final.ipynb` for the best experience! ⚡**
