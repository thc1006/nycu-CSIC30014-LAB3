# Data Directory Structure

This directory contains CSV metadata files for the chest X-ray classification dataset.

## Files Included

- `train_data.csv` - Training set metadata (70KB)
- `val_data.csv` - Validation set metadata (13KB)
- `test_data_sample.csv` - Test set metadata sample (16KB)

## Required Data (Not in GitHub)

Due to size constraints, the following directories are **NOT** included in this repository:

### Image Directories

You need to download or prepare these directories locally:

```
train_images/          # Training images (~2-5GB)
val_images/            # Validation images (~500MB-1GB)
test_images/           # Test images (~500MB-1GB)
```

### CSV File Format

Each CSV file has the following columns:

- `new_filename`: Image filename (e.g., "1234.jpeg")
- `normal`: One-hot label for Normal class (0 or 1)
- `bacteria`: One-hot label for Bacterial Pneumonia (0 or 1)
- `virus`: One-hot label for Viral Pneumonia (0 or 1)
- `COVID-19`: One-hot label for COVID-19 (0 or 1)

Example:
```csv
new_filename,normal,bacteria,virus,COVID-19
1234.jpeg,0.0,1.0,0.0,0.0
5678.jpeg,1.0,0.0,0.0,0.0
```

## Setup Instructions for Colab

When using this repository in Google Colab:

1. **Clone the repository**:
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

2. **Upload your image directories to Google Drive**:
   - Create a folder structure in Google Drive:
     ```
     MyDrive/chest-xray-data/
       ├── train_images/
       ├── val_images/
       └── test_images/
     ```

3. **Mount Google Drive in Colab**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. **Update config paths** in `configs/base.yaml`:
```yaml
data:
  images_dir_train: /content/drive/MyDrive/chest-xray-data/train_images
  images_dir_val: /content/drive/MyDrive/chest-xray-data/val_images
  images_dir_test: /content/drive/MyDrive/chest-xray-data/test_images
  train_csv: data/train_data.csv
  val_csv: data/val_data.csv
  test_csv: data/test_data.csv
```

5. **Generate test_data.csv if needed**:
```python
!python -m src.build_test_csv --config configs/model_stage1.yaml
```

## Data Statistics

### Training Set
- Total images: ~3000-4000
- Class distribution:
  - Normal: ~40%
  - Bacteria: ~35%
  - Virus: ~24%
  - COVID-19: ~1%

### Validation Set
- Total images: ~700
- Similar class distribution to training set

### Test Set
- Total images: ~1200
- No labels provided (for Kaggle submission)

## Important Notes

⚠️ **Do NOT commit image files to GitHub**
- Image directories are excluded in `.gitignore`
- Total image data is ~3-7GB, exceeding GitHub limits

✅ **CSV files are included**
- Small metadata files (<100KB total)
- Safe to include in version control
