import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

# Support both relative and absolute imports
try:
    from .aug import build_transforms
except ImportError:
    from aug import build_transforms

class CSVDataset(Dataset):
    def __init__(self, csv_path, images_dir, file_col, label_cols, img_size=224, augment=True, advanced_aug=False, aug_config=None, medical_preprocessing=False, preprocessing_preset='default'):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.file_col = file_col
        self.label_cols = list(label_cols)
        self.img_size = img_size
        self.transforms = build_transforms(img_size, augment, advanced_aug, aug_config)

        # 🏥 醫學影像預處理
        self.medical_preprocessing = medical_preprocessing
        if medical_preprocessing:
            from .medical_preprocessing import create_medical_preprocessor
            self.medical_preprocessor = create_medical_preprocessor(preprocessing_preset)
            print(f"[CSVDataset] Medical preprocessing enabled: {preprocessing_preset}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row[self.file_col])

        # 支持動態影像目錄 (K-Fold CSV 包含 source_dir 欄位)
        if 'source_dir' in row.index and pd.notna(row['source_dir']):
            path = os.path.join(row['source_dir'], fname)
        else:
            path = os.path.join(self.images_dir, fname)

        image = Image.open(path).convert("RGB")

        # 🏥 應用醫學影像預處理 (在數據增強之前)
        if self.medical_preprocessing:
            image = self.medical_preprocessor(image)
            # 轉回 RGB (預處理後是灰階)
            image = image.convert("RGB")

        x = self.transforms(image)

        # one-hot -> class index (argmax). If labels are NaN (test), return -1
        if self.label_cols and self.label_cols[0] in row.index and not np.any(pd.isna(row[self.label_cols])):
            y_vec = row[self.label_cols].values.astype(float)
            y_idx = int(np.argmax(y_vec))
        else:
            y_idx = -1
        return x, y_idx, fname

def make_loader(csv_path, images_dir, file_col, label_cols, img_size, batch_size, num_workers, augment, shuffle=True, weighted=False, advanced_aug=False, aug_config=None, medical_preprocessing=False, preprocessing_preset='default'):
    ds = CSVDataset(csv_path, images_dir, file_col, label_cols, img_size, augment, advanced_aug, aug_config, medical_preprocessing, preprocessing_preset)

    # Use pin_memory only if num_workers > 0 and CUDA is available
    use_pin_memory = (num_workers > 0) and torch.cuda.is_available()

    if weighted and shuffle:
        print(f"[DataLoader] Creating weighted sampler for {len(ds)} samples...")
        labels = ds.df[label_cols].values
        cls = labels.argmax(1)
        counts = np.bincount(cls, minlength=len(label_cols)).astype(float)
        weights = (counts.sum() / np.maximum(counts, 1.0))
        sample_weights = weights[cls]
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        print(f"[DataLoader] Weighted sampler created. Class counts: {counts}")
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=use_pin_memory)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=use_pin_memory)

    print(f"[DataLoader] Created loader: batch_size={batch_size}, num_workers={num_workers}, pin_memory={use_pin_memory}")
    return ds, loader
