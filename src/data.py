import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from .aug import build_transforms

class CSVDataset(Dataset):
    def __init__(self, csv_path, images_dir, file_col, label_cols, img_size=224, augment=True, advanced_aug=False, aug_config=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.file_col = file_col
        self.label_cols = list(label_cols)
        self.img_size = img_size
        self.transforms = build_transforms(img_size, augment, advanced_aug, aug_config)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row[self.file_col])
        path = os.path.join(self.images_dir, fname)
        image = Image.open(path).convert("RGB")
        x = self.transforms(image)

        # one-hot -> class index (argmax). If labels are NaN (test), return -1
        if self.label_cols and self.label_cols[0] in row.index and not np.any(pd.isna(row[self.label_cols])):
            y_vec = row[self.label_cols].values.astype(float)
            y_idx = int(np.argmax(y_vec))
        else:
            y_idx = -1
        return x, y_idx, fname

def make_loader(csv_path, images_dir, file_col, label_cols, img_size, batch_size, num_workers, augment, shuffle=True, weighted=False, advanced_aug=False, aug_config=None):
    ds = CSVDataset(csv_path, images_dir, file_col, label_cols, img_size, augment, advanced_aug, aug_config)
    if weighted and shuffle:
        labels = ds.df[label_cols].values
        cls = labels.argmax(1)
        counts = np.bincount(cls, minlength=len(label_cols)).astype(float)
        weights = (counts.sum() / np.maximum(counts, 1.0))
        sample_weights = weights[cls]
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return ds, loader
