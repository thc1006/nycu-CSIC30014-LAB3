"""
GPU-cached dataset with PyTorch native GPU augmentation
Preloads all images to GPU memory for faster training
No external dependencies required (no kornia)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from torchvision import transforms as T


class GPUCachedDataset(torch.utils.data.Dataset):
    """Dataset that preloads all images to GPU memory"""

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        file_col: str,
        label_cols: List[str],
        img_size: int = 224,
        augment: bool = False,
        device: str = 'cuda'
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.file_col = file_col
        self.label_cols = label_cols
        self.img_size = img_size
        self.augment = augment
        self.device = torch.device(device)

        print(f"[GPUCachedDataset] Loading {len(self.df)} images to GPU...")

        # Preload all images to GPU
        self.images_gpu = []
        self.labels = []

        # Simple resize transform (CPU side)
        self.resize = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        for idx in range(len(self.df)):
            # Load image
            fname = self.df.iloc[idx][file_col]
            img_path = self.images_dir / fname

            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.resize(img)  # [3, H, W]

                # Move to GPU immediately
                img_gpu = img_tensor.to(self.device)
                self.images_gpu.append(img_gpu)

                # Labels
                label = torch.tensor(
                    self.df.iloc[idx][label_cols].values.astype(np.float32)
                ).argmax()
                self.labels.append(label)

                if (idx + 1) % 500 == 0:
                    print(f"  Loaded {idx + 1}/{len(self.df)} images...")

            except Exception as e:
                print(f"[WARNING] Failed to load {img_path}: {e}")
                continue

        # Stack all images into single tensor (saves memory)
        self.images_gpu = torch.stack(self.images_gpu)  # [N, 3, H, W]
        self.labels = torch.tensor(self.labels, device=self.device)

        gpu_memory_gb = self.images_gpu.element_size() * self.images_gpu.nelement() / (1024**3)
        print(f"[GPUCachedDataset] ✅ Loaded {len(self.images_gpu)} images")
        print(f"[GPUCachedDataset] 📊 GPU memory used: {gpu_memory_gb:.2f} GB")

        # Store augmentation flag
        self.use_augment = augment

        # Normalization tensors (always needed)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        if augment:
            print(f"[GPUCachedDataset] ✅ GPU augmentation enabled (PyTorch native)")
        else:
            print(f"[GPUCachedDataset] ℹ️  Validation mode (no augmentation)")

    def __len__(self):
        return len(self.images_gpu)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return augmented image (already on GPU)"""
        img = self.images_gpu[idx].clone()  # Clone to avoid modifying cached data
        label = self.labels[idx]

        if self.use_augment:
            # Simple GPU augmentations using PyTorch
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])

            # Random color jitter (brightness/contrast)
            if torch.rand(1).item() < 0.5:
                # Brightness
                brightness_factor = 0.85 + torch.rand(1).item() * 0.3  # 0.85-1.15
                img = img * brightness_factor
                # Contrast
                contrast_factor = 0.75 + torch.rand(1).item() * 0.5  # 0.75-1.25
                img = (img - img.mean()) * contrast_factor + img.mean()
                img = torch.clamp(img, 0, 1)

        # Normalize (always)
        img = (img - self.norm_mean) / self.norm_std

        fname = self.df.iloc[idx][self.file_col]

        return img, label, fname


def make_gpu_cached_loader(
    csv_path: str,
    images_dir: str,
    file_col: str,
    label_cols: List[str],
    img_size: int,
    batch_size: int,
    num_workers: int = 0,  # Must be 0 for GPU cached dataset
    augment: bool = False,
    shuffle: bool = True,
    device: str = 'cuda'
):
    """Create DataLoader with GPU-cached dataset"""

    dataset = GPUCachedDataset(
        csv_path, images_dir, file_col, label_cols,
        img_size, augment, device
    )

    # Important: num_workers MUST be 0 since data is already on GPU
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Data already on GPU
        pin_memory=False,  # Not needed
    )

    print(f"[DataLoader] Created GPU-cached loader: batch_size={batch_size}, shuffle={shuffle}")

    return dataset, loader


if __name__ == "__main__":
    # Test
    print("Testing GPU-cached dataset...")

    ds, loader = make_gpu_cached_loader(
        csv_path="data/train_data.csv",
        images_dir="train_images",
        file_col="new_filename",
        label_cols=['normal', 'bacteria', 'virus', 'COVID-19'],
        img_size=224,
        batch_size=32,
        augment=True
    )

    # Test one batch
    for imgs, labels, fnames in loader:
        print(f"\nBatch shape: {imgs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Device: {imgs.device}")
        print(f"✅ GPU-cached dataset works!")
        break
