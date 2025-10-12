import torch
import numpy as np
import torchvision.transforms as T

def build_transforms(img_size=224, augment=True, advanced_aug=False, aug_config=None):
    """
    Build image transformation pipeline.

    Args:
        img_size: Target image size
        augment: Whether to apply augmentations
        advanced_aug: Whether to use advanced augmentations
        aug_config: Dict with augmentation parameters
    """
    if aug_config is None:
        aug_config = {}

    if augment:
        if advanced_aug:
            # Advanced augmentation for MEDICAL IMAGES (chest X-rays)
            # Optimized for 90%+ accuracy target
            rotation = aug_config.get('aug_rotation', 10)  # Reduced for medical images
            translate = aug_config.get('aug_translate', 0.08)
            scale_min = aug_config.get('aug_scale_min', 0.92)
            scale_max = aug_config.get('aug_scale_max', 1.08)
            shear = aug_config.get('aug_shear', 5)  # Reduced shear
            erase_prob = aug_config.get('random_erasing_prob', 0.25)

            return T.Compose([
                T.Resize((img_size, img_size)),

                # Geometric augmentations (conservative for medical)
                T.RandomHorizontalFlip(p=0.5),  # X-rays can be flipped
                T.RandomRotation(degrees=rotation),
                T.RandomAffine(
                    degrees=0,
                    translate=(translate, translate),
                    scale=(scale_min, scale_max),
                    shear=shear
                ),

                # MEDICAL-SPECIFIC: Contrast/brightness (simulate different X-ray exposures)
                T.RandomAutocontrast(p=0.3),  # Auto-adjust contrast
                T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),  # Sharpness variation

                # Standard color jitter (more conservative for medical)
                T.ColorJitter(
                    brightness=0.15,  # Reduced from 0.2
                    contrast=0.25,    # Increased - important for X-rays
                    saturation=0.05,  # Reduced - X-rays mostly grayscale
                    hue=0.02          # Reduced - minimal hue variation
                ),

                T.ToTensor(),

                # Random erasing (simulate artifacts)
                T.RandomErasing(p=erase_prob, scale=(0.02, 0.12)),

                # Normalization (ImageNet stats work well for pretrained models)
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Standard augmentation
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply Mixup augmentation.

    Args:
        x: Input images [B, C, H, W]
        y: Target labels [B] (class indices)
        alpha: Mixup interpolation strength
        device: Device for computation

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply CutMix augmentation.

    Args:
        x: Input images [B, C, H, W]
        y: Target labels [B] (class indices)
        alpha: CutMix interpolation strength
        device: Device for computation

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    _, _, W, H = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam
