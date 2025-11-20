# -*- coding: utf-8 -*-
"""
Medical-specific data augmentation for COVID-19 X-ray classification

医学特定的数据增强，针对胸部X光肺炎分类优化
"""
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from PIL import Image


class MedicalXrayAugmentation:
    """
    医学影像特定的增强
    针对不同类别的特有模式
    """

    def __init__(self, class_name, img_size=224, advanced_aug=False):
        """
        Args:
            class_name: 'Normal', 'Bacteria', 'Virus', or 'COVID-19'
            img_size: Target image size
            advanced_aug: Whether to apply advanced augmentations
        """
        self.class_name = class_name
        self.img_size = img_size
        self.advanced_aug = advanced_aug

    def apply(self, image):
        """
        Apply class-specific augmentation
        """
        if self.class_name == 'COVID-19':
            return self.augment_covid19(image)
        elif self.class_name == 'Virus':
            return self.augment_virus(image)
        elif self.class_name == 'Bacteria':
            return self.augment_bacteria(image)
        else:  # Normal
            return self.augment_normal(image)

    def augment_covid19(self, image):
        """
        COVID-19 specific augmentation
        - Enhance lower lobe (lower lobe predilection)
        - Enhance peripheral distribution (GGO pattern)
        - Add subtle GGO effect (magnifying glass opacity)
        """
        image_np = np.array(image) if isinstance(image, Image.Image) else image

        # 1. Lower lobe emphasis (COVID-19 has lower lobe predilection)
        if np.random.rand() < 0.3:
            h = image_np.shape[0]
            lower_half = image_np[h//2:, :]
            # Slightly enhance lower lobe
            lower_half = np.clip(lower_half * 1.1, 0, 255)
            image_np[h//2:, :] = lower_half

        # 2. Peripheral enhancement (COVID-19 is peripheral/GGO)
        if np.random.rand() < 0.3:
            image_np = self._enhance_periphery(image_np, strength=0.2)

        # 3. Add GGO effect (subtle Gaussian blur for ground-glass effect)
        if np.random.rand() < 0.2:
            image_np = self._add_ggo_effect(image_np, strength=0.15)

        # 4. Subtle elastic transformation
        if np.random.rand() < 0.2:
            image_np = self._elastic_transform(image_np, alpha=20, sigma=2.5)

        # 5. Medical-safe rotation (±5 degrees only)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-5, 5)
            image_np = self._rotate(image_np, angle)

        # 6. Gentle contrast adjustment (to preserve GGO pattern)
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            image_np = self._adjust_contrast(image_np, factor)

        return Image.fromarray(image_np.astype(np.uint8))

    def augment_virus(self, image):
        """
        Virus specific augmentation
        - Enhance multifocal pattern
        - Enhance diffuse distribution
        """
        image_np = np.array(image) if isinstance(image, Image.Image) else image

        # 1. Multifocal pattern emphasis
        if np.random.rand() < 0.3:
            image_np = self._enhance_multifocal(image_np, strength=0.15)

        # 2. Diffuse distribution
        if np.random.rand() < 0.2:
            image_np = self._add_diffuse_effect(image_np, strength=0.1)

        # 3. Standard augmentations for Virus
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-8, 8)
            image_np = self._rotate(image_np, angle)

        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            image_np = self._adjust_contrast(image_np, factor)

        return Image.fromarray(image_np.astype(np.uint8))

    def augment_bacteria(self, image):
        """
        Bacteria specific augmentation
        - Emphasize lobar consolidation
        - Enhance consolidation pattern
        """
        image_np = np.array(image) if isinstance(image, Image.Image) else image

        # 1. Lobar pattern emphasis
        if np.random.rand() < 0.3:
            image_np = self._emphasize_lobar_segments(image_np, strength=0.15)

        # 2. Consolidation enhancement
        if np.random.rand() < 0.3:
            image_np = self._enhance_consolidation(image_np, strength=0.15)

        # 3. Standard augmentations for Bacteria
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-8, 8)
            image_np = self._rotate(image_np, angle)

        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.85, 1.25)
            image_np = self._adjust_contrast(image_np, factor)

        return Image.fromarray(image_np.astype(np.uint8))

    def augment_normal(self, image):
        """
        Normal lung augmentation
        - Minimal augmentation (just geometric)
        """
        image_np = np.array(image) if isinstance(image, Image.Image) else image

        # Minimal augmentation to preserve normal pattern
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-5, 5)
            image_np = self._rotate(image_np, angle)

        if np.random.rand() < 0.2:
            factor = np.random.uniform(0.95, 1.05)
            image_np = self._adjust_contrast(image_np, factor)

        return Image.fromarray(image_np.astype(np.uint8))

    # Helper methods for augmentation

    def _enhance_periphery(self, image, strength=0.2):
        """增强外周特征 (外周增强)"""
        h, w = image.shape[:2]
        mask = np.ones_like(image, dtype=float)

        # Create peripheral mask (edges are enhanced)
        for i in range(h):
            for j in range(w):
                dist_from_center = np.sqrt((i - h/2)**2 + (j - w/2)**2)
                max_dist = np.sqrt((h/2)**2 + (w/2)**2)
                if dist_from_center > 0.4 * max_dist:
                    mask[i, j] = 1 + strength

        return np.clip(image * mask, 0, 255)

    def _add_ggo_effect(self, image, strength=0.15):
        """添加磨玻璃样效果 (GGO效果)"""
        # Slight Gaussian blur to simulate GGO
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return np.clip(image * (1 - strength) + blurred * strength, 0, 255)

    def _enhance_multifocal(self, image, strength=0.15):
        """增强多灶性模式 (Virus特征)"""
        # Enhance local contrast for multifocal pattern
        h, w = image.shape[:2]
        enhanced = image.copy()

        # Add subtle local contrast enhancement
        for _ in range(3):
            y = np.random.randint(0, h - 32)
            x = np.random.randint(0, w - 32)
            patch = enhanced[y:y+32, x:x+32]
            patch = self._adjust_contrast(patch, 1 + strength)
            enhanced[y:y+32, x:x+32] = patch

        return enhanced

    def _add_diffuse_effect(self, image, strength=0.1):
        """添加弥漫性效果 (Virus特征)"""
        # Spread infiltrates across the image
        noise = np.random.normal(0, 2, image.shape) * strength
        return np.clip(image + noise, 0, 255)

    def _emphasize_lobar_segments(self, image, strength=0.15):
        """强调肺段分布 (Bacteria特征)"""
        h, w = image.shape[:2]

        # Randomly emphasize one or two lobar segments
        if np.random.rand() < 0.5:
            # Upper lobe
            image[:h//2, :] = np.clip(image[:h//2, :] * (1 + strength), 0, 255)
        else:
            # Lower lobe
            image[h//2:, :] = np.clip(image[h//2:, :] * (1 + strength), 0, 255)

        return image

    def _enhance_consolidation(self, image, strength=0.15):
        """增强实变效果 (Bacteria特征)"""
        # Enhance dense opacification
        return np.clip(image * (1 + strength), 0, 255)

    def _rotate(self, image, angle):
        """医学安全旋转 (Medical-safe rotation)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated

    def _adjust_contrast(self, image, factor):
        """调整对比度 (Contrast adjustment)"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255)

    def _elastic_transform(self, image, alpha=30, sigma=3):
        """弹性变形 (Elastic transformation)"""
        h, w = image.shape[:2]

        # Random displacement fields
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma

        # Gaussian blur the displacement
        dx = cv2.GaussianBlur(dx, (5, 5), sigma)
        dy = cv2.GaussianBlur(dy, (5, 5), sigma)

        # Scale
        dx = dx * alpha / np.max(np.abs(dx)) if np.max(np.abs(dx)) > 0 else dx
        dy = dy * alpha / np.max(np.abs(dy)) if np.max(np.abs(dy)) > 0 else dy

        # Apply transformation
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = y + dy, x + dx

        # Clip indices
        indices[0] = np.clip(indices[0], 0, h - 1)
        indices[1] = np.clip(indices[1], 0, w - 1)

        # Map coordinates to nearest integer
        indices_int = (np.round(indices[0]).astype(int), np.round(indices[1]).astype(int))

        return image[indices_int]


def create_medical_augmentation_pipeline(class_name, img_size=224, augment=True):
    """
    Create augmentation pipeline for medical images
    """
    if not augment:
        # No augmentation, just resize and normalize
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create medical augmentation
    medical_aug = MedicalXrayAugmentation(class_name, img_size=img_size, advanced_aug=True)

    # Combine with standard transforms
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Lambda(lambda x: medical_aug.apply(x)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
