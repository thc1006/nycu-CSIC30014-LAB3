#!/usr/bin/env python3
"""
醫學影像預處理模組
針對胸部X光影像優化，特別增強COVID-19的低對比度特徵
"""
import cv2
import numpy as np
from PIL import Image


class MedicalImagePreprocessor:
    """醫學影像預處理器"""

    def __init__(
        self,
        apply_clahe=True,
        clahe_clip_limit=2.5,
        clahe_tile_size=8,
        apply_unsharp=True,
        unsharp_sigma=1.5,
        unsharp_amount=1.2,
        apply_lung_enhance=False,
    ):
        """
        Args:
            apply_clahe: 是否應用 CLAHE 對比度增強
            clahe_clip_limit: CLAHE 限制閾值 (2.0-3.0 適合醫學影像)
            clahe_tile_size: CLAHE 局部窗口大小
            apply_unsharp: 是否應用 Unsharp Masking 銳化
            unsharp_sigma: 高斯模糊的 sigma 值
            unsharp_amount: 銳化強度
            apply_lung_enhance: 是否應用肺部區域增強 (實驗性)
        """
        self.apply_clahe = apply_clahe
        self.apply_unsharp = apply_unsharp
        self.apply_lung_enhance = apply_lung_enhance

        # CLAHE 參數
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )

        # Unsharp Masking 參數
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_amount = unsharp_amount

    def clahe_enhance(self, img_array):
        """CLAHE 對比度限制自適應直方圖均衡化"""
        # 轉換為 uint8
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        # 應用 CLAHE
        enhanced = self.clahe.apply(img_array)

        return enhanced

    def unsharp_masking(self, img_array):
        """Unsharp Masking 銳化"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(
            img_array,
            (0, 0),
            self.unsharp_sigma
        )

        # Unsharp mask = 原圖 + amount * (原圖 - 模糊圖)
        sharpened = cv2.addWeighted(
            img_array, 1.0 + self.unsharp_amount,
            blurred, -self.unsharp_amount,
            0
        )

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def lung_region_enhance(self, img_array):
        """
        肺部區域增強（簡單版本）
        使用閾值分割來增強肺野區域
        """
        # 簡單閾值分割（背景通常較暗）
        _, mask = cv2.threshold(img_array, 20, 255, cv2.THRESH_BINARY)

        # 形態學操作去除噪點
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 對肺部區域應用更強的對比度增強
        lung_enhanced = self.clahe.apply(img_array)

        # 混合原圖和增強圖
        mask_float = mask.astype(np.float32) / 255.0
        result = (
            img_array.astype(np.float32) * (1 - mask_float) +
            lung_enhanced.astype(np.float32) * mask_float
        )

        return result.astype(np.uint8)

    def process(self, img):
        """
        處理單張影像

        Args:
            img: PIL Image 或 numpy array

        Returns:
            處理後的 PIL Image
        """
        # 轉換為 numpy array
        if isinstance(img, Image.Image):
            img_array = np.array(img.convert('L'))
        else:
            img_array = img

        # 確保是灰階
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 確保是 uint8
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        # 1. CLAHE 對比度增強
        if self.apply_clahe:
            img_array = self.clahe_enhance(img_array)

        # 2. Unsharp Masking 銳化
        if self.apply_unsharp:
            img_array = self.unsharp_masking(img_array)

        # 3. 肺部區域增強（實驗性，可選）
        if self.apply_lung_enhance:
            img_array = self.lung_region_enhance(img_array)

        # 轉回 PIL Image
        return Image.fromarray(img_array)

    def __call__(self, img):
        """支持直接調用"""
        return self.process(img)


def create_medical_preprocessor(preset='default'):
    """
    創建預配置的預處理器

    Args:
        preset: 預設配置
            - 'default': 標準配置
            - 'aggressive': 激進增強 (for low contrast images)
            - 'conservative': 保守增強
            - 'none': 不增強

    Returns:
        MedicalImagePreprocessor
    """
    presets = {
        'default': {
            'apply_clahe': True,
            'clahe_clip_limit': 2.5,
            'clahe_tile_size': 8,
            'apply_unsharp': True,
            'unsharp_sigma': 1.5,
            'unsharp_amount': 1.2,
            'apply_lung_enhance': False,
        },
        'aggressive': {
            'apply_clahe': True,
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': 8,
            'apply_unsharp': True,
            'unsharp_sigma': 2.0,
            'unsharp_amount': 1.5,
            'apply_lung_enhance': True,
        },
        'conservative': {
            'apply_clahe': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'apply_unsharp': False,
            'unsharp_sigma': 1.0,
            'unsharp_amount': 1.0,
            'apply_lung_enhance': False,
        },
        'none': {
            'apply_clahe': False,
            'apply_unsharp': False,
            'apply_lung_enhance': False,
        }
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    return MedicalImagePreprocessor(**presets[preset])


if __name__ == '__main__':
    """測試預處理效果"""
    import sys
    from pathlib import Path

    # 測試 COVID-19 影像
    test_images = [
        'train_images/0.jpg',
        'train_images/30.jpeg',
        'val_images/27.jpeg',
    ]

    print("=" * 80)
    print("測試醫學影像預處理")
    print("=" * 80)

    # 創建預處理器
    preprocessor = create_medical_preprocessor('default')

    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"⚠️  {img_path} 不存在")
            continue

        # 讀取影像
        img = Image.open(img_path)

        # 預處理
        processed = preprocessor.process(img)

        # 儲存結果
        output_dir = Path('outputs/preprocessing_test')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"processed_{Path(img_path).name}"
        processed.save(output_path)

        print(f"✅ {img_path} → {output_path}")

    print(f"\n✅ 預處理測試完成！")
    print(f"查看結果: outputs/preprocessing_test/")
