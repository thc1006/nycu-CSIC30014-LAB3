#!/usr/bin/env python3
"""
超級 TTA 增強：多尺度 + 翻轉 + 五點裁剪
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_breakthrough import build_model

class TTATestDataset(Dataset):
    """支持 TTA 的測試數據集"""
    def __init__(self, csv_path, images_dir, img_size=384, tta_mode='center'):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_size = img_size
        self.tta_mode = tta_mode

        # 基礎標準化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def _apply_tta(self, image):
        """應用 TTA 變換"""
        # Resize 到稍大尺寸用於裁剪
        resize_size = int(self.img_size * 1.15)
        image = TF.resize(image, (resize_size, resize_size))

        if self.tta_mode == 'center':
            # 中心裁剪
            image = TF.center_crop(image, (self.img_size, self.img_size))

        elif self.tta_mode == 'hflip':
            # 水平翻轉 + 中心裁剪
            image = TF.hflip(image)
            image = TF.center_crop(image, (self.img_size, self.img_size))

        elif self.tta_mode == 'top_left':
            # 左上角裁剪
            image = TF.crop(image, 0, 0, self.img_size, self.img_size)

        elif self.tta_mode == 'top_right':
            # 右上角裁剪
            image = TF.crop(image, 0, resize_size - self.img_size,
                          self.img_size, self.img_size)

        elif self.tta_mode == 'bottom_left':
            # 左下角裁剪
            image = TF.crop(image, resize_size - self.img_size, 0,
                          self.img_size, self.img_size)

        elif self.tta_mode == 'bottom_right':
            # 右下角裁剪
            image = TF.crop(image, resize_size - self.img_size,
                          resize_size - self.img_size,
                          self.img_size, self.img_size)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['new_filename'])
        image = Image.open(img_path).convert('RGB')

        # 應用 TTA
        image = self._apply_tta(image)

        # 轉換為 Tensor 並標準化
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return image, row['new_filename']

@torch.no_grad()
def generate_tta_predictions(model_path, model_arch, img_size=384, tta_modes=None):
    """生成帶 TTA 的預測"""
    if tta_modes is None:
        tta_modes = ['center', 'hflip', 'top_left', 'top_right', 'bottom_left', 'bottom_right']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入模型
    model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=img_size)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # 收集所有 TTA 預測
    all_tta_probs = []

    for tta_mode in tta_modes:
        test_dataset = TTATestDataset('data/test_data.csv', 'test_images',
                                     img_size, tta_mode=tta_mode)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                num_workers=8, pin_memory=True)

        mode_probs = []
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            mode_probs.append(probs)

        mode_probs = np.concatenate(mode_probs, axis=0)
        all_tta_probs.append(mode_probs)

    # 平均所有 TTA 預測
    avg_probs = np.mean(all_tta_probs, axis=0)

    del model
    torch.cuda.empty_cache()

    return avg_probs

def main():
    print("="*80)
    print("超級 TTA 增強 - 生成高質量測試預測")
    print("="*80)

    # TTA 模式
    tta_modes = ['center', 'hflip', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
    print(f"\nTTA 模式: {len(tta_modes)}x 增強")
    print(f"  {', '.join(tta_modes)}")

    # 模型配置
    configs = [
        {
            'name': 'efficientnet_v2_l',
            'arch': 'efficientnet_v2_l',
            'base_dir': 'outputs/breakthrough_20251113_004854/layer1/efficientnet_v2_l',
            'img_size': 384,
            'folds': 5
        },
        {
            'name': 'swin_large',
            'arch': 'swin_large_patch4_window12_384',
            'base_dir': 'outputs/breakthrough_20251113_004854/layer1/swin_large',
            'img_size': 384,
            'folds': 5
        }
    ]

    test_df = pd.read_csv('data/test_data.csv')
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    os.makedirs('data/test_predictions_tta', exist_ok=True)

    for config in configs:
        print(f"\n{'='*80}")
        print(f"處理: {config['name']} (6x TTA)")
        print(f"{'='*80}")

        for fold in range(config['folds']):
            model_path = Path(config['base_dir']) / f"fold{fold}" / "best.pt"

            if not model_path.exists():
                print(f"⚠️ 模型不存在: {model_path}")
                continue

            print(f"\n[{fold+1}/{config['folds']}] 載入模型: {model_path.name}")

            # 生成 TTA 預測
            probs = generate_tta_predictions(
                str(model_path),
                config['arch'],
                config['img_size'],
                tta_modes=tta_modes
            )

            # 保存預測
            output_df = test_df[['new_filename']].copy()
            for i, class_name in enumerate(class_names):
                output_df[class_name] = probs[:, i]

            output_path = f"data/test_predictions_tta/{config['name']}_fold{fold}_tta6x.csv"
            output_df.to_csv(output_path, index=False)

            avg_conf = np.max(probs, axis=1).mean()
            print(f"✅ 已保存: {output_path}")
            print(f"   平均置信度: {avg_conf:.4f} (6x TTA)")

    print(f"\n{'='*80}")
    print("✅ 所有 TTA 預測已生成！")
    print(f"{'='*80}")

    # 統計
    pred_files = list(Path('data/test_predictions_tta').glob('*_tta6x.csv'))
    print(f"\n總共生成: {len(pred_files)} 個 TTA 增強預測文件")
    print("\n下一步: 使用這些 TTA 預測進行 Stacking 集成")

if __name__ == '__main__':
    main()
