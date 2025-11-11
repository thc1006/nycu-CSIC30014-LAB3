#!/usr/bin/env python3
"""
融合多樣化模型：RegNet-Y-3.2GF + EfficientNet-V2-S
"""
import os, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from PIL import Image
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, image_dir, img_size):
        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])
        self.image_dir = image_dir
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        return self.transform(img), fname

def load_model_checkpoint(model, ckpt_path, device):
    """加載模型檢查點"""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    return model

@torch.no_grad()
def predict(model, loader, device):
    """生成預測概率"""
    all_probs = []
    for imgs, _ in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型配置
    models_config = [
        {
            'name': 'RegNet-Y-3.2GF',
            'ckpt': 'outputs/diverse_model2/best.pt',
            'img_size': 384,
            'builder': lambda: models.regnet_y_3_2gf(weights=None)
        },
        {
            'name': 'EfficientNet-V2-S',
            'ckpt': 'outputs/diverse_model3/best.pt',
            'img_size': 320,
            'builder': lambda: models.efficientnet_v2_s(weights=None)
        }
    ]

    # 準備測試集
    test_dir = 'test_images'

    # 收集每個模型的預測
    all_predictions = []
    filenames = None

    for config in models_config:
        print(f"\n{'='*60}")
        print(f"Processing {config['name']}")
        print(f"{'='*60}")

        # 構建模型
        model = config['builder']()
        # 修改最後一層為 4 分類
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, torch.nn.Sequential):
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 4)
            else:
                model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

        model = model.to(device)

        # 加載檢查點
        load_model_checkpoint(model, config['ckpt'], device)
        print(f"Loaded checkpoint: {config['ckpt']}")

        # 創建數據集和加載器
        dataset = TestDataset(test_dir, config['img_size'])
        if filenames is None:
            filenames = dataset.files
        loader = DataLoader(dataset, batch_size=48, shuffle=False, num_workers=8, pin_memory=True)

        # 預測
        probs = predict(model, loader, device)
        all_predictions.append(probs)
        print(f"{config['name']} predictions shape: {probs.shape}")

        # 清理 GPU 內存
        del model
        torch.cuda.empty_cache()

    # 融合預測（平均）
    print(f"\n{'='*60}")
    print("Ensembling predictions...")
    print(f"{'='*60}")
    ensemble_probs = np.mean(all_predictions, axis=0)
    pred_labels = ensemble_probs.argmax(axis=1)

    # 轉換為 one-hot
    one_hot = np.zeros_like(ensemble_probs)
    one_hot[np.arange(len(one_hot)), pred_labels] = 1

    # 創建提交文件
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    df = pd.DataFrame(one_hot, columns=label_cols)
    df.insert(0, 'new_filename', filenames)

    # 保存
    output_path = 'data/submission_diverse_ensemble.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Ensemble submission saved to: {output_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    for i, col in enumerate(label_cols):
        count = (pred_labels == i).sum()
        print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
