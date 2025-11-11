"""
快速測試訓練初始化 - 診斷卡住問題
"""
import sys
import torch
import yaml

print("[1/10] Importing modules...")
from src.config import load_config
from src.data import make_loader
from src.model import build_model

print("[2/10] Loading config...")
cfg = load_config("configs/exp3_resnet34_long.yaml")
print(f"  Config loaded: {cfg['model']['name']} @ {cfg['model']['img_size']}px")

print("[3/10] Setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

print("[4/10] Building model...")
model = build_model(cfg["model"]["name"], num_classes=4)
model = model.to(device)
print(f"  Model created and moved to {device}")

print("[5/10] Setting up data augmentation config...")
aug_config = {
    'rotation': cfg['train'].get('aug_rotation', 15),
    'translate': cfg['train'].get('aug_translate', 0.1),
    'scale_min': cfg['train'].get('aug_scale_min', 0.9),
    'scale_max': cfg['train'].get('aug_scale_max', 1.1),
    'shear': cfg['train'].get('aug_shear', 10),
}
print(f"  Aug config: {aug_config}")

print("[6/10] Creating train dataset...")
train_ds, train_loader = make_loader(
    csv_path=cfg['data']['train_csv'],
    images_dir=cfg['data']['images_dir_train'],
    file_col=cfg['data']['file_col'],
    label_cols=cfg['data']['label_cols'],
    img_size=cfg['model']['img_size'],
    batch_size=cfg['train']['batch_size'],
    num_workers=cfg['train']['num_workers'],
    augment=True,
    shuffle=True,
    weighted=cfg['train'].get('use_weighted_sampler', False),
    advanced_aug=cfg['train'].get('advanced_aug', False),
    aug_config=aug_config
)
print(f"  Train loader created: {len(train_ds)} samples, {len(train_loader)} batches")

print("[7/10] Creating val dataset...")
val_ds, val_loader = make_loader(
    csv_path=cfg['data']['val_csv'],
    images_dir=cfg['data']['images_dir_val'],
    file_col=cfg['data']['file_col'],
    label_cols=cfg['data']['label_cols'],
    img_size=cfg['model']['img_size'],
    batch_size=cfg['train']['batch_size'],
    num_workers=cfg['train']['num_workers'],
    augment=False,
    shuffle=False,
    weighted=False
)
print(f"  Val loader created: {len(val_ds)} samples, {len(val_loader)} batches")

print("[8/10] Testing first batch from train_loader...")
try:
    for imgs, targets, fnames in train_loader:
        print(f"  First batch loaded successfully!")
        print(f"    imgs shape: {imgs.shape}")
        print(f"    targets shape: {targets.shape}")
        print(f"    fnames count: {len(fnames)}")
        break
except Exception as e:
    print(f"  ERROR loading first batch: {e}")
    sys.exit(1)

print("[9/10] Testing forward pass...")
try:
    imgs = imgs.to(device)
    with torch.no_grad():
        outputs = model(imgs)
    print(f"  Forward pass successful! Output shape: {outputs.shape}")
except Exception as e:
    print(f"  ERROR in forward pass: {e}")
    sys.exit(1)

print("[10/10] All tests passed!")
print("\n" + "="*80)
print("SUCCESS: Training initialization should work fine.")
print("="*80)
