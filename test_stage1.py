"""
Quick test script to verify Stage 1 components work correctly.
Run this before starting full training.
"""
import torch
import sys
from src.utils import load_config, seed_everything
from src.train_v2 import build_model
from src.losses import ImprovedFocalLoss
from src.aug import mixup_data, cutmix_data
from src.data import make_loader

def test_stage1():
    print("="*60)
    print("Stage 1 Component Test")
    print("="*60)

    # 1. Test config loading
    print("\n[1/6] Testing config loading...")
    try:
        cfg = load_config('configs/model_stage1.yaml')
        print(f"[OK] Config loaded: {cfg['model']['name']}, {cfg['model']['img_size']}px")
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        return False

    # 2. Test model building
    print("\n[2/6] Testing ConvNeXt-Base model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model('convnext_base', 4).to(device)
        print(f"[OK] Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

        # Test forward pass
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
        print(f"[OK] Forward pass works: input {dummy_input.shape} -> output {output.shape}")
    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")
        return False

    # 3. Test Improved Focal Loss
    print("\n[3/6] Testing Improved Focal Loss...")
    try:
        loss_fn = ImprovedFocalLoss(
            alpha=[1.0, 1.5, 2.0, 1.2],
            gamma=2.0,
            label_smoothing=0.1
        )
        dummy_logits = torch.randn(4, 4).to(device)
        dummy_targets = torch.tensor([0, 1, 2, 3]).to(device)
        loss = loss_fn(dummy_logits, dummy_targets)
        assert loss.item() > 0, "Loss should be positive"
        print(f"[OK] ImprovedFocalLoss works: loss={loss.item():.4f}")
    except Exception as e:
        print(f"[FAIL] Loss function test failed: {e}")
        return False

    # 4. Test Mixup/CutMix
    print("\n[4/6] Testing Mixup/CutMix augmentation...")
    try:
        x = torch.randn(4, 3, 512, 512).to(device)
        y = torch.tensor([0, 1, 2, 3]).to(device)

        # Test Mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, device=device)
        assert mixed_x.shape == x.shape, "Mixup should preserve shape"
        assert 0 <= lam <= 1, "Lambda should be between 0 and 1"
        print(f"[OK] Mixup works: lambda={lam:.3f}")

        # Test CutMix
        cutmix_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0, device=device)
        assert cutmix_x.shape == x.shape, "CutMix should preserve shape"
        print(f"[OK] CutMix works: lambda={lam:.3f}")
    except Exception as e:
        print(f"[FAIL] Mixup/CutMix test failed: {e}")
        return False

    # 5. Test data loader
    print("\n[5/6] Testing data loader with advanced augmentation...")
    try:
        data_cfg = cfg['data']
        train_cfg = cfg['train']
        mdl_cfg = cfg['model']

        aug_config = {
            'aug_rotation': 15,
            'aug_translate': 0.1,
            'aug_scale_min': 0.9,
            'aug_scale_max': 1.1,
            'aug_shear': 10,
            'random_erasing_prob': 0.3,
        }

        train_ds, train_loader = make_loader(
            data_cfg["train_csv"],
            data_cfg["images_dir_train"],
            data_cfg["file_col"],
            data_cfg["label_cols"],
            mdl_cfg["img_size"],
            batch_size=2,  # Small batch for testing
            num_workers=0,  # No multiprocessing for test
            augment=True,
            shuffle=True,
            weighted=False,
            advanced_aug=True,
            aug_config=aug_config
        )

        # Get one batch
        imgs, labels, fnames = next(iter(train_loader))
        assert imgs.shape[0] == 2, "Batch size should be 2"
        assert imgs.shape[1:] == (3, 512, 512), f"Expected (3, 512, 512), got {imgs.shape[1:]}"
        print(f"[OK] DataLoader works: batch shape {imgs.shape}, labels {labels}")
    except Exception as e:
        print(f"[FAIL] DataLoader test failed: {e}")
        return False

    # 6. Test TTA
    print("\n[6/6] Testing Test-Time Augmentation...")
    try:
        from src.tta_predict import TTAWrapper

        tta_model = TTAWrapper(model).to(device)
        dummy_input = torch.randn(2, 3, 512, 512).to(device)

        with torch.no_grad():
            tta_output = tta_model(dummy_input)

        assert tta_output.shape == (2, 4), f"Expected shape (2, 4), got {tta_output.shape}"
        print(f"[OK] TTA works: input {dummy_input.shape} -> output {tta_output.shape}")
    except Exception as e:
        print(f"[FAIL] TTA test failed: {e}")
        return False

    print("\n" + "="*60)
    print("[SUCCESS] All tests passed! Ready for Stage 1 training.")
    print("="*60)
    print("\nTo start training:")
    print("  python -m src.train_v2 --config configs/model_stage1.yaml")
    print("\nExpected training time:")
    print("  RTX 3050: ~4-5 hours (30 epochs)")
    print("  A100: ~2 hours (30 epochs)")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_stage1()
    sys.exit(0 if success else 1)
