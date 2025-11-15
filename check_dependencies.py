#!/usr/bin/env python3
"""
🔍 依賴性檢查腳本 - Google Colab & 新機器環境驗證
"""

import sys
import importlib

def check_package(name, min_version=None, import_as=None):
    """檢查套件是否安裝並驗證版本"""
    try:
        # 特殊處理某些套件的導入名稱
        import_name = import_as if import_as else name
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')

        status = "✅"
        if min_version and version != 'unknown':
            # 簡單版本比較（避免依賴 packaging）
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    status = "⚠️ (版本過舊)"
            except ImportError:
                # 如果沒有 packaging，跳過版本檢查
                pass

        print(f"{status} {name:20} {version:15} (最低: {min_version or 'N/A'})")
        return True
    except ImportError:
        print(f"❌ {name:20} NOT INSTALLED")
        return False

def main():
    print("=" * 70)
    print("🔍 胸部 X 光分類項目 - 依賴性檢查")
    print("=" * 70)
    print()

    # Python 版本檢查
    print("🐍 Python 版本:")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info.major == 3 and sys.version_info.minor >= 10:
        print(f"✅ Python {py_version} (符合要求 ≥ 3.10)")
    else:
        print(f"⚠️ Python {py_version} (建議 ≥ 3.10)")
    print()

    # 核心套件檢查
    print("📦 核心套件:")
    print(f"{'套件':20} {'當前版本':15} {'最低要求':15}")
    print("-" * 70)

    packages = [
        ("torch", "2.0.0", None),
        ("torchvision", "0.15.0", None),
        ("timm", "0.9.0", None),
        ("pandas", "1.5.0", None),
        ("numpy", "1.24.0", None),
        ("PIL", "9.0.0", "PIL"),
        ("sklearn", "1.2.0", "sklearn"),
        ("tqdm", "4.60.0", None),
        ("yaml", "6.0", "yaml"),
    ]

    all_ok = True
    for pkg_info in packages:
        if len(pkg_info) == 3:
            pkg, min_ver, import_as = pkg_info
        else:
            pkg, min_ver = pkg_info
            import_as = None

        if not check_package(pkg, min_ver, import_as):
            all_ok = False

    print()

    # CUDA 檢查
    print("🎮 GPU & CUDA:")
    print("-" * 70)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            print(f"✅ GPU: {gpu_name}")
            print(f"✅ CUDA version: {cuda_version}")
            print(f"✅ GPU memory: {gpu_memory:.2f} GB")

            # 推薦 batch size
            if gpu_memory > 35:
                batch_size = 24
                gpu_type = "A100"
            elif gpu_memory > 14:
                batch_size = 12
                gpu_type = "T4/V100/RTX 4070 Ti"
            else:
                batch_size = 8
                gpu_type = "Small GPU"

            print(f"✅ GPU 類型: ~{gpu_type}")
            print(f"✅ 建議 Batch Size: {batch_size}")
        else:
            print("⚠️ CUDA not available (將使用 CPU 訓練，速度會非常慢)")
            all_ok = False
    except Exception as e:
        print(f"❌ GPU/CUDA 檢查失敗: {e}")
        all_ok = False

    print()

    # DINOv2 模型檢查
    print("🤖 DINOv2 模型可用性:")
    print("-" * 70)
    try:
        import timm
        dinov2_models = [m for m in timm.list_models() if 'dinov2' in m]
        if len(dinov2_models) > 0:
            print(f"✅ 找到 {len(dinov2_models)} 個 DINOv2 模型")
            print(f"   主要模型: vit_base_patch14_dinov2.lvd142m")

            # 測試載入模型
            try:
                model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)
                print(f"✅ 模型可以正常載入")
            except Exception as e:
                print(f"⚠️ 模型載入測試失敗: {e}")
        else:
            print("❌ 找不到 DINOv2 模型")
            print("   請升級 timm: pip install --upgrade timm")
            all_ok = False
    except Exception as e:
        print(f"❌ DINOv2 檢查失敗: {e}")
        all_ok = False

    print()
    print("=" * 70)

    # 最終結果
    if all_ok:
        print("✅ 所有依賴檢查通過！環境已準備好")
        print()
        print("📝 下一步:")
        print("   1. 下載 Kaggle 數據集 (如果還沒有)")
        print("   2. 檢查 Fold CSV 文件是否存在")
        print("   3. 開始訓練: python train_dinov2_breakthrough.py --fold 0 --epochs 35")
        print()
        return 0
    else:
        print("❌ 部分依賴缺失或配置不正確")
        print()
        print("🔧 修復建議:")
        print("   1. 安裝缺失的套件:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("      pip install timm pandas numpy Pillow tqdm scikit-learn pyyaml")
        print()
        print("   2. 如果 GPU 不可用，檢查:")
        print("      - NVIDIA 驅動是否安裝")
        print("      - CUDA toolkit 是否安裝")
        print("      - PyTorch 是否安裝了 CUDA 版本")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
