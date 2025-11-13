#!/usr/bin/env python3
"""
快速生成测试预测脚本
为已训练的 5 个 EfficientNet-V2-L 模型生成测试预测
"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from predict import predict_and_save

# 配置
OUTPUT_DIR = Path('outputs/breakthrough_20251113_004854')
LAYER1_DIR = OUTPUT_DIR / 'layer1' / 'efficientnet_v2_l'
TEST_PREDS_DIR = OUTPUT_DIR / 'layer1_test_predictions'
TEST_PREDS_DIR.mkdir(exist_ok=True)

# 模型配置
IMG_SIZE = 384
BATCH_SIZE = 32

print("=" * 80)
print("🚀 开始生成测试预测")
print("=" * 80)

# 为每个 fold 生成预测
for fold in range(5):
    checkpoint = LAYER1_DIR / f'fold{fold}' / 'best.pt'

    if not checkpoint.exists():
        print(f"❌ Fold {fold}: 检查点不存在 - {checkpoint}")
        continue

    output_file = TEST_PREDS_DIR / f'efficientnet_v2_l_fold{fold}_test_pred.csv'

    print(f"\n📊 Fold {fold}:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output: {output_file}")

    try:
        predict_and_save(
            checkpoint=str(checkpoint),
            test_csv='data/test_data.csv',
            test_dir='test_images',
            output_csv=str(output_file),
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        print(f"✅ Fold {fold} 完成")
    except Exception as e:
        print(f"❌ Fold {fold} 失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("🎉 所有预测生成完成")
print("=" * 80)
