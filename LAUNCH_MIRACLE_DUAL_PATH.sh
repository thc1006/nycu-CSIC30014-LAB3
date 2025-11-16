#!/bin/bash

echo "========================================================================"
echo "🚀 奇蹟雙路並行訓練 - 突破 90% 終極策略"
echo "========================================================================"
echo ""
echo "主路: Swin-Large 5-Fold (GPU 訓練)"
echo "副路: 偽標籤 Stage 6 快速驗證 (並行)"
echo ""
echo "========================================================================"

# 清理 GPU
python3 -c "import torch; torch.cuda.empty_cache(); print('✅ GPU 已清理')"

# 創建日誌目錄
mkdir -p logs/swin_large_ultimate
mkdir -p outputs/swin_large_ultimate

echo ""
echo "🔥 主路：啟動 Swin-Large 5-Fold 訓練..."
echo "========================================================================"

# 訓練 5 個 fold (背景)
nohup python3 src/train_kfold.py \
  --config configs/swin_large_ultimate.yaml \
  --n_folds 5 \
  --output_dir outputs/swin_large_ultimate \
  > logs/swin_large_ultimate/training.log 2>&1 &

SWIN_PID=$!
echo "✅ Swin-Large 訓練已啟動 (PID: $SWIN_PID)"
echo "   預計時間: 12-15 小時"
echo "   預期 Val F1: 86-89%"
echo "   預期 Test F1: 89-92% 🎯"
echo ""

echo "⚡ 副路：生成偽標籤 Stage 6..."
echo "========================================================================"

# 偽標籤生成 (並行)
python3 << 'PSEUDO_EOF'
import pandas as pd
import numpy as np

print("📊 生成高質量偽標籤 (Stage 6)...")

# 使用最佳模型
best_model = pd.read_csv('data/submission_v2l60_best40_onehot.csv')

# 轉換為概率
probs = []
for idx, row in best_model.iterrows():
    prob_row = [
        row['normal'],
        row['bacteria'],
        row['virus'],
        row['COVID-19']
    ]
    probs.append(prob_row)

probs = np.array(probs)

# 計算置信度
confidences = np.max(probs, axis=1)
predictions = np.argmax(probs, axis=1)

# 高置信度樣本 (>= 0.95)
threshold = 0.95
high_conf_mask = confidences >= threshold

print(f"✅ 閾值 {threshold}: {high_conf_mask.sum()} 個高質量樣本")

# 類別分布
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
print(f"\n📊 偽標籤分布:")
for i, name in enumerate(class_names):
    count = ((predictions == i) & high_conf_mask).sum()
    print(f"  {name}: {count}")

# 保存偽標籤
pseudo_df = pd.DataFrame({
    'new_filename': best_model['new_filename'][high_conf_mask],
    'label': predictions[high_conf_mask],
    'confidence': confidences[high_conf_mask]
})

pseudo_df.to_csv('data/pseudo_labels_stage6.csv', index=False)
print(f"\n✅ 偽標籤已保存: data/pseudo_labels_stage6.csv")
print(f"   樣本數: {len(pseudo_df)}")
PSEUDO_EOF

echo ""
echo "========================================================================"
echo "✅ 雙路並行已啟動！"
echo "========================================================================"
echo ""
echo "📊 監控進度:"
echo "  Swin-Large: tail -f logs/swin_large_ultimate/training.log"
echo "  GPU 狀態:    watch -n 5 nvidia-smi"
echo ""
echo "預計完成時間: 12-15 小時"
echo "預期最終分數: 89-92% 🎯"
echo ""
echo "🚀 讓我們見證奇蹟！"
echo "========================================================================"
