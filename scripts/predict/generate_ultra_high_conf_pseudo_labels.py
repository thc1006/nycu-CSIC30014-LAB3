#!/usr/bin/env python3
"""
生成超高置信度測試集偽標籤
策略：三個最佳模型投票 + 置信度 ≥0.99
"""

import pandas as pd
import numpy as np

def load_predictions_as_probs(csv_path):
    """載入預測並轉換為概率（如果是 one-hot）"""
    df = pd.read_csv(csv_path)
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    probs = df[class_cols].values

    # 檢查是否為 one-hot (所有值都是 0 或 1)
    is_onehot = np.all((probs == 0) | (probs == 1))

    if is_onehot:
        print(f"  ⚠️  {csv_path.split('/')[-1]} 是 one-hot 格式，需要概率版本")
        return None

    return df

def main():
    print("="*70)
    print("🔍 生成超高置信度測試集偽標籤")
    print("="*70)

    # 使用可用的概率格式預測文件
    model_files = {
        'V2-L 512 Ensemble': 'data/submission_v2l_512_ensemble.csv',
        'V2-L 512 (50-50)': 'data/submission_v2l50_best50.csv',
        'V2-L 512 (60-40)': 'data/submission_v2l60_best40.csv',
    }

    # 載入預測
    predictions = {}
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    print("\n📂 載入模型預測...")
    for name, path in model_files.items():
        df = load_predictions_as_probs(path)
        if df is not None:
            predictions[name] = df
            max_conf = df[class_cols].max(axis=1).mean()
            print(f"  ✅ {name}: {len(df)} 張，平均最大置信度 {max_conf:.4f}")
        else:
            print(f"  ❌ {name}: 格式不正確，跳過")

    if len(predictions) < 2:
        print("\n❌ 至少需要 2 個模型的概率預測！")
        print("\n💡 解決方案：使用以下文件（已是概率格式）")
        print("  • data/submission_v2l50_best50.csv (V2-L 512 概率版本)")
        print("  • data/submission_v2l_512_ensemble.csv (V2-L 純集成)")
        print("\n  或重新生成 Hybrid Adaptive 的概率版本")
        return

    # 取得樣本數量
    n_samples = len(list(predictions.values())[0])
    filenames = list(predictions.values())[0]['new_filename'].values

    # 計算每個樣本的：1) 預測類別，2) 置信度
    all_pred_classes = []
    all_confidences = []

    for name, df in predictions.items():
        probs = df[class_cols].values
        pred_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        all_pred_classes.append(pred_classes)
        all_confidences.append(confidences)

    all_pred_classes = np.array(all_pred_classes)  # (n_models, n_samples)
    all_confidences = np.array(all_confidences)    # (n_models, n_samples)

    # 計算平均置信度和一致性
    avg_confidence = np.mean(all_confidences, axis=0)

    # 檢查所有模型預測是否一致
    if len(predictions) == 3:
        consistent = (all_pred_classes[0] == all_pred_classes[1]) & \
                     (all_pred_classes[1] == all_pred_classes[2])
    else:  # 2 個模型
        consistent = (all_pred_classes[0] == all_pred_classes[1])

    print(f"\n📊 統計分析")
    print(f"  總樣本數: {n_samples}")
    print(f"  模型一致的樣本: {consistent.sum()} ({consistent.sum()/n_samples*100:.1f}%)")
    print(f"  平均置信度 ≥0.99: {(avg_confidence >= 0.99).sum()} ({(avg_confidence >= 0.99).sum()/n_samples*100:.1f}%)")

    # 篩選：一致 + 高置信度
    mask_normal = consistent & (avg_confidence >= 0.99)
    mask_covid = consistent & (avg_confidence >= 0.95) & (all_pred_classes[0] == 3)  # COVID-19 放寬

    final_mask = mask_normal | mask_covid

    print(f"\n✅ 最終篩選結果:")
    print(f"  一般類別 (conf ≥0.99): {mask_normal.sum()}")
    print(f"  COVID-19 (conf ≥0.95): {mask_covid.sum()}")
    print(f"  總計高置信度樣本: {final_mask.sum()} ({final_mask.sum()/n_samples*100:.1f}%)")

    # 創建偽標籤數據集
    selected_filenames = filenames[final_mask]
    selected_classes = all_pred_classes[0][final_mask]
    selected_confidences = avg_confidence[final_mask]

    # 構建 DataFrame
    pseudo_df = pd.DataFrame({
        'new_filename': selected_filenames,
        'label': selected_classes,
        'confidence': selected_confidences
    })

    # 添加 one-hot 列
    for i, col in enumerate(class_cols):
        pseudo_df[col] = (pseudo_df['label'] == i).astype(int)

    # 按類別統計
    print(f"\n📈 偽標籤類別分布:")
    for i, col in enumerate(class_cols):
        count = (pseudo_df['label'] == i).sum()
        print(f"  {col}: {count} ({count/len(pseudo_df)*100:.1f}%)")

    # 保存
    output_path = 'data/test_pseudo_labels_ultra_high_conf.csv'
    pseudo_df.to_csv(output_path, index=False)

    print(f"\n✅ 偽標籤已保存: {output_path}")
    print(f"   樣本數: {len(pseudo_df)}")
    print(f"   平均置信度: {pseudo_df['confidence'].mean():.4f}")
    print(f"   最低置信度: {pseudo_df['confidence'].min():.4f}")

    # 保存統計報告
    with open('data/pseudo_labels_stats.txt', 'w') as f:
        f.write(f"超高置信度偽標籤統計報告\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"生成時間: 2025-11-15\n")
        f.write(f"來源模型: {', '.join(predictions.keys())}\n")
        f.write(f"測試集總數: {n_samples}\n")
        f.write(f"高置信度樣本: {len(pseudo_df)} ({len(pseudo_df)/n_samples*100:.1f}%)\n\n")
        f.write(f"類別分布:\n")
        for i, col in enumerate(class_cols):
            count = (pseudo_df['label'] == i).sum()
            f.write(f"  {col}: {count} ({count/len(pseudo_df)*100:.1f}%)\n")
        f.write(f"\n置信度統計:\n")
        f.write(f"  平均: {pseudo_df['confidence'].mean():.4f}\n")
        f.write(f"  最小: {pseudo_df['confidence'].min():.4f}\n")
        f.write(f"  最大: {pseudo_df['confidence'].max():.4f}\n")

    print(f"   統計報告: data/pseudo_labels_stats.txt")

    print(f"\n{'='*70}")
    print("🎉 Phase 1 完成！")
    print(f"{'='*70}")
    print("\n下一步: Phase 2 - 合併到訓練集")

if __name__ == "__main__":
    main()
