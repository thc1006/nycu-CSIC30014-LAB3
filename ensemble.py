"""
Ensemble 腳本 - 合併多個模型的預測
使用方式：python ensemble.py
"""
import pandas as pd
import numpy as np
import os
from glob import glob

def load_submission(path):
    """載入提交檔案"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def ensemble_voting(submissions, method='soft'):
    """
    合併多個模型的預測

    Args:
        submissions: list of DataFrames
        method: 'soft' (probability averaging) or 'hard' (majority voting)

    Returns:
        Ensemble DataFrame
    """
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 檢查所有 submission 的檔名順序是否一致
    filenames = submissions[0]['new_filename'].values
    for sub in submissions[1:]:
        assert np.all(sub['new_filename'].values == filenames), "檔名順序不一致！"

    if method == 'soft':
        # Soft voting: 平均所有模型的概率
        probs = np.stack([sub[label_cols].values for sub in submissions], axis=0)
        avg_probs = probs.mean(axis=0)

        # Convert to one-hot
        pred_classes = avg_probs.argmax(axis=1)
        one_hot = np.eye(4)[pred_classes]

        df_ensemble = pd.DataFrame(one_hot, columns=label_cols)
        df_ensemble.insert(0, 'new_filename', filenames)

    elif method == 'hard':
        # Hard voting: 多數投票
        votes = np.stack([sub[label_cols].values.argmax(axis=1) for sub in submissions], axis=0)

        # 對每個樣本統計投票
        pred_classes = []
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i], minlength=4)
            pred_classes.append(counts.argmax())

        pred_classes = np.array(pred_classes)
        one_hot = np.eye(4)[pred_classes]

        df_ensemble = pd.DataFrame(one_hot, columns=label_cols)
        df_ensemble.insert(0, 'new_filename', filenames)

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_ensemble

def main():
    print("=" * 80)
    print("ENSEMBLE PREDICTIONS".center(80))
    print("=" * 80)
    print()

    # 尋找所有提交檔案
    submission_files = sorted(glob("submission_exp*.csv"))

    if len(submission_files) == 0:
        print("❌ 找不到任何提交檔案 (submission_exp*.csv)")
        print("請先執行 run_all_experiments.py")
        return

    print(f"找到 {len(submission_files)} 個提交檔案:")
    for f in submission_files:
        print(f"  ✓ {f}")
    print()

    # 載入所有提交
    submissions = []
    for f in submission_files:
        df = load_submission(f)
        if df is not None:
            submissions.append(df)
            print(f"[OK] 載入 {f}: {len(df)} 筆預測")

    if len(submissions) == 0:
        print("\n❌ 沒有成功載入任何提交檔案")
        return

    print(f"\n成功載入 {len(submissions)} 個模型的預測")

    # 方法 1: Soft voting (推薦)
    print("\n[1/2] 生成 Soft Voting Ensemble...")
    ensemble_soft = ensemble_voting(submissions, method='soft')
    out_soft = "submission_ensemble_soft.csv"
    ensemble_soft.to_csv(out_soft, index=False)
    print(f"  ✓ 儲存至: {out_soft}")

    # 顯示類別分佈
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    print("\n  Soft Ensemble 預測分佈:")
    for col in label_cols:
        count = ensemble_soft[col].sum()
        pct = 100 * count / len(ensemble_soft)
        print(f"    {col:12s}: {int(count):4d} ({pct:5.2f}%)")

    # 方法 2: Hard voting
    print("\n[2/2] 生成 Hard Voting Ensemble...")
    ensemble_hard = ensemble_voting(submissions, method='hard')
    out_hard = "submission_ensemble_hard.csv"
    ensemble_hard.to_csv(out_hard, index=False)
    print(f"  ✓ 儲存至: {out_hard}")

    # 顯示類別分佈
    print("\n  Hard Ensemble 預測分佈:")
    for col in label_cols:
        count = ensemble_hard[col].sum()
        pct = 100 * count / len(ensemble_hard)
        print(f"    {col:12s}: {int(count):4d} ({pct:5.2f}%)")

    print("\n" + "=" * 80)
    print("ENSEMBLE COMPLETE".center(80))
    print("=" * 80)
    print("\n建議:")
    print("  1. 優先提交: submission_ensemble_soft.csv")
    print("  2. 備選提交: submission_ensemble_hard.csv")
    print("  3. 也可以提交個別模型中表現最好的")
    print("\n預期 Ensemble 提升: +2-4%")
    print("目標分數: 87-92%")

if __name__ == "__main__":
    main()
