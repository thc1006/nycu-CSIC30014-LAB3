#!/usr/bin/env python3
"""
将概率预测转换为 one-hot 编码
"""

import pandas as pd
import numpy as np
import sys

def convert_to_onehot(input_csv, output_csv):
    """将概率预测转换为 one-hot"""
    df = pd.read_csv(input_csv)
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Get probabilities
    probs = df[class_cols].values

    # Convert to one-hot
    pred_classes = np.argmax(probs, axis=1)
    onehot = np.zeros_like(probs, dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1

    # Update dataframe
    df[class_cols] = onehot

    # Save
    df.to_csv(output_csv, index=False)

    # Print stats
    class_counts = np.bincount(pred_classes, minlength=len(class_cols))
    print(f"✅ 转换完成: {input_csv} → {output_csv}")
    print(f"预测分布:")
    for i, (name, count) in enumerate(zip(class_cols, class_counts)):
        print(f"  {name}: {count} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    files_to_convert = [
        ("data/submission_v2l50_best50.csv", "data/submission_v2l50_best50_onehot.csv"),
        ("data/submission_v2l60_best40.csv", "data/submission_v2l60_best40_onehot.csv"),
        ("data/submission_v2l40_best60.csv", "data/submission_v2l40_best60_onehot.csv"),
    ]

    for input_file, output_file in files_to_convert:
        convert_to_onehot(input_file, output_file)
        print()
