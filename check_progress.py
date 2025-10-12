"""
簡易進度監控腳本
執行方式：python check_progress.py
"""
import os
from datetime import datetime

def check_experiment_status():
    print("=" * 80)
    print(f"訓練進度檢查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    experiments = [
        ("實驗 1: ConvNeXt-Tiny", "exp1_convnext_tiny", "submission_exp1.csv"),
        ("實驗 2: EfficientNetV2-S", "exp2_efficientnetv2", "submission_exp2.csv"),
        ("實驗 3: ResNet34", "exp3_resnet34_long", "submission_exp3.csv"),
        ("實驗 4: EfficientNet-B0", "exp4_efficientnet_b0", "submission_exp4.csv"),
        ("實驗 5: ResNet18-Ultra", "exp5_resnet18_ultra", "submission_exp5.csv"),
    ]

    for i, (name, output_dir, submission) in enumerate(experiments, 1):
        print(f"\n[{i}] {name}")

        # Check if training is complete (best.pt exists)
        ckpt_path = f"outputs/{output_dir}/best.pt"
        train_log = f"outputs/{output_dir}/train.log"

        if os.path.exists(ckpt_path):
            ckpt_size = os.path.getsize(ckpt_path) / (1024*1024)
            print(f"    訓練: [OK] 已完成 (checkpoint: {ckpt_size:.1f} MB)")

            # Check last line of log for best F1
            if os.path.exists(train_log):
                try:
                    with open(train_log, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-50:]):
                            if 'best' in line.lower() or 'val' in line.lower():
                                print(f"    {line.strip()}")
                                break
                except:
                    pass
        elif os.path.exists(train_log):
            # Training in progress
            try:
                with open(train_log, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else "No data"
                    print(f"    訓練: [進行中] {last_line}")
            except:
                print(f"    訓練: [進行中] (無法讀取 log)")
        else:
            print(f"    訓練: [未開始]")

        # Check submission file
        if os.path.exists(submission):
            sub_size = os.path.getsize(submission) / 1024
            print(f"    提交: [OK] {submission} ({sub_size:.1f} KB)")
        else:
            print(f"    提交: [未生成]")

    # Summary
    print("\n" + "=" * 80)
    completed_ckpts = sum(1 for _, d, _ in experiments if os.path.exists(f"outputs/{d}/best.pt"))
    completed_subs = sum(1 for _, _, s in experiments if os.path.exists(s))
    print(f"總進度: 訓練 {completed_ckpts}/5 完成 | 提交檔案 {completed_subs}/5 已生成")

    if completed_subs >= 5:
        print("\n[!] 所有實驗已完成！下一步執行: python ensemble.py")
    print("=" * 80)

if __name__ == "__main__":
    check_experiment_status()
