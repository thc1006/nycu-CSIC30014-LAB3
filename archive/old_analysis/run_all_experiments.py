"""
自動化訓練腳本 - 依序執行所有實驗
執行方式：python run_all_experiments.py
"""
import os
import subprocess
import time
from datetime import datetime

# 實驗配置列表 (按預期訓練時間排序)
EXPERIMENTS = [
    {
        "name": "實驗 1: ConvNeXt-Tiny + 288px",
        "config": "configs/exp1_convnext_tiny.yaml",
        "est_time": "2.5 hours",
        "expected_score": "83-85%"
    },
    {
        "name": "實驗 2: EfficientNetV2-S + 320px + SWA",
        "config": "configs/exp2_efficientnetv2.yaml",
        "est_time": "3 hours",
        "expected_score": "84-86%"
    },
    {
        "name": "實驗 3: ResNet34 + 384px + Long",
        "config": "configs/exp3_resnet34_long.yaml",
        "est_time": "2 hours",
        "expected_score": "85-87%"
    },
    {
        "name": "實驗 4: EfficientNet-B0 + 256px + Ultra Long",
        "config": "configs/exp4_efficientnet_b0.yaml",
        "est_time": "2.5 hours",
        "expected_score": "84-86%"
    },
    {
        "name": "實驗 5: ResNet18 + 384px + Ultra Aug",
        "config": "configs/exp5_resnet18_ultra.yaml",
        "est_time": "1.5 hours",
        "expected_score": "83-85%"
    }
]

def print_banner(text):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def run_experiment(exp_num, exp):
    """執行單個實驗"""
    print_banner(f"開始執行 {exp['name']}")
    print(f"配置檔案: {exp['config']}")
    print(f"預計時間: {exp['est_time']}")
    print(f"預期分數: {exp['expected_score']}")
    print()

    start_time = time.time()

    # 訓練
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 開始訓練...")
    train_cmd = f"python -m src.train_v2 --config {exp['config']}"
    result = subprocess.run(train_cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[X] 訓練失敗！跳過此實驗。")
        return False

    # 提取checkpoint路徑
    config_name = os.path.basename(exp['config']).replace('.yaml', '')
    ckpt_dir = f"outputs/{config_name}"
    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    # TTA 預測
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 開始 TTA 預測...")
    tta_cmd = f"python -m src.tta_predict --config {exp['config']} --ckpt {ckpt_path}"
    result = subprocess.run(tta_cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[X] TTA 預測失敗！")
        return False

    elapsed = time.time() - start_time
    print(f"\n[OK] {exp['name']} 完成！")
    print(f"實際耗時: {elapsed/3600:.2f} hours")

    return True

def main():
    print_banner("開始執行所有實驗")
    print("總共 5 個實驗")
    print("預計總時間: 約 11-12 小時")
    print()

    completed = []
    failed = []

    overall_start = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        success = run_experiment(i, exp)
        if success:
            completed.append(exp['name'])
        else:
            failed.append(exp['name'])

        print(f"\n進度: {i}/{len(EXPERIMENTS)} 完成")
        print()

    total_time = time.time() - overall_start

    # 最終報告
    print_banner("所有實驗執行完畢")
    print(f"總耗時: {total_time/3600:.2f} hours")
    print(f"\n[成功]: {len(completed)}/{len(EXPERIMENTS)}")
    for name in completed:
        print(f"  - {name}")

    if failed:
        print(f"\n[失敗]: {len(failed)}/{len(EXPERIMENTS)}")
        for name in failed:
            print(f"  - {name}")

    print("\n生成的提交檔案:")
    for i in range(1, 6):
        sub_file = f"submission_exp{i}.csv"
        if os.path.exists(sub_file):
            print(f"  [v] {sub_file}")

    print("\n下一步: 執行 ensemble.py 來合併所有模型的預測！")

if __name__ == "__main__":
    main()
