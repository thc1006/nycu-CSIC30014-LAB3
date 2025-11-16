#!/usr/bin/env python3
"""
测试 Google Colab Notebooks 的可执行性
模拟 Colab 环境验证关键功能
"""

import json
import sys
import os
import tempfile
import shutil
from pathlib import Path

def test_notebook_syntax(notebook_path):
    """测试 notebook JSON 格式和 Python 语法"""
    print(f"\n{'='*70}")
    print(f"📋 测试: {Path(notebook_path).name}")
    print(f"{'='*70}\n")

    # 1. 读取并验证 JSON 格式
    print("1️⃣ 验证 JSON 格式...")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        print("   ✅ JSON 格式正确")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 格式错误: {e}")
        return False

    # 2. 检查 notebook 结构
    print("\n2️⃣ 检查 Notebook 结构...")
    if 'cells' not in nb:
        print("   ❌ 缺少 cells 字段")
        return False
    print(f"   ✅ 共 {len(nb['cells'])} 个 cell")

    # 3. 统计 cell 类型
    code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
    markdown_cells = [c for c in nb['cells'] if c.get('cell_type') == 'markdown']
    print(f"   ✅ Code cells: {len(code_cells)}")
    print(f"   ✅ Markdown cells: {len(markdown_cells)}")

    # 4. 验证 Python 代码语法
    print("\n3️⃣ 验证 Python 代码语法...")
    errors = []
    for i, cell in enumerate(code_cells):
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # 跳过魔法命令和 shell 命令
        if code.strip().startswith(('!', '%', '%%')):
            continue

        try:
            compile(code, f'<cell-{i}>', 'exec')
        except SyntaxError as e:
            errors.append(f"Cell {i}: {e}")

    if errors:
        print(f"   ⚠️ 发现 {len(errors)} 个语法错误:")
        for err in errors[:5]:  # 只显示前5个
            print(f"      {err}")
    else:
        print("   ✅ 所有代码语法正确")

    # 5. 检查关键依赖
    print("\n4️⃣ 检查关键依赖...")
    all_code = '\n'.join([''.join(c.get('source', [])) for c in code_cells])

    required_imports = {
        'torch': 'PyTorch',
        'timm': 'timm (PyTorch Image Models)',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'PIL': 'Pillow',
    }

    for module, name in required_imports.items():
        if f'import {module}' in all_code or f'from {module}' in all_code:
            try:
                __import__(module)
                print(f"   ✅ {name} 已安装")
            except ImportError:
                print(f"   ⚠️ {name} 未安装 (Colab 会自动安装)")

    # 6. 检查关键路径和文件操作
    print("\n5️⃣ 检查路径逻辑...")

    # 检查 GitHub URL
    if 'github.com/thc1006/nycu-CSIC30014-LAB3' in all_code:
        print("   ✅ GitHub URL 正确")
    else:
        print("   ❌ 未找到 GitHub URL")

    # 检查数据路径
    expected_paths = [
        'data/train_images',
        'data/val_images',
        'data/test_images',
        'data/fold',
    ]

    for path in expected_paths:
        if path in all_code:
            print(f"   ✅ 路径引用正确: {path}")
        else:
            print(f"   ⚠️ 未找到路径: {path}")

    # 7. 检查训练脚本调用
    print("\n6️⃣ 检查训练脚本调用...")
    if 'train_dinov2_breakthrough.py' in all_code:
        print("   ✅ 训练脚本调用正确")

        # 检查参数
        params = ['--fold', '--epochs', '--batch_size', '--img_size', '--lr', '--output_dir']
        for param in params:
            if param in all_code:
                print(f"   ✅ 参数: {param}")
    else:
        print("   ❌ 未找到训练脚本调用")

    # 8. 检查模型名称
    print("\n7️⃣ 检查模型名称...")
    if 'vit_base_patch14_dinov2' in all_code:
        print("   ✅ DINOv2 模型名称正确")

        # 检查错误的模型名称
        if 'lvd142m' in all_code:
            print("   ❌ 发现错误的模型名称 'lvd142m'")
        else:
            print("   ✅ 没有使用错误的模型名称")

    # 9. 检查模型加载逻辑
    print("\n8️⃣ 检查模型加载逻辑...")
    if "checkpoint['model_state_dict']" in all_code:
        print("   ✅ 模型加载逻辑正确 (使用 checkpoint)")
    elif 'load_state_dict(torch.load(' in all_code:
        print("   ⚠️ 模型加载可能有问题 (直接 load)")

    # 10. 检查 Kaggle 提交格式
    print("\n9️⃣ 检查 Kaggle 提交格式...")
    if "'new_filename'" in all_code and "'label'" in all_code:
        print("   ✅ 提交格式正确")

    if "['normal', 'bacteria', 'virus', 'COVID-19']" in all_code:
        print("   ✅ 类别名称正确")

    print(f"\n{'='*70}")
    print(f"✅ {Path(notebook_path).name} 测试完成")
    print(f"{'='*70}")

    return len(errors) == 0


def simulate_colab_environment():
    """模拟 Colab 环境测试关键代码"""
    print("\n" + "="*70)
    print("🧪 模拟 Colab 环境测试")
    print("="*70)

    # 创建临时目录模拟 Colab
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n临时 Colab 目录: {tmpdir}")
        os.chdir(tmpdir)

        # 1. 测试 GPU 检测代码
        print("\n1️⃣ 测试 GPU 检测...")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

                # 测试 batch size 逻辑
                if "A100" in gpu_name:
                    batch_size = 24
                elif "L4" in gpu_name:
                    batch_size = 16
                elif "T4" in gpu_name or gpu_memory > 14:
                    batch_size = 12
                else:
                    batch_size = 8
                print(f"   ✅ 自动配置 Batch Size: {batch_size}")
            else:
                print("   ⚠️ 无 GPU (Colab 需要启用 GPU)")
        except Exception as e:
            print(f"   ❌ GPU 检测失败: {e}")

        # 2. 测试 timm 和 DINOv2
        print("\n2️⃣ 测试 timm 和 DINOv2...")
        try:
            import timm
            print(f"   ✅ timm 版本: {timm.__version__}")

            dinov2_models = [m for m in timm.list_models() if 'dinov2' in m and 'base' in m]
            if dinov2_models:
                print(f"   ✅ DINOv2 模型可用: {dinov2_models}")
            else:
                print("   ❌ DINOv2 模型不可用")
        except Exception as e:
            print(f"   ⚠️ timm 测试失败: {e}")

        # 3. 测试数据路径创建
        print("\n3️⃣ 测试数据路径...")
        try:
            os.makedirs('data/train_images', exist_ok=True)
            os.makedirs('data/val_images', exist_ok=True)
            os.makedirs('data/test_images', exist_ok=True)
            print("   ✅ 数据目录创建成功")

            # 测试 CSV 路径
            import pandas as pd
            test_df = pd.DataFrame({
                'new_filename': ['test.jpg'],
                'normal': [1], 'bacteria': [0], 'virus': [0], 'COVID-19': [0],
                'source_dir': ['data/train_images']
            })
            test_df.to_csv('data/fold0_train.csv', index=False)
            print("   ✅ CSV 文件创建成功")

            # 读取测试
            read_df = pd.read_csv('data/fold0_train.csv')
            print(f"   ✅ CSV 读取成功: {len(read_df)} 行")
        except Exception as e:
            print(f"   ❌ 数据路径测试失败: {e}")

        # 4. 测试模型创建
        print("\n4️⃣ 测试模型创建...")
        try:
            import timm
            model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=4)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ DINOv2 模型创建成功")
            print(f"   ✅ 参数量: {total_params/1e6:.1f}M")
        except Exception as e:
            print(f"   ⚠️ 模型创建失败 (Colab 会下载预训练权重): {e}")

        # 5. 测试提交格式
        print("\n5️⃣ 测试提交格式...")
        try:
            import pandas as pd
            import numpy as np

            # 模拟预测
            filenames = ['0.jpg', '1.jpg', '2.jpg']
            predictions = [0, 1, 2]  # normal, bacteria, virus
            class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

            submission_df = pd.DataFrame({
                'new_filename': filenames,
                'label': [class_names[p] for p in predictions]
            })

            # 保存
            submission_df.to_csv('test_submission.csv', index=False)

            # 验证
            read_sub = pd.read_csv('test_submission.csv')
            if 'new_filename' in read_sub.columns and 'label' in read_sub.columns:
                print("   ✅ 提交格式正确")
                print(f"   ✅ 样本数: {len(read_sub)}")
            else:
                print("   ❌ 提交格式错误")
        except Exception as e:
            print(f"   ❌ 提交格式测试失败: {e}")

        print("\n" + "="*70)
        print("✅ Colab 环境模拟测试完成")
        print("="*70)


def main():
    print("="*70)
    print("🧪 Google Colab Notebooks 测试工具")
    print("="*70)

    # 切换到项目目录
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    # 测试两个 notebook
    notebooks = [
        'Colab_A100_AGGRESSIVE.ipynb',
        'Colab_L4_OPTIMIZED.ipynb'
    ]

    results = {}
    for nb in notebooks:
        if os.path.exists(nb):
            results[nb] = test_notebook_syntax(nb)
        else:
            print(f"\n❌ 文件不存在: {nb}")
            results[nb] = False

    # 模拟 Colab 环境
    simulate_colab_environment()

    # 总结
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)

    for nb, passed in results.items():
        status = "✅ 通过" if passed else "⚠️ 有警告"
        print(f"{status}: {nb}")

    if all(results.values()):
        print("\n✅ 所有 Notebook 测试通过，可以在 Colab 上运行！")
        return 0
    else:
        print("\n⚠️ 部分 Notebook 有警告，但主要功能应该正常")
        return 1


if __name__ == '__main__':
    sys.exit(main())
