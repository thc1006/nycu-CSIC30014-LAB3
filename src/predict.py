import os, argparse, pandas as pd, numpy as np, torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Support both relative and absolute imports
try:
    from .utils import load_config
except ImportError:
    from utils import load_config

def ensure_test_csv(cfg):
    data = cfg["data"]
    test_csv = data["test_csv"]
    test_dir = data.get("images_dir_test", None)
    if os.path.exists(test_csv):
        return test_csv
    if test_dir and os.path.isdir(test_dir):
        # build a CSV with 'new_filename' from the folder listing
        files = [f for f in os.listdir(test_dir) if not f.startswith(".")]
        df = pd.DataFrame({"new_filename": sorted(files)})
        os.makedirs(os.path.dirname(test_csv), exist_ok=True)
        df.to_csv(test_csv, index=False)
        print(f"[auto] wrote test CSV with {len(df)} files -> {test_csv}")
        return test_csv
    raise FileNotFoundError(f"test_csv not found and images_dir_test missing: {test_csv}")

class TestSet(Dataset):
    def __init__(self, csv_path, images_dir, file_col, img_size):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.file_col = file_col
        import torchvision.transforms as T
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row[self.file_col])
        path = os.path.join(self.images_dir, fname)
        x = Image.open(path).convert("RGB")
        x = self.tf(x)
        return x, fname

def build_model(name: str, num_classes: int):
    from torch import nn
    if name == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=None); m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m

@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits, names = [], []
    for x, f in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        names.extend(f)
    return np.concatenate(all_logits), names

def main(args):
    cfg = load_config(args.config)
    data_cfg, mdl_cfg, out_cfg = cfg["data"], cfg["model"], cfg["out"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = ensure_test_csv(cfg)
    ts = TestSet(test_csv, data_cfg["images_dir_test"], data_cfg["file_col"], mdl_cfg["img_size"])
    tl = DataLoader(ts, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = build_model(mdl_cfg["name"], data_cfg["num_classes"]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"Loaded checkpoint: {args.ckpt}")

    logits, names = predict_logits(model, tl, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(one_hot.shape[0]), pred_idx] = 1

    df = pd.DataFrame(one_hot, columns=data_cfg["submission_cols"])
    df.insert(0, data_cfg["submission_file_col"], names)

    # default to config's submission_path when --out is not provided
    out_path = args.out or out_cfg.get("submission_path", "outputs/submission.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved submission to", out_path)

def predict_and_save(checkpoint, test_csv, test_dir, output_csv, img_size=384, batch_size=32, num_classes=4):
    """
    直接预测函数，供 Pipeline 调用
    Args:
        checkpoint: 模型检查点路径
        test_csv: 测试集 CSV 文件
        test_dir: 测试图像目录
        output_csv: 输出 CSV 路径
        img_size: 图像尺寸
        batch_size: 批次大小
        num_classes: 类别数量
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据集
    ts = TestSet(test_csv, test_dir, 'new_filename', img_size)
    tl = DataLoader(ts, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 加载模型
    state = torch.load(checkpoint, map_location=device)

    # 从 config 或 checkpoint 文件名推断模型类型
    if 'config' in state and isinstance(state['config'], dict):
        if 'model' in state['config']:
            # 扁平的 config 结构
            if isinstance(state['config']['model'], str):
                model_name = state['config']['model']
            else:
                model_name = state['config']['model'].get('name', 'efficientnet_v2_s')
        else:
            model_name = 'efficientnet_v2_s'
    else:
        model_name = state.get('model_name', 'efficientnet_v2_s')

    # 支持更多模型
    from torch import nn
    if 'efficientnet_v2_l' in model_name or 'efficientnet_v2_l' in str(checkpoint):
        model = models.efficientnet_v2_l(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'efficientnet_v2_m' in model_name or 'efficientnet_v2_m' in str(checkpoint):
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = build_model(model_name, num_classes)

    # 支持不同的 checkpoint 格式
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        # 直接加载 state dict
        model.load_state_dict(state)

    model = model.to(device)
    print(f"✅ Loaded checkpoint: {checkpoint}")

    # 预测
    logits, names = predict_logits(model, tl, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(one_hot.shape[0]), pred_idx] = 1

    # 保存结果
    df = pd.DataFrame(one_hot, columns=['Normal', 'Bacteria', 'Virus', 'COVID-19'])
    df.insert(0, 'new_filename', names)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved submission to {output_csv}")
    return output_csv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args)
