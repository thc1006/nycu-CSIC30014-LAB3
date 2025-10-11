import os, argparse, numpy as np, torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import models
from .data import make_loader
from .utils import load_config

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
def eval_and_report(model, loader, device, class_names):
    model.eval()
    all_preds, all_tgts = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds); all_tgts.append(targets.numpy())
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_tgts)
    f1 = f1_score(y_true, y_pred, average="macro")
    print("Macro-F1:", f1)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    try:
        import seaborn as sns
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Pred")
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/confusion_matrix.png", dpi=160, bbox_inches="tight")
        print("Saved: outputs/confusion_matrix.png")
    except Exception as e:
        print(f"[warn] could not draw CM: {e}")
    return f1

def main(args):
    cfg = load_config(args.config)
    data_cfg, mdl_cfg = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = make_loader(
        data_cfg["val_csv"], data_cfg["images_dir_val"], data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], cfg["train"]["batch_size"], cfg["train"]["num_workers"], augment=False,
        shuffle=False, weighted=False
    )

    model = build_model(mdl_cfg["name"], data_cfg["num_classes"]).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Loaded checkpoint: {args.ckpt}")

    if args.sanity_only:
        print("[sanity] batch shapes and sample preview:")
        for i, (x, y, f) in enumerate(val_loader):
            print(" batch", i, x.shape, y[:8].numpy(), f[:3]); break
        return

    f1 = eval_and_report(model, val_loader, device, class_names=data_cfg["label_cols"])
    print("Validation Macro-F1:", f1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--sanity_only", type=str, default="false")
    args = ap.parse_args()
    args.sanity_only = str(args.sanity_only).lower() in ("1","true","yes","y")
    main(args)
