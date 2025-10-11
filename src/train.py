import os, math, argparse, torch, numpy as np, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from torchvision import models
from .data import make_loader
from .losses import LabelSmoothingCE, FocalLoss
from .utils import load_config, seed_everything, set_perf_flags, get_amp_dtype

def build_model(name: str, num_classes: int):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m

def cosine_lr(optimizer, base_lr, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, amp_dtype):
    model.train()
    total, correct = 0, 0
    all_preds, all_tgts = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, targets)
        else:
            logits = model(imgs); loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        preds = logits.argmax(1)
        total += targets.size(0); correct += (preds == targets).sum().item()
        all_preds.append(preds.detach().cpu().numpy())
        all_tgts.append(targets.detach().cpu().numpy())

    acc = correct / total if total else 0.0
    f1 = f1_score(np.concatenate(all_tgts), np.concatenate(all_preds), average="macro")
    return acc, f1

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_preds, all_tgts = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        total += targets.size(0); correct += (preds == targets).sum().item()
        all_preds.append(preds.detach().cpu().numpy())
        all_tgts.append(targets.detach().cpu().numpy())
    acc = correct / total if total else 0.0
    f1 = f1_score(np.concatenate(all_tgts), np.concatenate(all_preds), average="macro")
    return acc, f1

def main(args):
    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])
    set_perf_flags(cfg.get("perf", {}))
    amp_dtype = get_amp_dtype(cfg.get("perf", {}))
    use_channels_last = bool(cfg.get("perf", {}).get("channels_last", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | CUDA name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    data_cfg, train_cfg, mdl_cfg, out_cfg = cfg["data"], cfg["train"], cfg["model"], cfg["out"]
    # data loaders with split-specific image directories
    train_ds, train_loader = make_loader(
        data_cfg["train_csv"], data_cfg["images_dir_train"], data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], train_cfg["batch_size"], train_cfg["num_workers"], augment=True,
        shuffle=True, weighted=bool(train_cfg.get("use_weighted_sampler", False))
    )
    val_ds, val_loader = make_loader(
        data_cfg["val_csv"], data_cfg["images_dir_val"], data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], train_cfg["batch_size"], train_cfg["num_workers"], augment=False,
        shuffle=False, weighted=False
    )

    model = build_model(mdl_cfg["name"], data_cfg["num_classes"]).to(device)
    if use_channels_last: model = model.to(memory_format=torch.channels_last)

    # opt + loss + sched
    if train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=0.9, weight_decay=train_cfg["weight_decay"])

    loss_fn = FocalLoss(gamma=2.0) if train_cfg["loss"] == "focal" else LabelSmoothingCE(eps=float(train_cfg.get("label_smoothing", 0.0)))

    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_lr(optimizer, train_cfg["lr"], warmup_steps=int(train_cfg.get("warmup_epochs",1))*steps_per_epoch,
                          total_steps=train_cfg["epochs"]*steps_per_epoch)

    scaler = torch.cuda.amp.GradScaler() if (amp_dtype == torch.float16 and device.type == "cuda") else None

    best_f1 = -1.0
    os.makedirs(out_cfg["dir"], exist_ok=True)
    for epoch in range(train_cfg["epochs"]):
        acc_tr, f1_tr = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn, amp_dtype)
        acc_val, f1_val = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"[epoch {epoch+1:02d}] train acc={acc_tr:.4f} f1={f1_tr:.4f} | val acc={acc_val:.4f} f1={f1_val:.4f}")
        if f1_val > best_f1:
            best_f1 = f1_val
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(out_cfg["dir"], "best.pt"))
            print(f"  -> saved new best to {os.path.join(out_cfg['dir'], 'best.pt')} (val macro-F1={best_f1:.4f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args)
