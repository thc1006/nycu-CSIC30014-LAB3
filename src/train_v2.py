"""
Enhanced training script with Stage 1 optimizations:
- ConvNeXt-Base support
- Improved Focal Loss with class weights
- Mixup/CutMix augmentation
- Stochastic Weight Averaging (SWA)
- Advanced data augmentation
"""
import os, math, argparse, torch, numpy as np, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from torchvision import models
from .data import make_loader
from .losses import LabelSmoothingCE, FocalLoss, ImprovedFocalLoss
from .aug import mixup_data, cutmix_data
from .utils import load_config, seed_everything, set_perf_flags, get_amp_dtype

def build_model(name: str, num_classes: int):
    """Build model with support for ConvNeXt variants"""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
    elif name == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext_base":
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name.startswith('vit_') or name.startswith('swin_') or name.startswith('deit_'):
        # Vision Transformer and Swin Transformer support via timm
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViT/Swin models. Install with: pip install timm")
        m = timm.create_model(name, pretrained=True, num_classes=num_classes)
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

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, amp_dtype,
                    use_mixup=False, mixup_alpha=1.0, mixup_prob=0.5):
    """Training loop with optional Mixup/CutMix support"""
    model.train()
    total, correct = 0, 0
    all_preds, all_tgts = [], []

    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup/CutMix randomly
        if use_mixup and np.random.rand() < mixup_prob:
            if np.random.rand() < 0.5:
                # Mixup
                imgs, targets_a, targets_b, lam = mixup_data(imgs, targets, mixup_alpha, device)
            else:
                # CutMix
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, targets, mixup_alpha, device)

            # Forward pass with AMP
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(imgs)
                    loss = lam * loss_fn(logits, targets_a) + (1 - lam) * loss_fn(logits, targets_b)
            else:
                logits = model(imgs)
                loss = lam * loss_fn(logits, targets_a) + (1 - lam) * loss_fn(logits, targets_b)
        else:
            # Standard training
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(imgs)
                    loss = loss_fn(logits, targets)
            else:
                logits = model(imgs)
                loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
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
        total += targets.size(0)
        correct += (preds == targets).sum().item()
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

    print("[init] Extracting augmentation config...")
    # Extract augmentation config
    aug_config = {
        'aug_rotation': train_cfg.get('aug_rotation', 15),
        'aug_translate': train_cfg.get('aug_translate', 0.1),
        'aug_scale_min': train_cfg.get('aug_scale_min', 0.9),
        'aug_scale_max': train_cfg.get('aug_scale_max', 1.1),
        'aug_shear': train_cfg.get('aug_shear', 10),
        'random_erasing_prob': train_cfg.get('random_erasing_prob', 0.3),
    }
    advanced_aug = train_cfg.get('advanced_aug', False)

    print(f"[init] Creating train data loader...")
    # Data loaders
    train_ds, train_loader = make_loader(
        data_cfg["train_csv"], data_cfg["images_dir_train"], data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], train_cfg["batch_size"], train_cfg["num_workers"], augment=True,
        shuffle=True, weighted=bool(train_cfg.get("use_weighted_sampler", False)),
        advanced_aug=advanced_aug, aug_config=aug_config
    )
    print(f"[init] Train loader created. Creating val data loader...")
    val_ds, val_loader = make_loader(
        data_cfg["val_csv"], data_cfg["images_dir_val"], data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], train_cfg["batch_size"], train_cfg["num_workers"], augment=False,
        shuffle=False, weighted=False
    )
    print(f"[init] Val loader created. Building model...")

    model = build_model(mdl_cfg["name"], data_cfg["num_classes"]).to(device)
    print(f"[init] Model built and moved to {device}")
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Optimizer
    if train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=0.9, weight_decay=train_cfg["weight_decay"])

    # Loss function
    loss_type = train_cfg.get("loss", "ce")
    if loss_type == "focal_improved":
        focal_alpha = train_cfg.get("focal_alpha", None)
        focal_gamma = train_cfg.get("focal_gamma", 2.0)
        label_smoothing = train_cfg.get("label_smoothing", 0.1)
        loss_fn = ImprovedFocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"[loss] ImprovedFocalLoss (gamma={focal_gamma}, alpha={focal_alpha}, smoothing={label_smoothing})")
    elif loss_type == "focal":
        loss_fn = FocalLoss(gamma=2.0)
        print("[loss] FocalLoss (gamma=2.0)")
    else:
        loss_fn = LabelSmoothingCE(eps=float(train_cfg.get("label_smoothing", 0.0)))
        print(f"[loss] LabelSmoothingCE (eps={train_cfg.get('label_smoothing', 0.0)})")

    # Learning rate scheduler
    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_lr(optimizer, train_cfg["lr"],
                         warmup_steps=int(train_cfg.get("warmup_epochs", 1)) * steps_per_epoch,
                         total_steps=train_cfg["epochs"] * steps_per_epoch)

    scaler = torch.cuda.amp.GradScaler() if (amp_dtype == torch.float16 and device.type == "cuda") else None

    # Mixup/CutMix settings
    use_mixup = train_cfg.get("use_mixup", False)
    mixup_alpha = train_cfg.get("mixup_alpha", 1.0)
    mixup_prob = train_cfg.get("mixup_prob", 0.5)

    if use_mixup:
        print(f"[augment] Mixup/CutMix enabled (alpha={mixup_alpha}, prob={mixup_prob})")

    # SWA settings
    use_swa = train_cfg.get("use_swa", False)
    swa_start = train_cfg.get("swa_start", train_cfg["epochs"] - 5)
    swa_model = None

    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=train_cfg.get("swa_lr", 0.00005))
        print(f"[SWA] enabled (start epoch={swa_start}, lr={train_cfg.get('swa_lr', 0.00005)})")

    best_f1 = -1.0
    os.makedirs(out_cfg["dir"], exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        acc_tr, f1_tr = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn, amp_dtype,
                                        use_mixup, mixup_alpha, mixup_prob)
        acc_val, f1_val = evaluate(model, val_loader, device)

        # Update scheduler
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(f"[epoch {epoch+1:02d}] train acc={acc_tr:.4f} f1={f1_tr:.4f} | val acc={acc_val:.4f} f1={f1_val:.4f}")

        if f1_val > best_f1:
            best_f1 = f1_val
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(out_cfg["dir"], "best.pt"))
            print(f"  -> saved new best to {os.path.join(out_cfg['dir'], 'best.pt')} (val macro-F1={best_f1:.4f})")

    # Finalize SWA
    if use_swa:
        print("[SWA] Updating BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Evaluate SWA model
        acc_swa, f1_swa = evaluate(swa_model, val_loader, device)
        print(f"[SWA final] val acc={acc_swa:.4f} f1={f1_swa:.4f}")

        # Save SWA model if better
        if f1_swa > best_f1:
            torch.save({"model": swa_model.module.state_dict(), "cfg": cfg},
                      os.path.join(out_cfg["dir"], "best_swa.pt"))
            print(f"  -> saved SWA model to {os.path.join(out_cfg['dir'], 'best_swa.pt')} (val macro-F1={f1_swa:.4f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args)
