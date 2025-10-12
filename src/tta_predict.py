"""
Test-Time Augmentation (TTA) for improved predictions.
Applies multiple augmentations at inference time and averages predictions.
"""
import os, argparse, torch, pandas as pd, numpy as np
from torch import nn
from .utils import load_config, set_perf_flags
from .data import make_loader
from .train_v2 import build_model

class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation wrapper that applies multiple transformations
    and averages the predictions.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Apply 6 different augmentations:
        1. Original
        2. Horizontal flip
        3. Vertical flip
        4. Rotate 90
        5. Rotate 180
        6. Rotate 270
        """
        batch_size = x.size(0)
        device = x.device

        # Original
        pred_orig = self.model(x)

        # Horizontal flip
        pred_hflip = self.model(torch.flip(x, dims=[3]))

        # Vertical flip
        pred_vflip = self.model(torch.flip(x, dims=[2]))

        # Rotate 90 degrees
        pred_rot90 = self.model(torch.rot90(x, k=1, dims=[2, 3]))

        # Rotate 180 degrees
        pred_rot180 = self.model(torch.rot90(x, k=2, dims=[2, 3]))

        # Rotate 270 degrees
        pred_rot270 = self.model(torch.rot90(x, k=3, dims=[2, 3]))

        # Stack and average all predictions
        all_preds = torch.stack([
            pred_orig,
            pred_hflip,
            pred_vflip,
            pred_rot90,
            pred_rot180,
            pred_rot270
        ], dim=0)

        return all_preds.mean(dim=0)

def predict_with_tta(model, test_loader, device, num_classes, submission_cols, submission_file_col):
    """
    Generate predictions using Test-Time Augmentation.

    Returns:
        DataFrame with filenames and one-hot predictions
    """
    tta_model = TTAWrapper(model).to(device)
    tta_model.eval()

    all_probs = []
    all_fnames = []

    with torch.no_grad():
        for imgs, _, fnames in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = tta_model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_fnames.extend(fnames)

    # Concatenate all predictions
    all_probs = np.concatenate(all_probs, axis=0)

    # Convert to one-hot (argmax)
    pred_classes = all_probs.argmax(axis=1)
    one_hot = np.eye(num_classes)[pred_classes]

    # Build submission DataFrame
    df = pd.DataFrame(one_hot, columns=submission_cols)
    df.insert(0, submission_file_col, all_fnames)

    return df

def main(args):
    cfg = load_config(args.config)
    set_perf_flags(cfg.get("perf", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data_cfg, mdl_cfg, out_cfg, train_cfg = cfg["data"], cfg["model"], cfg["out"], cfg["train"]

    # Load test data
    _, test_loader = make_loader(
        data_cfg["test_csv"], data_cfg["images_dir_test"],
        data_cfg["file_col"], data_cfg["label_cols"],
        mdl_cfg["img_size"], train_cfg["batch_size"], train_cfg["num_workers"],
        augment=False, shuffle=False, weighted=False
    )

    # Build model and load checkpoint
    model = build_model(mdl_cfg["name"], data_cfg["num_classes"]).to(device)

    ckpt_path = args.ckpt if args.ckpt else os.path.join(out_cfg["dir"], "best.pt")
    print(f"[loading] {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    print("[TTA] Generating predictions with Test-Time Augmentation...")
    print("[TTA] Applying 6 augmentations: original, hflip, vflip, rot90, rot180, rot270")

    df_submission = predict_with_tta(
        model, test_loader, device,
        data_cfg["num_classes"],
        data_cfg["submission_cols"],
        data_cfg["submission_file_col"]
    )

    # Save submission - Always use submission_tta.csv for TTA
    out_path = "submission_tta.csv"
    df_submission.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(df_submission)} rows)")
    print(f"[preview]\n{df_submission.head()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (default: outputs/<run>/best.pt)")
    args = ap.parse_args()
    main(args)
