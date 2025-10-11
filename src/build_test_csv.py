import os, argparse, pandas as pd
from .utils import load_config

def main(args):
    cfg = load_config(args.config)
    data = cfg["data"]
    test_dir = data["images_dir_test"]
    out_csv = data["test_csv"]
    files = [f for f in os.listdir(test_dir) if not f.startswith(".")]
    df = pd.DataFrame({"new_filename": sorted(files)})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} entries to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args)
