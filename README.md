# CXR Minimal Workflow v2 (RTX 3050 → Colab A100)

**What's new in v2**
- Separate image folders per split: `train_images/`, `val_images/`, `test_images/` (Windows paths already set).
- Default **submission.csv** will be written to: `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/data`.
- `predict.py` can auto-build `test_data.csv` from `test_images/` if it doesn't exist.
- Added `src/build_test_csv.py` (one-click to create test CSV).

## Your Windows paths (preconfigured)
- train: `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/train_images`
- val:   `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/val_images`
- test:  `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/test_images`
- CSV & submission: `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/data`

## Quick Start (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (Optional) Build test CSV from the folder if you don't have one yet:
python src\build_test_csv.py --config configs\model_small.yaml

# Local sanity run (RTX 3050)
python src\train.py   --config configs\model_small.yaml
python src\eval.py    --config configs\model_small.yaml --ckpt outputs\run1\best.pt
python src\predict.py --config configs\model_small.yaml --ckpt outputs\run1\best.pt
#  ↳ writes submission to C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/data\submission.csv by default
```

## Colab (A100)
Open `notebooks/A100_Final_Train.ipynb` and **Run all**.
- It expects the config at `configs/model_big.yaml` (edit the Windows paths inside if you mirror your data to Drive).

_Last generated: 2025-10-11T18:08:44.326624_
