import yaml, random, numpy as np, torch

def load_config(cfg_path: str):
    # support a minimal "inherits" field to merge two files (parent <- child).
    def _read(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    def _deep_merge(base, override):
        """Deep merge override dict into base dict"""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
    cfg = _read(cfg_path)
    if isinstance(cfg, dict) and "inherits" in cfg:
        parent = _read(cfg["inherits"])
        child = {k:v for k,v in cfg.items() if k != "inherits"}
        _deep_merge(parent, child)
        cfg = parent
    return cfg

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def set_perf_flags(perf: dict):
    # TF32
    prec = str(perf.get("tf32","float32")).lower()
    try:
        if prec in ("high","medium"):
            torch.set_float32_matmul_precision(prec)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.set_float32_matmul_precision("high")
    except Exception as e:
        print(f"[warn] TF32 not applied: {e}")

    torch.backends.cudnn.benchmark = bool(perf.get("cudnn_benchmark", False))

def get_amp_dtype(perf: dict):
    dtype = str(perf.get("amp_dtype","none")).lower()
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp16":
        return torch.float16
    return None
