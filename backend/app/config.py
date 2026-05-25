from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "good_models" / "cinn_model_ep100.pt"
H5_PATH = ROOT / "stellar_training_data.h5"
NORM_STATS_PATH = ROOT / "backend" / "norm_stats.npz"

INPUT_DIM = 8575
COND_DIM = 3
INFER_STEPS = 80
INFER_LR = 0.1

LABEL_NAMES = ["Teff", "log_g", "[Fe/H]"]
LABEL_UNITS = ["K", "dex", "dex"]
