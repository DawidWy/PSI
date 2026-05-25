import os

import numpy as np
import torch
import torch.nn as nn
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .config import (
    COND_DIM,
    H5_PATH,
    INFER_LR,
    INFER_STEPS,
    INPUT_DIM,
    MODEL_PATH,
    NORM_STATS_PATH,
)

def _pick_device() -> torch.device:
    if os.environ.get("STELLAR_USE_CUDA", "").lower() in ("1", "true", "yes"):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _pick_device()


def subnet_fc(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim),
    )


def build_cinn(input_dim: int, cond_dim: int, n_blocks: int = 6) -> Ff.SequenceINN:
    inn = Ff.SequenceINN(input_dim)
    for _ in range(n_blocks):
        inn.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(cond_dim,),
            subnet_constructor=subnet_fc,
            affine_clamping=2.0,
            permute_soft=False,
        )
    return inn


def _load_or_compute_norm_stats() -> dict[str, np.ndarray]:
    if NORM_STATS_PATH.exists():
        data = np.load(NORM_STATS_PATH)
        return {k: data[k] for k in data.files}

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {H5_PATH} — potrzebny do statystyk normalizacji."
        )

    with h5py.File(H5_PATH, "r") as f:
        x = torch.tensor(f["spectra"][:], dtype=torch.float32)
        c = torch.tensor(f["stellar_labels"][:], dtype=torch.float32)

    stats = {
        "x_mean": x.mean(0, keepdim=True).cpu().numpy(),
        "x_std": x.std(0, keepdim=True).clamp_min(1e-6).cpu().numpy(),
        "c_mean": c.mean(0, keepdim=True).cpu().numpy(),
        "c_std": c.std(0, keepdim=True).clamp_min(1e-6).cpu().numpy(),
    }
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NORM_STATS_PATH, **stats)
    return stats


class StellarInferenceService:
    def __init__(self) -> None:
        self.stats = _load_or_compute_norm_stats()
        self.x_mean = torch.tensor(self.stats["x_mean"], dtype=torch.float32, device=DEVICE)
        self.x_std = torch.tensor(self.stats["x_std"], dtype=torch.float32, device=DEVICE)
        self.c_mean = torch.tensor(self.stats["c_mean"], dtype=torch.float32, device=DEVICE)
        self.c_std = torch.tensor(self.stats["c_std"], dtype=torch.float32, device=DEVICE)

        self.model = build_cinn(INPUT_DIM, COND_DIM)
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()

    def normalize_spectrum(self, spectrum: np.ndarray) -> torch.Tensor:
        x = np.asarray(spectrum, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != INPUT_DIM:
            raise ValueError(f"Widmo musi mieć {INPUT_DIM} punktów, otrzymano {x.shape[1]}.")
        t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        return (t - self.x_mean) / self.x_std

    @torch.enable_grad()
    def infer_labels(self, spectrum: np.ndarray, steps: int = INFER_STEPS) -> dict:
        x_norm = self.normalize_spectrum(spectrum)
        c_opt = torch.zeros(1, COND_DIM, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([c_opt], lr=INFER_LR)

        for _ in range(steps):
            optimizer.zero_grad()
            z, log_jac_det = self.model(x_norm, c=[c_opt])
            nll = 0.5 * (z**2).sum(dim=1).mean() - log_jac_det.mean()
            nll.backward()
            optimizer.step()

        with torch.no_grad():
            z, log_jac_det = self.model(x_norm, c=[c_opt])
            nll_final = (0.5 * (z**2).sum(dim=1) - log_jac_det).item()

        labels_norm = c_opt.detach()
        labels = (labels_norm * self.c_std + self.c_mean).squeeze(0).cpu().numpy()
        return {
            "labels": {name: float(v) for name, v in zip(
                ["Teff", "log_g", "[Fe/H]"], labels
            )},
            "nll": float(nll_final),
        }


_service: StellarInferenceService | None = None


def get_service() -> StellarInferenceService:
    global _service
    if _service is None:
        _service = StellarInferenceService()
    return _service
