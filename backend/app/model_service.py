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


def subnet_fc(in_dim: int, out_dim: int, hidden_dim: int = 512, layers: int = 2) -> nn.Sequential:
    if layers == 2:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    if layers == 3:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    raise ValueError(f"Nieobsługiwana liczba warstw subnetu: {layers}")


def build_cinn(
    input_dim: int,
    cond_dim: int,
    n_blocks: int = 6,
    hidden_dim: int = 512,
    subnet_layers: int = 2,
) -> Ff.SequenceINN:
    inn = Ff.SequenceINN(input_dim)
    for _ in range(n_blocks):
        inn.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(cond_dim,),
            subnet_constructor=lambda in_d, out_d: subnet_fc(
                in_d, out_d, hidden_dim=hidden_dim, layers=subnet_layers
            ),
            affine_clamping=2.0,
            permute_soft=False,
        )
    return inn


def _infer_architecture_from_state(state: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    block_ids = {
        int(k.split(".")[1])
        for k in state.keys()
        if k.startswith("module_list.") and ".global_scale" in k
    }
    if not block_ids:
        raise ValueError("Checkpoint nie zawiera oczekiwanych kluczy module_list.*.global_scale.")

    n_blocks = max(block_ids) + 1
    input_dim = int(state["module_list.0.global_scale"].shape[1])
    subnet_in = int(state["module_list.0.subnet.0.weight"].shape[1])
    cond_dim = subnet_in - input_dim + (input_dim // 2)
    subnet_layers = 3 if "module_list.0.subnet.4.weight" in state else 2

    return input_dim, cond_dim, n_blocks, subnet_layers


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

        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model_input_dim, model_cond_dim, model_blocks, model_subnet_layers = _infer_architecture_from_state(state)

        valid_shapes = {
            (INPUT_DIM, COND_DIM),
            (COND_DIM, INPUT_DIM),
        }
        if (model_input_dim, model_cond_dim) not in valid_shapes:
            raise ValueError(
                "Niezgodne wymiary modelu w checkpointcie: "
                f"input_dim={model_input_dim}, cond_dim={model_cond_dim}. "
                f"Oczekiwano jednej z par: {sorted(valid_shapes)}."
            )

        self.predict_labels_directly = (model_input_dim, model_cond_dim) == (COND_DIM, INPUT_DIM)
        self.model = build_cinn(
            model_input_dim,
            model_cond_dim,
            n_blocks=model_blocks,
            subnet_layers=model_subnet_layers,
        )
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
        labels_norm = torch.zeros(1, COND_DIM, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([labels_norm], lr=INFER_LR)

        for _ in range(steps):
            optimizer.zero_grad()
            if self.predict_labels_directly:
                z, log_jac_det = self.model(labels_norm, c=[x_norm])
            else:
                z, log_jac_det = self.model(x_norm, c=[labels_norm])
            nll = 0.5 * (z**2).sum(dim=1).mean() - log_jac_det.mean()
            nll.backward()
            optimizer.step()

        with torch.no_grad():
            if self.predict_labels_directly:
                z, log_jac_det = self.model(labels_norm, c=[x_norm])
            else:
                z, log_jac_det = self.model(x_norm, c=[labels_norm])
            nll_final = (0.5 * (z**2).sum(dim=1) - log_jac_det).item()

        labels = (labels_norm.detach() * self.c_std + self.c_mean).squeeze(0).cpu().numpy()
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
