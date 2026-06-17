import io
from typing import BinaryIO

import numpy as np
from astropy.io import fits

from .config import INPUT_DIM

FLUX_COLUMNS = ("flux", "FLUX", "flx", "spec", "spectrum")
SPECTRUM_COLUMNS = FLUX_COLUMNS + ("nmf_rectified_model_flux",)


def _flux_from_row(table, row_idx: int, column: str) -> np.ndarray | None:
    if column not in table.names:
        return None
    values = np.asarray(table[column][row_idx], dtype=np.float32).reshape(-1)
    if values.size == 0:
        return None
    return values


def _flux_from_image(hdu) -> np.ndarray | None:
    data = hdu.data
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1 and arr.size > 0:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0].reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].reshape(-1)
    return None


def parse_fits_spectrum(source: bytes | BinaryIO, target_dim: int = INPUT_DIM) -> np.ndarray:
    fileobj = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    candidates: list[tuple[str, np.ndarray]] = []

    with fits.open(fileobj, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            is_table = hasattr(hdu.data, "names") and hdu.data.names is not None
            if not is_table:
                img_flux = _flux_from_image(hdu)
                if img_flux is not None and img_flux.size > 0:
                    candidates.append((f"{hdu.name or 'PRIMARY'}:image", img_flux))
                continue

            for col in SPECTRUM_COLUMNS:
                if col not in hdu.data.names:
                    continue
                for row_idx in range(len(hdu.data)):
                    flux = _flux_from_row(hdu.data, row_idx, col)
                    if flux is not None:
                        label = f"{hdu.name or 'HDU'}:{col}[{row_idx}]"
                        candidates.append((label, flux))

    if not candidates:
        raise ValueError(
            "Nie znaleziono widma w pliku FITS. "
            "Oczekiwana kolumna 'flux' w tabeli lub dane 1D o długości "
            f"{target_dim}."
        )

    exact = [(n, f) for n, f in candidates if f.size == target_dim]
    if exact:
        return exact[0][1].astype(np.float32, copy=False)

    candidates.sort(key=lambda item: abs(item[1].size - target_dim))
    name, flux = candidates[0]
    if flux.size != target_dim:
        raise ValueError(
            f"Widmo w FITS ({name}) ma {flux.size} punktów, "
            f"wymagane {target_dim}."
        )
    return flux.astype(np.float32, copy=False)
