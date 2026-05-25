import io
import json

import h5py
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import INPUT_DIM, LABEL_NAMES, LABEL_UNITS
from .fits_parser import parse_fits_spectrum
from .model_service import get_service

app = FastAPI(
    title="Stellar cINN API",
    description="Przewodzenie parametrów gwiazd z widma spektralnego.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpectrumRequest(BaseModel):
    spectrum: list[float] = Field(..., min_length=INPUT_DIM, max_length=INPUT_DIM)


class PredictResponse(BaseModel):
    labels: dict[str, float]
    nll: float


@app.on_event("startup")
def load_model() -> None:
    get_service()


@app.get("/api/health")
def health() -> dict:
    from .model_service import DEVICE

    get_service()
    return {
        "status": "ok",
        "input_dim": INPUT_DIM,
        "device": str(DEVICE),
    }


@app.get("/api/metadata")
def metadata() -> dict:
    return {
        "input_dim": INPUT_DIM,
        "label_names": LABEL_NAMES,
        "label_units": LABEL_UNITS,
        "supported_formats": ["csv", "json", "npy", "h5", "fits", "fit"],
        "description": (
            "Prześlij widmo o długości 8575 punktów (float) lub plik FITS "
            "(mwmStar/APOGEE z kolumną flux). Model zwróci Teff [K], log g oraz [Fe/H]."
        ),
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict_json(body: SpectrumRequest) -> PredictResponse:
    svc = get_service()
    try:
        result = svc.infer_labels(np.array(body.spectrum, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)


def _parse_upload(content: bytes, filename: str) -> np.ndarray:
    name = (filename or "").lower()

    if name.endswith((".fits", ".fit", ".fts")):
        return parse_fits_spectrum(content)

    if name.endswith(".h5") or name.endswith(".hdf5"):
        with h5py.File(io.BytesIO(content), "r") as f:
            if "spectra" not in f:
                raise ValueError("Plik H5 musi zawierać dataset 'spectra'.")
            data = np.asarray(f["spectra"], dtype=np.float32)
            if data.ndim == 2:
                return data[0]
            return data.reshape(-1)

    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(content), allow_pickle=False)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    text = content.decode("utf-8", errors="replace").strip()
    if name.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "spectrum" in parsed:
            values = parsed["spectrum"]
        else:
            values = parsed
    else:
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if len(lines) == 1 and ("," in lines[0] or ";" in lines[0]):
            sep = "," if "," in lines[0] else ";"
            values = [float(v) for v in lines[0].split(sep)]
        else:
            values = [float(ln.split(",")[0]) for ln in lines]

    return np.asarray(values, dtype=np.float32).reshape(-1)


@app.post("/api/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)) -> PredictResponse:
    content = await file.read()
    try:
        spectrum = _parse_upload(content, file.filename or "")
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Nie udało się odczytać pliku: {exc}") from exc

    svc = get_service()
    try:
        result = svc.infer_labels(spectrum)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)
