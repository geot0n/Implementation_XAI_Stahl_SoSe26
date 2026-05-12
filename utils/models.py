"""
utils/models.py – Speichern und Laden der trainierten Modelle.

Verwendet joblib (laut Plan empfohlen). Beide Modelle werden über
dieselben Helper-Funktionen geladen, damit alle Notebooks konsistent
auf identische Modell-Artefakte zugreifen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from . import MODELS_DIR

EBM_FILENAME = "ebm.pkl"
XGB_FILENAME = "xgb.pkl"


# ---------------------------------------------------------------------------
# Loss-Optionen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossOption:
    key: str
    label: str
    description: str
    ebm_objective: str
    xgb_objective: str
    contribution_space: str


LOSS_OPTIONS: Dict[str, LossOption] = {
    "squared_error": LossOption(
        key="squared_error",
        label="Option 1: Squared Error",
        description=(
            "Klassische quadratische Verlustfunktion. Einfach zu interpretieren, "
            "aber nicht ideal für rechtsschiefe Count-Daten — kann negative "
            "Vorhersagen liefern."
        ),
        ebm_objective="rmse",
        xgb_objective="reg:squarederror",
        contribution_space="native",
    ),
    "poisson_log": LossOption(
        key="poisson_log",
        label="Option 2: Poisson-Deviance (Beitraege auf Log-Skala)",
        description=(
            "Poisson-Deviance-Verlust. Vorhersagen strikt positiv via exp(). "
            "Beiträge werden auf der Log-Skala extrahiert und interpretiert."
        ),
        ebm_objective="poisson_deviance",
        xgb_objective="count:poisson",
        contribution_space="log",
    ),
    "poisson_native": LossOption(
        key="poisson_native",
        label="Option 3: Poisson-Deviance (Beitraege approximativ auf Ausleihe-Skala)",
        description=(
            "Identisches Modell wie Option 2. Beiträge werden approximativ auf "
            "der Ausleihe-Skala extrahiert (XGBoost output_margin=False, EBM analog)."
        ),
        ebm_objective="poisson_deviance",
        xgb_objective="count:poisson",
        contribution_space="native",
    ),
}


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Berechnet Regressionsmetriken auf der Original-Skala."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Mean Poisson deviance (nur für y_pred > 0 sinnvoll)
    eps = 1e-8
    pred_pos = np.clip(y_pred, eps, None)
    poisson_deviance = float(
        2.0 * np.mean(y_true * np.log((y_true + eps) / pred_pos) - (y_true - pred_pos))
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "poisson_deviance": poisson_deviance,
        "min_prediction": float(y_pred.min()),
        "n_negative_predictions": int((y_pred < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Generisches Speichern / Laden
# ---------------------------------------------------------------------------

def save_model(model: Any, model_type: str, loss_key: str,
               models_dir: Path | str | None = None) -> Path:
    """Speichert ein Modell unter ``{model_type}_{loss_key}.pkl``."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_type}_{loss_key}.pkl"
    joblib.dump(model, path)
    return path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Speichern
# -----------------------------------------------------------------------------
def save_ebm(model: Any, models_dir: Path | str | None = None) -> Path:
    """Speichert ein trainiertes EBM-Modell."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    _ensure_dir(models_dir)
    path = models_dir / EBM_FILENAME
    joblib.dump(model, path)
    return path


def save_xgb(model: Any, models_dir: Path | str | None = None) -> Path:
    """Speichert ein trainiertes XGBoost-Modell."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    _ensure_dir(models_dir)
    path = models_dir / XGB_FILENAME
    joblib.dump(model, path)
    return path


# -----------------------------------------------------------------------------
# Laden
# -----------------------------------------------------------------------------
def load_ebm(models_dir: Path | str | None = None) -> Any:
    """Lädt das gespeicherte EBM-Modell."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = models_dir / EBM_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"EBM-Modell nicht gefunden unter {path}. "
            "Bitte zuerst Notebook 02_Modeling.ipynb ausführen."
        )
    return joblib.load(path)


def load_xgb(models_dir: Path | str | None = None) -> Any:
    """Lädt das gespeicherte XGBoost-Modell."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = models_dir / XGB_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost-Modell nicht gefunden unter {path}. "
            "Bitte zuerst Notebook 02_Modeling.ipynb ausführen."
        )
    return joblib.load(path)


def load_models(models_dir: Path | str | None = None) -> Tuple[Any, Any]:
    """Lädt beide Modelle in einem Aufruf. Returns: (ebm, xgb)."""
    return load_ebm(models_dir), load_xgb(models_dir)
