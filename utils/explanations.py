"""
utils/explanations.py – Zentrales Feature-Schema und Erklärungs-Builder.

Das FEATURE_SCHEMA wird in der JSON-Schnittstelle an das LLM übergeben
(siehe Implementierungsplan, Abschnitt 5). Es ist hier *einmal* definiert,
damit alle Notebooks (03, 04, 05, 06) und die Tool-Use-Pipeline (06)
identische Feature-Beschreibungen verwenden.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import EXPLANATIONS_DIR

# -----------------------------------------------------------------------------
# Feature-Schema
# -----------------------------------------------------------------------------
FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    # --- Kategoriale Features (category-dtype) ---
    "weathersit": {
        "type": "categorical",
        "description": "Wetterlage",
        "categories": {
            1: "klar / wenige Wolken",
            2: "Nebel / bewölkt",
            3: "leichter Regen / Schnee",
            4: "Starkregen / Eisregen / Gewitter",
        },
    },
    "mnth": {
        "type": "categorical",
        "description": "Monat (1=Januar, 12=Dezember)",
        "range": [1, 12],
    },
    "hr": {
        "type": "categorical",
        "description": "Stunde des Tages (0-23)",
        "range": [0, 23],
    },
    "weekday": {
        "type": "categorical",
        "description": "Wochentag (0=Sonntag, 6=Samstag)",
        "range": [0, 6],
    },
    # --- Numerische Features (float64) ---
    "yr": {
        "type": "binary",
        "description": "Jahr (0=2011, 1=2012)",
        "categories": {0: "2011", 1: "2012"},
    },
    "holiday": {
        "type": "binary",
        "description": "Feiertag (0=kein Feiertag, 1=Feiertag)",
        "categories": {0: "kein Feiertag", 1: "Feiertag"},
    },
    "temp": {
        "type": "numerical",
        "description": "Normalisierte Temperatur in Celsius (geteilt durch 41)",
        "range": [0.0, 1.0],
    },
    "hum": {
        "type": "numerical",
        "description": "Normalisierte Luftfeuchtigkeit (geteilt durch 100)",
        "range": [0.0, 1.0],
    },
    "windspeed": {
        "type": "numerical",
        "description": "Normalisierte Windgeschwindigkeit (geteilt durch 67)",
        "range": [0.0, 1.0],
    },
    # Entfernte Features (redundant, nicht im Modell):
    #   season     → aus mnth ableitbar
    #   workingday → aus weekday + holiday ableitbar
}

TARGET_DESCRIPTION: dict[str, Any] = {
    "name": "cnt",
    "description": (
        "Anzahl der ausgeliehenen Fahrräder pro Stunde "
        "(Summe aus Casual- und Registered-Nutzern)"
    ),
    "type": "count",
}


# -----------------------------------------------------------------------------
# Interne Hilfsfunktionen
# -----------------------------------------------------------------------------

# Module-level cache: model object → shap.TreeExplainer (avoid re-creating per call)
_shap_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_shap_explainer(model: Any) -> Any:
    if model not in _shap_cache:
        import shap
        _shap_cache[model] = shap.TreeExplainer(model)
    return _shap_cache[model]


def _feat_value(val: Any) -> Any:
    """Konvertiert numpy-Skalare in Python-native Typen für JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _global_xgb(model: Any, X_train: pd.DataFrame) -> tuple[list[str], list[float], float]:
    """SHAP mean |value| über Trainingsset als globale Importance für XGBoost."""
    explainer = _get_shap_explainer(model)
    shap_vals = explainer.shap_values(X_train)
    importance = np.abs(shap_vals).mean(axis=0).tolist()
    base_value = float(explainer.expected_value)
    return X_train.columns.tolist(), importance, base_value


def _global_ebm(model: Any) -> tuple[list[str], list[float], float]:
    """EBM term importances (nur Haupteffekte, ohne Interaktionen)."""
    gexp = model.explain_global()
    gd = gexp.data()
    names = gd["names"]
    scores = [float(s) for s in gd["scores"]]
    # Intercept steckt in explain_local; für global nutzen wir predict-Mittelwert
    # als Näherung des Basiswerts (wird in build_global übergeben).
    main_names, main_scores = [], []
    for n, s in zip(names, scores):
        if " & " not in n:  # Interaktionsterme ausschließen
            main_names.append(n)
            main_scores.append(s)
    return main_names, main_scores, 0.0  # base_value über Trainingsset separat


def _local_xgb(
    model: Any, X_train: pd.DataFrame, instance: pd.DataFrame
) -> tuple[dict[str, float], float, float]:
    """SHAP-Werte (log-Raum) + Basiswert + Vorhersage für eine Instanz."""
    explainer = _get_shap_explainer(model)
    shap_vals = explainer.shap_values(instance)[0]
    contribs = {col: float(v) for col, v in zip(instance.columns, shap_vals)}
    base_value = float(explainer.expected_value)
    prediction = float(model.predict(instance)[0])
    return contribs, base_value, prediction


def _local_ebm(
    model: Any, instance: pd.DataFrame
) -> tuple[dict[str, float], float, float]:
    """EBM-Beiträge (log-Raum) + Intercept + Vorhersage für eine Instanz."""
    lexp = model.explain_local(instance)
    d = lexp.data(0)
    contribs = {n: float(s) for n, s in zip(d["names"], d["scores"])}
    base_value = float(d["extra"]["scores"][0])  # Intercept
    prediction = float(d["perf"]["predicted"])
    return contribs, base_value, prediction


# -----------------------------------------------------------------------------
# Öffentliche Builder
# -----------------------------------------------------------------------------

def build_global(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    metrics: dict[str, float],
) -> dict:
    """
    Erstellt eine globale Erklärungsstruktur (JSON-serialisierbar).

    Rückgabe:
        {
          "model": str,
          "task": TARGET_DESCRIPTION,
          "feature_schema": FEATURE_SCHEMA,
          "metrics": {...},
          "base_value": float,          # log-Raum (Poisson) oder cnt-Raum (RMSE)
          "global_importance": [
              {"feature": str, "importance": float, "rank": int}, ...
          ]
        }
    """
    if model_name == "xgb":
        names, scores, base_value = _global_xgb(model, X_train)
    elif model_name == "ebm":
        names, scores, _ = _global_ebm(model)
        # EBM-Basiswert: Mittelwert der Trainingsvorhersagen (log-Raum)
        import numpy as np
        base_value = float(np.log(model.predict(X_train)).mean())
    else:
        raise ValueError(f"Unbekannter model_name: {model_name!r}")

    ranked = sorted(
        zip(names, scores), key=lambda x: -x[1]
    )
    return {
        "model": model_name,
        "task": TARGET_DESCRIPTION,
        "feature_schema": FEATURE_SCHEMA,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "base_value": round(base_value, 6),
        "global_importance": [
            {"feature": f, "importance": round(imp, 6), "rank": i + 1}
            for i, (f, imp) in enumerate(ranked)
        ],
    }


def build_local(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    instance_id: int,
) -> dict:
    """
    Erstellt eine lokale Erklärungsstruktur für eine einzelne Test-Instanz.

    instance_id ist der Positions-Index im X_test DataFrame (iloc).

    Rückgabe:
        {
          "model": str,
          "instance_id": int,
          "feature_values": {feature: value, ...},
          "y_true": float,
          "prediction": float,          # original Skala (cnt)
          "base_value": float,          # log-Raum
          "contribution_space": "log",
          "contributions": [
              {"feature": str, "value": ..., "contribution": float}, ...
          ]
        }
    """
    instance = X_test.iloc[[instance_id]]
    y_true = float(y_test.iloc[instance_id])

    if model_name == "xgb":
        contribs, base_value, prediction = _local_xgb(model, X_test, instance)
    elif model_name == "ebm":
        contribs, base_value, prediction = _local_ebm(model, instance)
    else:
        raise ValueError(f"Unbekannter model_name: {model_name!r}")

    feature_values = {
        col: _feat_value(instance.iloc[0][col])
        for col in instance.columns
    }

    contributions_list = sorted(
        [
            {
                "feature": f,
                "value": feature_values.get(f),
                "contribution": round(float(c), 6),
            }
            for f, c in contribs.items()
            if f in feature_values  # Interaktionsterme beim EBM überspringen
        ],
        key=lambda x: -abs(x["contribution"]),
    )

    return {
        "model": model_name,
        "instance_id": instance_id,
        "feature_values": feature_values,
        "y_true": y_true,
        "prediction": round(prediction, 4),
        "base_value": round(base_value, 6),
        "contribution_space": "log",
        "contributions": contributions_list,
    }


def save_explanation(data: dict, filename: str, out_dir: Path | None = None) -> Path:
    """Speichert ein Erklärungsdict als JSON."""
    import json
    out_dir = Path(out_dir) if out_dir is not None else EXPLANATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path
