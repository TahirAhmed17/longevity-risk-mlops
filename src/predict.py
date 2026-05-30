"""
predict.py — Inference module for the Longevity Risk Predictor.

This module exposes a simple ``predict_risk`` function that:
  1. Loads the persisted model (``models/longevity_model.pkl``) and
     scaler (``models/scaler.pkl``) artifacts.
  2. Scales the incoming feature vector.
  3. Produces a probability estimate and applies a configurable
     decision threshold.
  4. Returns a probability, binary label, and human-readable risk string.

Usage:
    from src.predict import predict_risk

    prob, label, risk = predict_risk([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1])
"""

import json
import os
from typing import Optional

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_model_path() -> str:
    """Return the default path to the serialised model."""
    return os.path.join(_get_project_root(), "models", "longevity_model.pkl")


def _default_scaler_path() -> str:
    """Return the default path to the serialised scaler."""
    return os.path.join(_get_project_root(), "models", "scaler.pkl")


def _default_threshold_path() -> str:
    """Return the default path to the threshold JSON file."""
    return os.path.join(_get_project_root(), "models", "threshold.json")


# ---------------------------------------------------------------------------
# Model / scaler loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> tuple:
    """Load the saved Random-Forest model and StandardScaler.

    Parameters
    ----------
    model_path : str or None
        Explicit path to the ``.pkl`` model file.
        Defaults to ``models/longevity_model.pkl``.
    scaler_path : str or None
        Explicit path to the ``.pkl`` scaler file.
        Defaults to ``models/scaler.pkl``.

    Returns
    -------
    model : sklearn estimator
        The trained classifier.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler.

    Raises
    ------
    FileNotFoundError
        If either artifact file does not exist.
    """
    model_path = model_path or _default_model_path()
    scaler_path = scaler_path or _default_scaler_path()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.  "
            "Train the model first by running `python -m src.train`."
        )
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at '{scaler_path}'.  "
            "Run preprocessing first to generate the scaler."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[predict] Model  loaded from {model_path}")
    print(f"[predict] Scaler loaded from {scaler_path}")
    return model, scaler


def _load_optimal_threshold() -> float:
    """Load the optimal threshold from the saved JSON file.

    Returns
    -------
    float
        The optimal probability threshold, or ``0.5`` if the file is
        not found (graceful fallback).
    """
    path = _default_threshold_path()
    if os.path.isfile(path):
        with open(path, "r") as fp:
            data = json.load(fp)
        return float(data.get("optimal_threshold", 0.5))
    return 0.5


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_risk(
    features: list | np.ndarray,
    model=None,
    scaler=None,
    threshold: Optional[float] = None,
) -> tuple[float, int, str]:
    """Predict the longevity / heart-disease risk for a single patient.

    Parameters
    ----------
    features : list or np.ndarray
        A 1-D array of 13 clinical feature values in the order returned
        by ``preprocess.get_feature_names()``.
    model : sklearn estimator or None
        Pre-loaded model.  Loaded on demand when *None*.
    scaler : StandardScaler or None
        Pre-loaded scaler.  Loaded on demand when *None*.
    threshold : float or None
        Decision threshold.  When *None*, the optimal threshold saved
        during training is used (falls back to 0.5).

    Returns
    -------
    probability : float
        Predicted probability for the positive (HIGH RISK) class.
    prediction : int
        Binary prediction (1 = high risk, 0 = low risk).
    risk_label : str
        Human-readable string: ``"HIGH RISK"`` or ``"LOW RISK"``.
    """
    # Load artifacts lazily if not supplied
    if model is None or scaler is None:
        model, scaler = load_model()

    # Resolve threshold
    if threshold is None:
        threshold = _load_optimal_threshold()

    # Ensure the feature vector is a 2-D numpy array
    features_arr = np.asarray(features, dtype=float).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features_arr)

    # Predict probability for the positive class
    probability = float(model.predict_proba(features_scaled)[0, 1])

    # Apply threshold
    prediction = int(probability >= threshold)
    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

    return probability, prediction, risk_label


# ---------------------------------------------------------------------------
# CLI entry-point — example prediction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Longevity Risk Predictor — Inference Demo")
    print("=" * 60)

    # Example patient features (same order as get_feature_names):
    # age, sex, cp, trestbps, chol, fbs, restecg,
    # thalach, exang, oldpeak, slope, ca, thal
    example_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]

    try:
        prob, pred, label = predict_risk(example_features)
        print(f"\n  Input features : {example_features}")
        print(f"  Probability    : {prob:.4f}")
        print(f"  Prediction     : {pred}")
        print(f"  Risk label     : {label}")
    except FileNotFoundError as exc:
        print(f"\n  [ERROR] {exc}")
        print("  Please run `python -m src.train` to train the model first.")

    print("=" * 60)
