"""
preprocess.py — Data preprocessing module for the Longevity Risk Predictor.

This module handles:
  1. Loading the UCI Cleveland Heart Disease dataset.
  2. Cleaning missing / placeholder values.
  3. Encoding categorical features.
  4. Feature scaling (StandardScaler) with artifact persistence.
  5. Class balancing via SMOTE.
  6. Stratified train / test splitting.

Usage:
    from src.preprocess import load_data, preprocess_data, get_feature_names
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_project_root() -> str:
    """Return the absolute path to the project root directory.

    The project root is assumed to be two levels above this file:
        <project_root>/src/preprocess.py  →  <project_root>
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_path(filepath: str | None = None) -> str:
    """Resolve the path to the heart disease CSV file.

    Parameters
    ----------
    filepath : str or None
        Explicit path to a CSV file.  When *None* the default location
        ``<project_root>/data/heart.csv`` is used.

    Returns
    -------
    str
        Absolute path to the CSV file.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    if filepath is not None:
        resolved = os.path.abspath(filepath)
    else:
        resolved = os.path.join(_get_project_root(), "data", "heart.csv")

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Dataset not found at '{resolved}'.  "
            "Pass the correct path or ensure data/heart.csv exists."
        )
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_names() -> list[str]:
    """Return the ordered list of feature column names (excludes target).

    Returns
    -------
    list[str]
        The 13 clinical feature names used by the model.
    """
    return [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str or None
        Path to the CSV.  Defaults to ``<project_root>/data/heart.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    path = _resolve_data_path(filepath)
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df)} records from {path}")
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline on a raw dataframe.

    Steps
    -----
    1. Replace ``'?'`` placeholders with ``NaN``.
    2. Cast every column to numeric (coerce errors to ``NaN``).
    3. Impute missing numeric values with the column median,
       missing categorical values with the column mode.
    4. Encode categorical columns with ``LabelEncoder``.
    5. Separate features (*X*) and target (*y*).
    6. Scale features with ``StandardScaler`` and persist the
       fitted scaler to ``models/scaler.pkl``.
    7. Balance classes with SMOTE.
    8. Perform a stratified 80 / 20 train–test split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by :func:`load_data`.

    Returns
    -------
    X_train : np.ndarray
        Training features (scaled, SMOTE-balanced).
    X_test : np.ndarray
        Test features (scaled, from original distribution).
    y_train : np.ndarray
        Training labels (SMOTE-balanced).
    y_test : np.ndarray
        Test labels (from original distribution).
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1–2. Replace '?' with NaN and coerce to numeric
    # ------------------------------------------------------------------
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 3. Impute missing values
    # ------------------------------------------------------------------
    # Categorical-like features (low cardinality integers)
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != "target"]

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ------------------------------------------------------------------
    # 4. Encode categorical features with LabelEncoder
    # ------------------------------------------------------------------
    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(int).astype(str))
            label_encoders[col] = le

    # ------------------------------------------------------------------
    # 5. Separate features (X) and target (y)
    # ------------------------------------------------------------------
    feature_names = get_feature_names()
    X = df[feature_names].values
    y = df["target"].values.astype(int)

    print(f"[preprocess] Features shape: {X.shape}  |  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 6. Scale features and persist the scaler
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir = os.path.join(_get_project_root(), "models")
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[preprocess] Scaler saved to {scaler_path}")

    # ------------------------------------------------------------------
    # 7. Train / test split  (BEFORE SMOTE — avoids data leakage)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[preprocess] Train size (pre-SMOTE): {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 8. Apply SMOTE to training data only
    # ------------------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(
        f"[preprocess] Train size (post-SMOTE): {X_train.shape[0]}  |  "
        f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Longevity Risk Predictor — Preprocessing Pipeline")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n--- Summary ---")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}  (balanced)")
    print(f"  y_test  : {y_test.shape}")
    print("=" * 60)
