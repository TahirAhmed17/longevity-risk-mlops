"""
train.py — Model training module for the Longevity Risk Predictor.

This module:
  1. Loads and preprocesses data via ``src.preprocess``.
  2. Trains a ``RandomForestClassifier`` with balanced class weights.
  3. Selects the optimal decision threshold from the Precision–Recall curve.
  4. Logs hyper-parameters, metrics, and the model artifact to **MLflow**.
  5. Registers the model under the name ``LongevityRiskModel``.
  6. Persists the trained model to ``models/longevity_model.pkl``.

Usage:
    python -m src.train          # from the project root
    python src/train.py          # direct execution
"""

import os
import sys
import json

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src.preprocess`` can be imported
# regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import load_data, preprocess_data  # noqa: E402


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the PR curve.

    The optimal threshold is chosen as the point on the precision–recall
    curve where the F1 score (harmonic mean of precision and recall) is
    maximised.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Optimal probability threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F1 = 2 * (P * R) / (P + R);  avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    # ``precision_recall_curve`` returns one more element for precision/recall
    # than for thresholds — the last entry corresponds to threshold = 1.
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model() -> None:
    """Train a Random-Forest classifier and log everything to MLflow."""

    # ------------------------------------------------------------------
    # 1. Configure MLflow tracking
    # ------------------------------------------------------------------
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Longevity-Risk-Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"[train] MLflow tracking URI : {tracking_uri}")
    print(f"[train] MLflow experiment   : {experiment_name}")

    # ------------------------------------------------------------------
    # 2. Load & preprocess data
    # ------------------------------------------------------------------
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # ------------------------------------------------------------------
    # 3. Define hyper-parameters
    # ------------------------------------------------------------------
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced",
    }

    # ------------------------------------------------------------------
    # 4. Train inside an MLflow run
    # ------------------------------------------------------------------
    with mlflow.start_run() as run:
        print(f"\n[train] MLflow run ID: {run.info.run_id}")

        # ---- Train the model ----
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        print("[train] Model training complete.")

        # ---- Predictions & probabilities ----
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ---- Optimal threshold from PR curve ----
        threshold = _optimal_threshold(y_test, y_prob)

        # Apply the optimal threshold
        y_pred_optimal = (y_prob >= threshold).astype(int)

        # ---- Compute metrics ----
        roc_auc = roc_auc_score(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        f1 = f1_score(y_test, y_pred_optimal)

        print(f"\n[train] === Evaluation Metrics ===")
        print(f"  ROC-AUC            : {roc_auc:.4f}")
        print(f"  PR-AUC             : {pr_auc:.4f}")
        print(f"  F1 (opt threshold) : {f1:.4f}")
        print(f"  Optimal threshold  : {threshold:.4f}")

        # ---- Classification report & confusion matrix ----
        report = classification_report(y_test, y_pred_optimal)
        cm = confusion_matrix(y_test, y_pred_optimal)
        print(f"\n[train] Classification Report:\n{report}")
        print(f"[train] Confusion Matrix:\n{cm}\n")

        # ---- Log to MLflow ----
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("optimal_threshold", threshold)

        # Log the scikit-learn model as an MLflow artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="LongevityRiskModel",
        )
        print("[train] Model logged & registered in MLflow.")

        # ---- Save threshold alongside the model for inference ----
        threshold_path = os.path.join(_PROJECT_ROOT, "models", "threshold.json")
        os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
        with open(threshold_path, "w") as fp:
            json.dump({"optimal_threshold": threshold}, fp, indent=2)

    # ------------------------------------------------------------------
    # 5. Persist model locally with joblib
    # ------------------------------------------------------------------
    models_dir = os.path.join(_PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "longevity_model.pkl")
    joblib.dump(model, model_path)
    print(f"[train] Model saved to {model_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Longevity Risk Predictor — Model Training")
    print("=" * 60)
    train_model()
    print("=" * 60)
    print("  Training complete.")
    print("=" * 60)
