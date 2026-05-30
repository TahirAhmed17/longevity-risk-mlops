"""
test_preprocess.py — Unit tests for the preprocessing module.

Run with:
    pytest tests/test_preprocess.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so ``src`` can be imported
# regardless of the working directory or how pytest is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import load_data, preprocess_data, get_feature_names  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_dataframe() -> pd.DataFrame:
    """Load the raw heart-disease dataframe once per test module."""
    return load_data()


@pytest.fixture(scope="module")
def preprocessed_data(raw_dataframe):
    """Run the full preprocessing pipeline once per test module.

    Returns a tuple: ``(X_train, X_test, y_train, y_test)``.
    """
    return preprocess_data(raw_dataframe)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Data Loading
# ═══════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """Tests that verify the raw dataset loads correctly."""

    def test_data_loads_successfully(self, raw_dataframe):
        """Test that data loads without errors and returns a DataFrame."""
        assert isinstance(raw_dataframe, pd.DataFrame)

    def test_data_has_correct_columns(self, raw_dataframe):
        """Test that all expected columns (features + target) are present."""
        expected_columns = get_feature_names() + ["target"]
        for col in expected_columns:
            assert col in raw_dataframe.columns, f"Missing column: {col}"

    def test_data_not_empty(self, raw_dataframe):
        """Test that the dataset contains at least one record."""
        assert len(raw_dataframe) > 0, "Dataset is empty"

    def test_target_column_exists(self, raw_dataframe):
        """Test that the 'target' column exists and contains valid values."""
        assert "target" in raw_dataframe.columns
        unique_values = set(raw_dataframe["target"].unique())
        assert unique_values.issubset({0, 1}), (
            f"Target column has unexpected values: {unique_values}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test: Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Tests for the full preprocessing pipeline output."""

    def test_preprocessing_returns_correct_shapes(self, preprocessed_data):
        """Test that preprocessing returns four arrays with compatible dims."""
        X_train, X_test, y_train, y_test = preprocessed_data

        n_features = len(get_feature_names())

        # Feature matrices must be 2-D with correct number of columns
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert X_train.shape[1] == n_features
        assert X_test.shape[1] == n_features

        # Labels must be 1-D and match their respective feature matrices
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    def test_no_null_values_after_preprocessing(self, preprocessed_data):
        """Test that no NaN / null values remain after preprocessing."""
        X_train, X_test, y_train, y_test = preprocessed_data

        assert not np.isnan(X_train).any(), "X_train contains NaN values"
        assert not np.isnan(X_test).any(), "X_test contains NaN values"
        assert not np.isnan(y_train).any(), "y_train contains NaN values"
        assert not np.isnan(y_test).any(), "y_test contains NaN values"

    def test_features_are_scaled(self, preprocessed_data):
        """Test that features have been standardised (zero-centred).

        After standard-scaling the combined (train + test) set should
        have means near zero.  We check the training set only (post-SMOTE
        the distribution is altered but should still be roughly centred).
        """
        X_train, X_test, _, _ = preprocessed_data

        # The test set, which was scaled with the training scaler, should
        # have means within a reasonable range of 0 (allow ±1.0 for small
        # datasets and SMOTE effects).
        train_means = np.abs(X_train.mean(axis=0))
        assert (train_means < 1.0).all(), (
            f"Some feature means are unexpectedly far from 0: {train_means}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test: SMOTE Balancing
# ═══════════════════════════════════════════════════════════════════════════

class TestSMOTE:
    """Tests that verify SMOTE class-balancing behaviour."""

    def test_smote_balances_classes(self, preprocessed_data):
        """Test that SMOTE produces perfectly balanced classes in training."""
        _, _, y_train, _ = preprocessed_data

        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))

        assert len(class_counts) == 2, "Expected exactly two classes"
        assert class_counts[0] == class_counts[1], (
            f"Classes are not balanced after SMOTE: {class_counts}"
        )

    def test_smote_increases_minority_class(self, raw_dataframe):
        """Test that SMOTE increases the minority class count.

        We compare the minority class size in the raw dataset to the
        balanced training set produced by ``preprocess_data``.
        """
        # Determine original minority class count
        original_counts = raw_dataframe["target"].value_counts()
        original_minority = original_counts.min()

        # Run preprocessing (separate call to avoid fixture coupling issues)
        _, _, y_train, _ = preprocess_data(raw_dataframe)

        unique, counts = np.unique(y_train, return_counts=True)
        balanced_minority = min(counts)

        assert balanced_minority >= original_minority, (
            f"SMOTE should not reduce the minority class "
            f"(original={original_minority}, balanced={balanced_minority})"
        )
