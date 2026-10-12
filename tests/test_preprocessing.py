"""
Unit tests for data preprocessing and feature transformations.
Validates schema validation, median imputation, categorical handling,
and scikit-learn pipeline operations.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocess import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    load_data,
    build_preprocessor,
    preprocess_and_save,
)


@pytest.fixture
def sample_raw_dataframe():
    """Create a controlled sample DataFrame with some missing values."""
    data = {
        "sqft": [1500.0, np.nan, 2200.0, 3100.0, 1800.0],
        "bedrooms": [3, 2, 4, 5, 3],
        "bathrooms": [2.0, 1.5, np.nan, 3.5, 2.0],
        "neighborhood": ["Downtown", "Oakridge", np.nan, "Riverside", "Downtown"],
        "price": [450000.0, 520000.0, 610000.0, 890000.0, 480000.0],
    }
    return pd.DataFrame(data)


def test_build_preprocessor_structure():
    """Verify preprocessor builds expected transformers."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_handles_missing_values(sample_raw_dataframe):
    """Verify median and categorical imputers fill missing values without errors."""
    X = sample_raw_dataframe[ALL_FEATURES]
    preprocessor = build_preprocessor()

    # Fit and transform
    X_trans = preprocessor.fit_transform(X)

    # Output should not contain any NaN values
    assert not np.isnan(X_trans).any(), "Transformed feature matrix contains NaNs"
    assert X_trans.shape[0] == 5


def test_preprocessor_handles_unseen_category():
    """Verify OneHotEncoder ignores unseen categories at inference time."""
    train_data = pd.DataFrame({
        "sqft": [1200.0, 2000.0],
        "bedrooms": [2, 3],
        "bathrooms": [1.5, 2.0],
        "neighborhood": ["Downtown", "Oakridge"],
    })
    test_data = pd.DataFrame({
        "sqft": [1800.0],
        "bedrooms": [3],
        "bathrooms": [2.0],
        "neighborhood": ["UnseenDistrict"],  # Category not in training data
    })

    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    # Should not raise exception
    test_trans = preprocessor.transform(test_data)
    assert test_trans.shape[0] == 1
    assert not np.isnan(test_trans).any()


def test_load_data_validation():
    """Verify load_data validates missing columns and non-existent files."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        # Missing 'neighborhood' and 'price'
        f.write("sqft,bedrooms,bathrooms\n1500,3,2\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_data(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file_path_xyz.csv")


def test_preprocess_and_save_pipeline(sample_raw_dataframe):
    """Test full preprocess_and_save workflow generating Parquet files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_csv = os.path.join(tmp_dir, "raw.csv")
        sample_raw_dataframe.to_csv(raw_csv, index=False)
        proc_dir = os.path.join(tmp_dir, "processed")

        result = preprocess_and_save(
            raw_data_path=raw_csv,
            processed_dir=proc_dir,
            test_size=0.2,
            random_state=42
        )

        assert os.path.exists(result["x_train_path"])
        assert os.path.exists(result["x_test_path"])
        assert os.path.exists(result["y_train_path"])
        assert os.path.exists(result["y_test_path"])

        # Check parquet files are loadable
        df_xtrain = pd.read_parquet(result["x_train_path"])
        assert len(df_xtrain) > 0
