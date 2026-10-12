"""
Data preprocessing and feature engineering pipeline for House Price Prediction in Karnataka.
Handles schema validation, train/test splitting, median imputation for numerical features,
categorical imputation, scaling, one-hot encoding, and Parquet persistence.
"""

import os
import sys
import argparse
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = ["sqft", "bedrooms", "bathrooms"]
CATEGORICAL_FEATURES = ["location"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "price"


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset with schema validation and backwards-compatibility mapping."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Raw dataset at {file_path} is empty.")
    
    # Backwards-compatibility alias: if 'neighborhood' is present instead of 'location'
    if "neighborhood" in df.columns and "location" not in df.columns:
        df["location"] = df["neighborhood"]
    
    # Validate required columns
    required_cols = ALL_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}. Expected: {required_cols}")
    
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the scikit-learn ColumnTransformer for preprocessing.
    - Numerical: Median imputer + StandardScaler
    - Categorical: Most frequent imputer + OneHotEncoder (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def preprocess_and_save(
    raw_data_path: str = "data/raw/housing_data.csv",
    processed_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    save_preprocessor: bool = True
):
    """
    Execute full preprocessing pipeline preventing data leakage:
    1. Loads dataset.
    2. Drops rows with missing target.
    3. Splits into train and test sets with reproducible seed.
    4. Fits transformations strictly on training set.
    5. Transforms both sets into processed DataFrames.
    6. Persists to Parquet format.
    """
    print(f"Loading raw data from: {raw_data_path}")
    df = load_data(raw_data_path)

    # Drop target nulls if any
    initial_len = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing target '{TARGET_COLUMN}'")

    X = df[ALL_FEATURES]
    y = df[[TARGET_COLUMN]]

    # Train/test split strictly before any transformer fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    preprocessor = build_preprocessor()

    # Fit ONLY on training data to prevent data leakage
    preprocessor.fit(X_train)

    # Transform train and test
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Ensure output directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Save processed datasets to Parquet
    x_train_path = os.path.join(processed_dir, "X_train.parquet")
    x_test_path = os.path.join(processed_dir, "X_test.parquet")
    y_train_path = os.path.join(processed_dir, "y_train.parquet")
    y_test_path = os.path.join(processed_dir, "y_test.parquet")

    X_train_processed.to_parquet(x_train_path, index=False)
    X_test_processed.to_parquet(x_test_path, index=False)
    y_train.to_parquet(y_train_path, index=False)
    y_test.to_parquet(y_test_path, index=False)

    print(f"Processed datasets successfully saved to {processed_dir}:")
    print(f"  - {x_train_path} ({X_train_processed.shape})")
    print(f"  - {x_test_path} ({X_test_processed.shape})")
    print(f"  - {y_train_path} ({y_train.shape})")
    print(f"  - {y_test_path} ({y_test.shape})")

    if save_preprocessor:
        preprocessor_path = os.path.join(processed_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Saved fitted preprocessor to {preprocessor_path}")

    return {
        "x_train_path": x_train_path,
        "x_test_path": x_test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Karnataka house price data.")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    raw_path = "data/raw/housing_data.csv"
    processed_dir = "data/processed"
    test_size = 0.2
    random_state = 42

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            if params and "preprocess" in params:
                prep_cfg = params["preprocess"]
                raw_path = prep_cfg.get("raw_data_path", raw_path)
                processed_dir = prep_cfg.get("processed_dir", processed_dir)
                test_size = prep_cfg.get("test_size", test_size)
                random_state = prep_cfg.get("random_state", random_state)

    preprocess_and_save(
        raw_data_path=raw_path,
        processed_dir=processed_dir,
        test_size=test_size,
        random_state=random_state
    )


if __name__ == "__main__":
    main()
