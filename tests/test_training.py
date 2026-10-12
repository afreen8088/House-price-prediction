"""
Integration and regression tests for model training pipeline in Karnataka, India.
Runs training on explicitly synthetic test fixtures in INR, verifies metrics evaluation,
and validates single pipeline inference.
"""

import os
import tempfile
import pytest
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, ALL_FEATURES
from src.train import train_and_evaluate


@pytest.fixture
def synthetic_training_fixture():
    """Create a temporary synthetic dataset fixture for fast reproducible training tests in INR."""
    rng = np.random.default_rng(123)
    n = 60
    sqfts = rng.uniform(800, 3500, size=n)
    beds = rng.integers(1, 5, size=n)
    baths = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n)
    locations = rng.choice(["Bengaluru", "Mysuru", "Kalaburagi"], size=n)

    # Hedonic valuation formula in INR for synthetic test
    prices = 1500000 + (sqfts * 5000) + (beds * 300000) + (baths * 200000) + rng.normal(0, 50000, size=n)

    df = pd.DataFrame({
        "sqft": sqfts,
        "bedrooms": beds,
        "bathrooms": baths,
        "location": locations,
        "price": prices
    })
    return df


def test_end_to_end_training_pipeline_on_synthetic_fixture(synthetic_training_fixture):
    """
    Test end-to-end training on a small, explicitly labeled synthetic fixture in a temporary directory.
    Verifies that preprocessing + XGBRegressor trains, computes metrics in INR, and produces a valid model.joblib.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_csv = os.path.join(tmp_dir, "synthetic_housing.csv")
        synthetic_training_fixture.to_csv(raw_csv, index=False)

        proc_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(proc_dir, exist_ok=True)

        # Preprocess synthetic fixture
        preprocessor = build_preprocessor()
        train_df = synthetic_training_fixture.iloc[:48]
        test_df = synthetic_training_fixture.iloc[48:]

        preprocessor.fit(train_df[ALL_FEATURES])
        X_train_trans = preprocessor.transform(train_df[ALL_FEATURES])
        X_test_trans = preprocessor.transform(test_df[ALL_FEATURES])
        feature_names = preprocessor.get_feature_names_out()

        pd.DataFrame(X_train_trans, columns=feature_names).to_parquet(
            os.path.join(proc_dir, "X_train.parquet"), index=False
        )
        pd.DataFrame(X_test_trans, columns=feature_names).to_parquet(
            os.path.join(proc_dir, "X_test.parquet"), index=False
        )
        train_df[["price"]].to_parquet(os.path.join(proc_dir, "y_train.parquet"), index=False)
        test_df[["price"]].to_parquet(os.path.join(proc_dir, "y_test.parquet"), index=False)
        joblib.dump(preprocessor, os.path.join(proc_dir, "preprocessor.joblib"))

        # Write params.yaml for test
        params_file = os.path.join(tmp_dir, "params.yaml")
        model_out = os.path.join(tmp_dir, "test_model.joblib")
        metrics_out = os.path.join(tmp_dir, "test_metrics.json")

        with open(params_file, "w") as f:
            f.write(f"""
train:
  random_state: 42
  n_estimators: 20
  max_depth: 3
  learning_rate: 0.1
  subsample: 0.9
  colsample_bytree: 0.9
  model_output_path: "{model_out.replace('\\', '/')}"
  metrics_output_path: "{metrics_out.replace('\\', '/')}"
  experiment_name: "test_experiment"
""")

        # Execute training
        results = train_and_evaluate(
            params_path=params_file,
            processed_dir=proc_dir,
            raw_data_path=raw_csv
        )

        assert os.path.exists(model_out), "Trained model artifact was not created"
        assert os.path.exists(metrics_out), "Metrics output file was not created"

        # Check metrics validity
        assert results["rmse"] > 0
        assert results["mae"] > 0
        assert results["r2"] > 0.5

        # Verify inference pipeline accepts raw DataFrame directly
        loaded_pipeline = joblib.load(model_out)
        sample_input = pd.DataFrame([{
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "Bengaluru"
        }])
        pred = loaded_pipeline.predict(sample_input)
        assert len(pred) == 1
        assert pred[0] > 0
