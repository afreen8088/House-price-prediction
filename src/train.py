"""
Model training and evaluation module for House Price Prediction.
Trains an XGBoost Regressor, computes RMSE, MAE, and R2 evaluation metrics,
tracks experiments via MLflow, bundles preprocessing and model into a unified
inference pipeline, and persists `model.joblib` and `metrics.json`.
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import argparse
from datetime import datetime, timezone
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow
import mlflow.sklearn

from src.preprocess import ALL_FEATURES, build_preprocessor, load_data

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def load_processed_data(processed_dir: str = "data/processed"):
    """Load train and test Parquet datasets."""
    x_train_path = os.path.join(processed_dir, "X_train.parquet")
    x_test_path = os.path.join(processed_dir, "X_test.parquet")
    y_train_path = os.path.join(processed_dir, "y_train.parquet")
    y_test_path = os.path.join(processed_dir, "y_test.parquet")

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required processed dataset missing: {p}. "
                f"Please run preprocessing first ('python src/preprocess.py')."
            )

    X_train = pd.read_parquet(x_train_path)
    X_test = pd.read_parquet(x_test_path)
    y_train = pd.read_parquet(y_train_path).values.ravel()
    y_test = pd.read_parquet(y_test_path).values.ravel()

    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    params_path: str = "params.yaml",
    processed_dir: str = "data/processed",
    raw_data_path: str = "data/raw/housing_data.csv",
):
    """
    Train XGBoost Regressor, track with MLflow, evaluate, and save deployable artifacts.
    """
    # 1. Load parameters
    hyperparams = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
    }
    experiment_name = "house_price_prediction"
    model_output_path = "model.joblib"
    metrics_output_path = "metrics.json"

    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg and "train" in cfg:
                t_cfg = cfg["train"]
                for k in hyperparams:
                    if k in t_cfg:
                        hyperparams[k] = t_cfg[k]
                experiment_name = t_cfg.get("experiment_name", experiment_name)
                model_output_path = t_cfg.get("model_output_path", model_output_path)
                metrics_output_path = t_cfg.get("metrics_output_path", metrics_output_path)

    print("Loaded Hyperparameters:", hyperparams)

    # 2. Load Processed Datasets
    X_train_trans, X_test_trans, y_train, y_test = load_processed_data(processed_dir)
    print(f"Loaded processed data: {X_train_trans.shape[0]} train rows, {X_test_trans.shape[0]} test rows")

    # 3. Load preprocessor
    preprocessor_path = os.path.join(processed_dir, "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    else:
        # Fallback: re-fit preprocessor on raw data if standalone
        print("Fitted preprocessor not found in processed_dir. Fitting from raw data...")
        df_raw = load_data(raw_data_path).dropna(subset=["price"])
        from sklearn.model_selection import train_test_split
        X_raw = df_raw[ALL_FEATURES]
        y_raw = df_raw["price"]
        X_tr, _, _, _ = train_test_split(X_raw, y_raw, test_size=0.2, random_state=hyperparams["random_state"])
        preprocessor = build_preprocessor()
        preprocessor.fit(X_tr)

    # 4. Train Model
    regressor = xgb.XGBRegressor(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        learning_rate=hyperparams["learning_rate"],
        subsample=hyperparams["subsample"],
        colsample_bytree=hyperparams["colsample_bytree"],
        random_state=hyperparams["random_state"],
        objective="reg:squarederror",
    )

    print("Training XGBRegressor...")
    regressor.fit(X_train_trans, y_train)

    # 5. Evaluate on Test Set
    y_pred = regressor.predict(X_test_trans)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\n--- Model Evaluation ---")
    print(f"  RMSE:     ${rmse:,.2f}")
    print(f"  MAE:      ${mae:,.2f}")
    print(f"  R2 Score: {r2:.4f}")

    # 6. Build unified Inference Pipeline
    inference_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Verify unified pipeline on a sample raw input
    test_sample = pd.DataFrame([
        {"sqft": 2100, "bedrooms": 3, "bathrooms": 2.0, "neighborhood": "Suburban Heights"}
    ])
    sample_pred = inference_pipeline.predict(test_sample)[0]
    print(f"Inference pipeline sanity check prediction: ${sample_pred:,.2f}")

    # 7. Save Deployable Artifacts
    joblib.dump(inference_pipeline, model_output_path)
    print(f"Saved complete inference pipeline artifact to: {model_output_path}")

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "model_type": "XGBRegressor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparams,
        "features": ALL_FEATURES,
        "status": "Trained & Validated",
    }

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_output_path}")

    # 8. Track via MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.log_params(hyperparams)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        })
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("pipeline", "Scikit-learn + XGBoost")

        # Log pipeline artifact to MLflow
        try:
            mlflow.sklearn.log_model(
                sk_model=inference_pipeline,
                name="model",
                serialization_format="cloudpickle",
                input_example=test_sample
            )
        except Exception as e:
            print(f"Logging model via mlflow.sklearn raised ({e}), logging joblib artifact directly...")
            mlflow.log_artifact(model_output_path, artifact_path="model")

        mlflow.log_artifact(metrics_output_path)
        print(f"MLflow Run completed. Run ID: {run.info.run_id}, Experiment: {experiment_name}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": model_output_path,
        "metrics_path": metrics_output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train house price prediction model.")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")
    args = parser.parse_args()

    train_and_evaluate(
        params_path=args.config,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()
