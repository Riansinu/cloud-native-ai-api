"""Training script for the California Housing price prediction model.

Usage:
    python ml/train.py

The script:
1. Loads the California Housing dataset from scikit-learn.
2. Splits it into training and test sets.
3. Trains a Gradient Boosting Regressor.
4. Evaluates the model and prints metrics.
5. Saves the trained model to models/model.joblib.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Resolve project root so the script works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def train() -> None:
    """Train and save the model."""
    print("=" * 60)
    print("  California Housing — Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    print(f"      Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"      Features: {feature_names}")

    # ------------------------------------------------------------------
    # 2. Split data
    # ------------------------------------------------------------------
    print("\n[2/5] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Training samples: {X_train.shape[0]}")
    print(f"      Test samples:     {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Train model
    # ------------------------------------------------------------------
    print("\n[3/5] Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("      Training complete.")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"      MAE:  {mae:.4f}")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      R²:   {r2:.4f}")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    print("\n[5/5] Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"      Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
