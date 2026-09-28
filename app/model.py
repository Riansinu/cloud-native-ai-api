"""ML model loading and management."""

import logging
import os
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# Model metadata
MODEL_NAME = "California Housing Price Predictor"
MODEL_VERSION = "1.0.0"
MODEL_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
MODEL_DESCRIPTION = (
    "Gradient Boosting Regressor trained on the California Housing dataset. "
    "Predicts median house value in hundreds of thousands of dollars."
)

# Resolve model path from environment or default
MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models"))
MODEL_PATH = Path(MODEL_DIR) / "model.joblib"

# Global model reference
_model = None


def load_model() -> None:
    """Load the trained model from disk into memory."""
    global _model

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run 'python ml/train.py' first to train the model."
        )

    logger.info("Loading model from %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully: %s v%s", MODEL_NAME, MODEL_VERSION)


def get_model():
    """Return the loaded model instance.

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    return _model
