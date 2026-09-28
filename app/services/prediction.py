"""Prediction service — business logic for making predictions."""

import logging

import numpy as np

from app.model import MODEL_FEATURES, MODEL_NAME, MODEL_VERSION, get_model
from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def make_prediction(request: PredictionRequest) -> PredictionResponse:
    """Run a prediction using the loaded model.

    Args:
        request: Validated prediction input.

    Returns:
        PredictionResponse with the predicted value and model info.
    """
    model = get_model()

    # Build feature array in the correct order
    features = np.array(
        [[getattr(request, feature) for feature in MODEL_FEATURES]]
    )

    logger.info("Running prediction with features: %s", dict(request))
    prediction = model.predict(features)[0]
    predicted_price = round(float(prediction), 4)
    logger.info("Prediction result: %s", predicted_price)

    return PredictionResponse(
        predicted_price=predicted_price,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
