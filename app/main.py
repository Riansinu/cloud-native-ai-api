"""Cloud-Native AI Prediction API — FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import (
    MODEL_DESCRIPTION,
    MODEL_FEATURES,
    MODEL_NAME,
    MODEL_VERSION,
    load_model,
)
from app.schemas import (
    APIInfoResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction import make_prediction

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model when the application starts."""
    logger.info("Starting Cloud-Native AI Prediction API")
    try:
        load_model()
        logger.info("Application startup complete")
    except FileNotFoundError as exc:
        logger.error("Failed to load model: %s", exc)
        raise
    yield
    logger.info("Shutting down application")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cloud-Native AI Prediction API",
    description=(
        "A production-style REST API serving machine-learning predictions. "
        "Built with FastAPI, Docker, Nginx, and GitHub Actions CI/CD."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=APIInfoResponse, tags=["General"])
async def root() -> APIInfoResponse:
    """Return basic information about the API."""
    return APIInfoResponse(
        name="Cloud-Native AI Prediction API",
        version="1.0.0",
        description="REST API for California Housing price predictions",
        docs_url="/docs",
    )


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    """Return the health status of the service."""
    return HealthResponse(status="healthy")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Return information about the loaded ML model."""
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        features=MODEL_FEATURES,
        description=MODEL_DESCRIPTION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a price prediction from the input features.

    The model predicts the median house value in **hundreds of thousands of
    dollars** for a California census block group.
    """
    try:
        result = make_prediction(request)
    except RuntimeError as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error")
    return result
